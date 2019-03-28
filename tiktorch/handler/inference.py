import logging
import os
import queue
import torch.nn
import threading
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor, Future
from multiprocessing.connection import Connection
from typing import Any, List, Generic, Iterator, Iterable, Sequence, TypeVar, Mapping, Callable, Dict, Optional, Tuple, Set, Collection

from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION, REQUEST_FOR_DEVICES
from tiktorch.rpc import RPCInterface, exposed, Shutdown, RPCFuture
from tiktorch.rpc.mp import MPServer
from tiktorch.tiktypes import TikTensor, TikTensorBatch
from tiktorch import log


class IInference(RPCInterface):
    @exposed
    def set_devices(self, device_names: Sequence[str]):
        raise NotImplementedError()

    @exposed
    def shutdown(self):
        raise NotImplementedError()

    @exposed
    def forward(self, data: TikTensorBatch):
        raise NotImplementedError()


def run(conn: Connection, config: dict, model: torch.nn.Module, log_queue: Optional[mp.Queue] = None):
    log.configure(log_queue)
    inference_proc = InferenceProcess(config, model)
    srv = MPServer(inference_proc, conn)
    srv.listen()


class InferenceProcess(IInference):
    """
    Process for neural network inference
    """

    name = "tiktorch.InferenceProcess"

    def __init__(self, config: dict, model: torch.nn.Module) -> None:
        self.logger = logging.getLogger(self.name)
        self.logger.info("Starting")
        self.config = config
        self.training_model = model

        self.shutdown_event = threading.Event()

        self.batch_size: int = config.get("inference_batch_size", None)
        self.forward_queue = queue.Queue()
        self.shutdown_worker_events = {}
        self.forward_worker_threads = {}
        self.devices = set()
        self.add_devices({torch.device("cpu")})

        self.device_change_queue = queue.Queue()
        self.device_setter_thread = threading.Thread(target=self._device_setter_worker)
        self.device_setter_thread.start()

    def _device_setter_worker(self):
        while not self.shutdown_event.is_set():
            try:
                devices, fut = self.device_change_queue.get(block=True, timeout=5)
                self.add_devices(devices - self.devices)
                remove = self.devices - devices
                self.remove_devices(remove)
                fut.set_result(remove)
            except queue.Empty:
                pass

        while not self.device_change_queue.empty():
            devices, fut = self.device_change_queue.get(block=True, timeout=5)
            fut.set_result(set())

    def remove_devices(self, devices: Set[torch.device]) -> None:
        """
        :param devices: devices to remove
        :return: set of device threads that are shutting down (wait on them with '.join()')
        """
        for d in devices:
            assert d in self.devices
            assert d in self.shutdown_worker_events
            assert d in self.forward_worker_threads
            self.shutdown_worker_events[d].set()


        for d in devices:
            self.forward_worker_threads[d].join()
            del self.forward_worker_threads[d]
            del self.shutdown_worker_events[d]

        self.devices -= devices

    def add_devices(self, devices: Set[torch.device]) -> None:
        for d in devices:
            assert d not in self.devices
            assert d not in self.shutdown_worker_events
            assert d not in self.forward_worker_threads
            self.shutdown_worker_events[d] = threading.Event()
            self.forward_worker_threads[d] = threading.Thread(target=self._forward_worker, kwargs={"device": d})
            self.forward_worker_threads[d].start()

    def _forward_worker(self, device: torch.device) -> None:
        local_data = threading.local()
        local_data.increase_batch_size = True
        if self.batch_size is None:
            local_data.batch_size = 1
        else:
            local_data.batch_size = self.batch_size

        while not self.shutdown_worker_events[device].is_set() and not self.shutdown_event.is_set():
            data_batch, fut_batch = [], []
            while not self.forward_queue.empty() and len(data_batch) < local_data.batch_size:
                data, fut = self.forward_queue.get()
                data_batch.append(data)
                fut_batch.append(fut)

            if data_batch:
                local_data.batch_size, local_data.increase_batch_size = self._forward(
                    TikTensorBatch(data_batch),
                    fut_batch,
                    device,
                    local_data.batch_size,
                    local_data.increase_batch_size,
                )


    def set_devices(self, devices: Collection[torch.device]) -> RPCFuture[Set[torch.device]]:
        fut = RPCFuture()
        self.device_change_queue.put((set(devices), fut))
        return fut

    def shutdown(self) -> None:
        self.shutdown_event.set()
        self.device_setter_thread.join()
        for ft in self.forward_worker_threads.values():
            ft.join()
        self.logger.debug("Shutdown complete")
        raise Shutdown

    def forward(self, data: TikTensor) -> Future:
        fut = Future()
        self.forward_queue.put((data, fut))
        return fut

    def _forward(
        self, data: TikTensorBatch, fut: List[Future], device: torch.device, batch_size: int, increase_batch_size: bool
    ) -> Tuple[int, bool]:
        """
        :param data: input data to neural network
        """
        if device.type == 'cuda':
            with device:
                model = self.training_model.__class__()
        else:
            model = self.training_model.__class__()

        model.load_state_dict(self.training_model.state_dict())
        model.eval()

        keys: List = [d.id for d in data]
        data: List[torch.Tensor] = data.as_torch()

        # TODO: fixT        return data

        self.logger.debug("this is forward")

        start = 0
        last_batch_size = batch_size

        def create_end_generator(start, end, batch_size):
            for batch_end in range(start + batch_size, end, batch_size):
                yield batch_end

            yield end

        end_generator = create_end_generator(start, len(keys), batch_size)
        while start < len(keys):
            # todo: callback
            end = next(end_generator)
            try:
                with torch.no_grad():
                    pred = model(torch.stack(data[start:end])).cpu()
            except Exception as e:
                if batch_size > last_batch_size:
                    self.logger.info(
                        "forward pass with batch size %d threw exception %s. Using previous batch size %d again.",
                        batch_size,
                        e,
                        last_batch_size,
                    )
                    batch_size = last_batch_size
                    increase_batch_size = False
                else:
                    last_batch_size = batch_size
                    batch_size //= 2
                    if batch_size == 0:
                        self.logger.error("Forward pass failed. Processed %d/%d", start, len(keys))
                        for i in range(start, len(keys)):
                            fut[i].set_exception(e)

                        break

                    increase_batch_size = True
                    self.logger.info(
                        "forward pass with batch size %d threw exception %s. Trying again with smaller batch_size %d",
                        last_batch_size,
                        e,
                        batch_size,
                    )
                end_generator = create_end_generator(start, len(keys), batch_size)
            else:
                for i in range(start, end):
                    fut[i].set_result(TikTensor(pred[i], id_=keys[i]))
                start = end
                last_batch_size = batch_size
                if increase_batch_size:
                    batch_size += 1
                    end_generator = create_end_generator(start, len(keys), batch_size)

        return batch_size, increase_batch_size
