import logging
import os
import queue
import torch.nn
import threading
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor, Future
from multiprocessing.connection import Connection
from typing import Any, List, Generic, Iterator, Iterable, Sequence, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION, REQUEST_FOR_DEVICES
from tiktorch.rpc import RPCInterface, exposed, Shutdown
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
        self.model = model.__class__()
        self.model.eval()
        self._shutdown_event = threading.Event()

        self.devices = []
        self._forward_queue = queue.Queue()
        self.batch_size: int = config.get("inference_batch_size", None)
        if self.batch_size is None:
            self.batch_size = 1
            self.increase_batch_size = True
        else:
            self.increase_batch_size = False

        self.forward_thread = threading.Thread(target=self._forward_worker)
        self.forward_thread.start()

    def _forward_worker(self) -> None:
        while not self._shutdown_event.is_set():
            data_batch, fut_batch = [], []
            while not self._forward_queue.empty() and len(data_batch) < self.batch_size:
                data, fut = self._forward_queue.get()
                data_batch.append(data)
                fut_batch.append(fut)

            if data_batch:
                self._forward(TikTensorBatch(data_batch), fut_batch)

    def shutdown(self) -> None:
        self._shutdown_event.set()
        self.forward_thread.join()
        self.logger.debug("Shutdown complete")
        raise Shutdown

    def set_devices(self, device_names: Sequence[str]):
        raise NotImplementedError()
        # todo: with lock
        # torch.cuda.empty_cache()
        # os.environ['CUDA_VISIBLE_DEVICES'] = ...

    def update_inference_model(self):
        self.model.load_state_dict(self.training_model.state_dict())
        assert not self.model.training, "Model switched back to training mode somehow???"

    def forward(self, data: TikTensor) -> Future:
        fut = Future()
        self._forward_queue.put((data, fut))
        return fut

    def _forward(self, data: TikTensorBatch, fut: List[Future]) -> None:
        """
        :param data: input data to neural network
        :return: predictions
        """
        keys: List = [d.id for d in data]
        data: List[torch.Tensor] = data.as_torch()

        # TODO: fixT        return data

        self.logger.debug("this is forward")

        start = 0
        last_batch_size = self.batch_size

        def create_end_generator(start, end, batch_size):
            for batch_end in range(start + batch_size, end, batch_size):
                yield batch_end

            yield end

        end_generator = create_end_generator(start, len(keys), self.batch_size)
        while start < len(keys):
            # todo: callback
            end = next(end_generator)
            self.update_inference_model()
            try:
                with torch.no_grad():
                    pred = self.model(torch.stack(data[start:end]))
            except Exception as e:
                if self.batch_size > last_batch_size:
                    self.logger.info(
                        "forward pass with batch size %d threw exception %s. Using previous batch size %d again.",
                        self.batch_size,
                        e,
                        last_batch_size,
                    )
                    self.batch_size = last_batch_size
                    self.increase_batch_size = False
                else:
                    last_batch_size = self.batch_size
                    self.batch_size //= 2
                    if self.batch_size == 0:
                        self.logger.error("Forward pass failed. Processed %d/%d", start, len(keys))
                        break

                    self.increase_batch_size = True
                    self.logger.info(
                        "forward pass with batch size %d threw exception %s. Trying again with smaller batch_size %d",
                        last_batch_size,
                        e,
                        self.batch_size,
                    )
                end_generator = create_end_generator(start, len(keys), self.batch_size)
            else:
                for i in range(start, end):
                    fut[i].set_result(TikTensor(pred[i], id_=keys[i]))
                start = end
                last_batch_size = self.batch_size
                if self.increase_batch_size:
                    self.batch_size += 1
                    end_generator = create_end_generator(start, len(keys), self.batch_size)
