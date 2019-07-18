import argparse
import logging
import logging.handlers
import os
import threading
import time
from datetime import datetime
from typing import Generator, Iterable, List, Optional, Tuple, Union

import numpy
import torch
from inferno.io.transform import Compose
from torch import multiprocessing as mp

from tiktorch.configkeys import DIRECTORY, LOGGING, TESTING, TRANSFORMS
from tiktorch.rpc import Client, RPCFuture, Server, Shutdown, TCPConnConf
from tiktorch.rpc.mp import MPClient, create_client
from tiktorch.rpc_interface import IFlightControl, INeuralNetworkAPI
from tiktorch.tiktypes import LabeledTikTensor, LabeledTikTensorBatch, TikTensor, TikTensorBatch
from tiktorch.types import (
    LabeledNDArray,
    LabeledNDArrayBatch,
    Model,
    ModelState,
    NDArray,
    NDArrayBatch,
    SetDeviceReturnType,
)
from tiktorch.utils import convert_to_SetDeviceReturnType, get_error_msg_for_incomplete_config

from .handler import IHandler
from .handler import run as run_handler
from .utils import get_transform

mp.set_start_method("spawn", force=True)

logging.basicConfig(level=logging.INFO)

KILL_TIMEOUT = 60  # seconds


class Watchdog:
    def __init__(self, client, kill_timeout: int = KILL_TIMEOUT):
        self._client = client
        self._kill_timeout = kill_timeout
        self._stop_evt = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self.run, name="TikTorchServer.Watchdog")
        self._thread.daemon = True
        self._thread.start()

    def run(self):
        while not self._stop_evt.wait(timeout=self._kill_timeout):
            ts = self._client.last_ping()
            if ts and time.time() - ts > self._kill_timeout:
                try:
                    self._client.shutdown()
                finally:
                    break

    def stop(self):
        self._stop_evt.set()
        self._thread.join()


class TikTorchServer(INeuralNetworkAPI, IFlightControl):
    RANK = 1
    SIZE = 2

    def __init__(self):
        self.logger = logging.getLogger("tiktorch.server")
        self._log_listener: Optional[logging.handlers.QueueListener] = None
        self._handler: Optional[IHandler] = None
        self._last_ping = 0

    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, new_handler: IHandler):
        if self._handler is not None:
            self._handler.shutdown()

        self._handler: IHandler = new_handler

    @staticmethod
    def get_cuda_and_handler_device_names(devices: Iterable[str]) -> Tuple[List[str], List[str]]:
        add_cpu = False
        cuda_devices = []
        for d in devices:
            d = d.lower()
            if d == "cpu":
                add_cpu = True
            elif ":" in d:
                base, index = d.split(":")

                if base == "cpu":
                    if index not in ["0", "-1"]:
                        raise ValueError(f"Invalid index '{index}' in device name '{d}'")

                    add_cpu = True
                elif base not in ["gpu", "cuda"]:
                    raise ValueError(f"Invalid base name '{base}' in device name '{d}'")

                if not index:
                    raise ValueError(f"device name '{d}' is missing index after the colon")

                if not index.isdigit():
                    raise ValueError(f"device name '{d}' contains non-numeral index after the colon")

                cuda_devices.append(index)
            else:
                raise ValueError(f"device name '{d}' does not specify an index")

        # Pytorch uses CUDA device order, which means devices are ordered computing power.
        # We want to ensure the same order here:
        cuda_devices.sort()
        # When limiting CUDA_VISIBLE_DEVICES, indices might be missing,
        # e.g. CUDA_VISIBLE_DEVICES="2,3" => available torch.devices: "cuda:0", "cuda:1",
        # corresponding to "cuda:2" and "cuda:3", respectively, if CUDA_VISIBLE_DEVICES was not set.
        handler_devices = [f"cuda:{i}" for i in range(len(cuda_devices))]
        if add_cpu:
            handler_devices.append(("cpu"))

        return cuda_devices, handler_devices

    def is_valid_device_name(self, device_name: str):
        try:
            self.get_cuda_and_handler_device_names([device_name])
        except ValueError:
            return False
        else:
            return True

    def _start_logging_handler(self):
        self.log_queue = mp.Queue()
        root_logger = logging.getLogger()
        if self._log_listener is not None:
            self._log_listener.stop()
        self._log_listener = logging.handlers.QueueListener(self.log_queue, *root_logger.handlers)
        self._log_listener.start()

    def log(self, msg: str) -> None:
        self.logger.debug(msg)

    def get_available_devices(self) -> List[Tuple[str, str]]:
        available = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                vram = prop.total_memory / 1.074e9  # vram in Gibibytes
                available.append((vram, i, prop.name))

            available = sorted(available, reverse=True)

            available = [(f"cuda:{a[1]}", f"{a[2]} ({a[0]:.2f}GB)") for a in available]

        available.append(("cpu", "CPU"))
        return available

    def load_model(self, model: Model, state: ModelState, devices: list) -> RPCFuture[SetDeviceReturnType]:
        log_dir = model.config.get(LOGGING, {}).get(DIRECTORY, "")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.logger.info("log dir: %s", os.path.abspath(log_dir))

        self._start_logging_handler()
        incomplete_msg = get_error_msg_for_incomplete_config(model.config)
        if incomplete_msg:
            raise ValueError(incomplete_msg)

        # todo: move test_transforms elsewhere
        self.test_transforms = model.config.get(TESTING, {}).get(TRANSFORMS, {"Normalize": {}})

        if not devices:
            devices = ["cpu"]

        cuda_visible_devices, handler_devices = self.get_cuda_and_handler_device_names(devices)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)
        self.logger.info("Set CUDA_VISIBLE_DEVICES to '%s'", os.environ["CUDA_VISIBLE_DEVICES"])

        server_conn, handler_conn = mp.Pipe()
        p = mp.Process(
            target=run_handler,
            name="Handler",
            kwargs={
                "conn": handler_conn,
                "config": model.config,
                "model_file": model.code,
                "model_state": state.model_state,
                "optimizer_state": state.optimizer_state,
                "log_queue": self.log_queue,
            },
        )
        try:
            p.start()
        except Exception as e:
            self.logger.error(e)
            err_fut = RPCFuture()
            err_fut.set_exception(e)
            return err_fut
        else:
            self.handler = create_client(IHandler, server_conn)
            try:
                tik_fut = self.handler.set_devices(handler_devices)
            except Exception as e:
                self.logger.exception("set_devices failed")
                err_fut = RPCFuture()
                err_fut.set_exception(e)
                return err_fut
            else:
                self.logger.info("got tik_fut")
                fut = tik_fut.map(convert_to_SetDeviceReturnType)
                self.logger.info("converted tik_fut")
                return fut

    def active_children(self) -> List[str]:
        return [c.name for c in mp.active_children()]

    def forward(self, image: NDArray) -> RPCFuture[NDArray]:
        # todo: do transform in separate thread
        transform = Compose(*[get_transform(name, **kwargs) for name, kwargs in self.test_transforms.items()])
        return self.handler.forward(
            data=TikTensor(transform(image.as_numpy()).astype(numpy.float32), id_=image.id)
        ).map(lambda val: NDArray(val.as_numpy()))

    def update_training_data(self, data: NDArrayBatch, labels: NDArrayBatch) -> None:
        self.handler.update_training_data(TikTensorBatch(data), TikTensorBatch(labels))

    def update_validation_data(self, data: NDArrayBatch, labels: NDArrayBatch) -> None:
        return self.handler.update_validation_data(TikTensorBatch(data), TikTensorBatch(labels))

    def pause_training(self) -> None:
        self.handler.pause_training()

    def resume_training(self) -> None:
        self.handler.resume_training()

    def update_config(self, partial_config: dict) -> None:
        self.handler.update_config(partial_config)

    def ping(self) -> bytes:
        self._last_ping = time.time()
        return b"pong"

    def last_ping(self) -> Optional[float]:
        return self._last_ping

    def get_model_state(self) -> ModelState:
        return self.handler.get_state()

    def shutdown(self):
        self.logger.info("Shutting down...")
        if self.handler:
            try:
                res = self.handler.shutdown.async_().result(timeout=20)
            except TimeoutError as e:
                self.logger.error(e)

            self._log_listener.stop()

        raise Shutdown


class ServerProcess:
    def __init__(self, address: str, port: str, notify_port: str, kill_timeout: int):
        self._addr = address
        self._port = port
        self._notify_port = notify_port
        self._kill_timeout = kill_timeout

    def listen(self, provider_cls: INeuralNetworkAPI = TikTorchServer):
        api_provider = provider_cls()

        conf = TCPConnConf(self._addr, self._port, self._notify_port)
        srv = Server(api_provider, conf)
        client = Client(IFlightControl(), conf)

        watchdog = Watchdog(client, self._kill_timeout)
        watchdog.start()

        srv.listen()
