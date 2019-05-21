import argparse
import logging
import logging.handlers
import torch
import os

from torch import multiprocessing as mp
from typing import Optional, List, Tuple, Generator, Iterable, Union

from tiktorch.rpc import Server, Shutdown, TCPConnConf, RPCFuture
from tiktorch.rpc.mp import MPClient, create_client
from tiktorch.types import NDArray, LabeledNDArray, NDArrayBatch, LabeledNDArrayBatch, SetDeviceReturnType
from tiktorch.tiktypes import TikTensor, LabeledTikTensor, TikTensorBatch, LabeledTikTensorBatch
from tiktorch.handler import IHandler, run as run_handler
from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl
from tiktorch.utils import (
    convert_to_SetDeviceReturnType,
    get_error_msg_for_invalid_config,
    get_error_msg_for_incomplete_config,
)
from tiktorch.configkeys import MINIMAL_CONFIG


mp.set_start_method("spawn", force=True)

logging.basicConfig(level=logging.INFO)


class TikTorchServer(INeuralNetworkAPI, IFlightControl):
    RANK = 1
    SIZE = 2

    def __init__(self):
        self.logger = logging.getLogger("tiktorch.server")
        self._log_listener: Optional[logging.handlers.QueueListener] = None
        self._handler: Optional[IHandler] = None

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

    def load_model(
        self, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes, devices: list
    ) -> RPCFuture[SetDeviceReturnType]:
        self._start_logging_handler()
        incomplete_msg = get_error_msg_for_incomplete_config(config)
        if incomplete_msg:
            raise ValueError(incomplete_msg)

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
                "config": config,
                "model_file": model_file,
                "model_state": model_state,
                "optimizer_state": optimizer_state,
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

    def forward(self, batch: NDArray) -> RPCFuture[NDArray]:
        return self.handler.forward(data=TikTensor(batch)).map(lambda val: NDArray(val.as_numpy()))

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
        return b"pong"

    def get_model_state(self) -> bytes:
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
    def __init__(self, address: str, port: str, notify_port: str):
        self._addr = address
        self._port = port
        self._notify_port = notify_port

    def listen(self, api_provider: Optional[INeuralNetworkAPI] = None):
        if api_provider is None:
            api_provider = TikTorchServer()

        srv = Server(api_provider, TCPConnConf(self._addr, self._port, self._notify_port))
        srv.listen()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # Output pid for process tracking
    print(os.getpid(), flush=True)

    parsey = argparse.ArgumentParser()
    parsey.add_argument("--addr", type=str, default="127.0.0.1")
    parsey.add_argument("--port", type=str, default="29500")
    parsey.add_argument("--notify-port", type=str, default="29501")
    parsey.add_argument("--debug", action="store_true")
    parsey.add_argument("--dummy", action="store_true")

    args = parsey.parse_args()
    logger.info("Starting server on %s:%s", args.addr, args.port)

    srv = ServerProcess(address=args.addr, port=args.port, notify_port=args.notify_port)
    if args.dummy:
        from tiktorch.dev import DummyServerForFrontendDev

        srv.listen(api_provider=DummyServerForFrontendDev())
    else:
        srv.listen()
