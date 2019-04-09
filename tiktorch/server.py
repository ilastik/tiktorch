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
from tiktorch.tiktypes import (
    TikTensor,
    LabeledTikTensor,
    TikTensorBatch,
    LabeledTikTensorBatch,
    Point2D,
    Point3D,
    Point4D,
)
from tiktorch.handler import IHandler, run as run_handler
from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl
from tiktorch.utils import (
    convert_tik_fut_to_ndarray_fut,
    convert_to_SetDeviceReturnType,
    get_error_msg_for_invalid_config,
    get_error_msg_for_incomplete_config,
)
from tiktorch.configkeys import MINIMAL_CONFIG

if torch.cuda.is_available():
    torch.multiprocessing.set_start_method("spawn", force=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TikTorchServer")


from functools import wraps


class State:
    INIT = "init"
    WORKING = "working"

    STATES = {INIT: [WORKING], WORKING: [INIT]}

    def __init__(self):
        self.current = self.INIT

    def next(self, state):
        allowed = self.STATES.get(self.current)
        if state not in allowed:
            raise Exception(f"Transition from {self.current} to {state} not allowed")

    def __repr__(self):
        return f"State({self.current})"


def expect_state(*allowed):
    def decorator(func):
        # return func

        @wraps(func)
        def wrapped(self, *args, **kwargs):
            if self.state.current not in allowed:
                raise Exception(f"Call to {func} not allowed in {self.state}")

            func(*args, **kwargs)

        return wrapped

    return decorator


class TikTorchServer(INeuralNetworkAPI, IFlightControl):
    RANK = 1
    SIZE = 2

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._log_listener: Optional[logging.handlers.QueueListener] = None
        self._handler: Optional[IHandler] = None
        self.state = State()

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
        p.start()
        self.handler = create_client(IHandler, server_conn)
        return convert_to_SetDeviceReturnType(self.handler.set_devices(handler_devices))

    def active_children(self) -> List[str]:
        return [c.name for c in mp.active_children()]

    def forward(self, batch: NDArray) -> RPCFuture[NDArray]:
        return self.handler.forward(data=TikTensor(batch)).map(lambda val: NDArray(val.as_numpy()))

    def update_training_data(self, data: LabeledNDArrayBatch) -> None:
        return self.handler.update_training_data(LabeledTikTensorBatch(data))

    def update_validation_data(self, data: LabeledNDArrayBatch) -> None:
        return self.handler.update_validation_data(LabeledTikTensorBatch(data))

    def pause_training(self) -> None:
        self.handler.pause_training()

    def resume_training(self) -> None:
        self.handler.resume_training()

    def update_config(self, partial_config: dict) -> None:
        self.handler.update_config(partial_config)

    def request_state(self) -> RPCFuture:
        self.logger.info("Requesting model state dict from handler...")
        return self.handler.get_state()

    def ping(self) -> bytes:
        return b"pong"

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

    def listen(self):
        api_provider = TikTorchServer()
        srv = Server(api_provider, TCPConnConf(self._addr, self._port, self._notify_port))
        srv.listen()


if __name__ == "__main__":
    # Output pid for process tracking
    print(os.getpid(), flush=True)

    parsey = argparse.ArgumentParser()
    parsey.add_argument("--addr", type=str, default="127.0.0.1")
    parsey.add_argument("--port", type=str, default="29500")
    parsey.add_argument("--notify-port", type=str, default="29501")
    parsey.add_argument("--debug", type=bool, default=False)

    args = parsey.parse_args()

    logger.info("Starting server on %s:%s", args.addr, args.port)

    srv = ServerProcess(address=args.addr, port=args.port, notify_port=args.notify_port)
    srv.listen()
