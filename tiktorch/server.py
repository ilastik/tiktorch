import argparse
import logging
import logging.handlers
import torch
import os

from torch import multiprocessing as mp

from typing import Optional, List, Tuple, Generator, Iterable

from tiktorch.rpc import Server, Shutdown, TCPConnConf, RPCFuture
from tiktorch.rpc.mp import MPClient
from tiktorch.types import NDArrayBatch
from tiktorch.tiktypes import TikTensorBatch
from tiktorch.handler import IHandler, run as run_handler
from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl


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
        self._handler: Optional[MPClient] = None
        self.state = State()

    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, new_handler):
        if self._handler is not None:
            self._handler.shutdown()

        self._handler = new_handler

    @staticmethod
    def get_cuda_and_handler_device_names(devices: Iterable[str]) -> Generator[Tuple[Optional[str], str], None, None]:
        handler_device_index = -1
        for d in devices:
            d = d.lower()
            if d == 'cpu':
                yield None, 'cpu'
            elif ':' in d:
                base, index = d.split(':')

                if base == 'cpu':
                    if index not in ['0', '-1']:
                        raise ValueError(f"Invalid index '{index}' in device name '{d}'")

                    yield None, 'cpu'
                elif base not in ['gpu', 'cuda']:
                    raise ValueError(f"Invalid base name '{base}' in device name '{d}'")

                if not index:
                    raise ValueError(f"device name '{d}' is missing index after the colon")

                if not index.isdigit():
                    raise ValueError(f"device name '{d}' contains non-numeral index after the colon")

                handler_device_index += 1
                yield index, f"cuda:{handler_device_index}"
            else:
                raise ValueError(f"device name '{d}' does not specify an index")

    def is_valid_device_name(self, device_name: str):
        try:
            for _ in self.get_cuda_and_handler_device_names([device_name]):
                pass
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

    def load_model(self, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes, devices: list) -> None:
        if not devices:
            devices = ['cpu']

        cuda_visible_devices: List[str] = []
        handler_devices: List[str] = []
        for cuda_name, handler_name in self.get_cuda_and_handler_device_names(devices):
            if cuda_name:
                cuda_visible_devices.append(cuda_name)

            handler_devices.append(handler_name)

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cuda_visible_devices)
        self.logger.info("Set CUDA_VISIBLE_DEVICES to '%s'", os.environ["CUDA_VISIBLE_DEVICES"])

        self._start_logging_handler()
        server_conn, handler_conn = mp.Pipe()
        p = mp.Process(
            target=run_handler,
            name='Handler',
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
        self.handler = MPClient(IHandler(), server_conn)
        self.handler.set_devices(handler_devices)

    def active_children(self) -> List[str]:
        return [c.name for c in mp.active_children()]

    def forward(self, batch: NDArrayBatch) -> RPCFuture[NDArrayBatch]:
        return self.handler.forward(data=TikTensorBatch(batch))

    def train(self, raw: NDArrayBatch, labels: NDArrayBatch) -> RPCFuture:
        return self.handler.train(TikTensorBatch(raw, labels))

    def pause(self) -> None:
        self.handler.pause_training()

    def resume(self) -> None:
        self.handler.resume_training()

    def set_hparams(self, hparams: dict) -> None:
        self.handler.set_hparams(hparams)

    def request_state(self) -> RPCFuture:
        self.logger.info("Requesting model state dict from handler...")
        return self.handler.get_state()

    def ping(self) -> bytes:
        return b"pong"

    def shutdown(self):
        self.logger.info("Shutting down...")
        if self.handler:
            try:
                self.handler.shutdown().result(timeout=60)
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
