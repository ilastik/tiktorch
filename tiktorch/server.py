import argparse
import logging
import logging.handlers
import torch
import os

from torch import multiprocessing as mp

from typing import Optional, List

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
        return func

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

    def __init__(self, devices: Optional[List[str]] = None):

        self.logger = logging.getLogger(__name__)
        self.log_queue = mp.Queue()
        self.handler: Optional[MPClient] = None
        self.state = State()

    def _start_logging_handler(self):
        root_logger = logging.getLogger()
        self._log_listener = logging.handlers.QueueHandler(self.log_queue, *root_logger.handlers)
        self._log_listener.start()

    def load_model(self, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes) -> None:
        if self.handler is not None:
            self.handler.shutdown()

        server_conn, handler_conn = mp.Pipe()
        p = mp.Process(
            target=run_handler,
            kwargs={
                "conn": handler_conn,
                "config": config,
                "model_file": model_file,
                "model_state": model_state,
                "optimizer_state": optimizer_state,
                "log_queue": mp.Queue(),
            },
        )
        p.start()
        self._start_logging_handler()
        self.handler = MPClient(IHandler(), server_conn)

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
            self.handler.shutdown()
        self._log_listener.stop()

        raise Shutdown()


class ServerProcess:
    def __init__(self, address: str, port: str, notify_port: str, devices: Optional[List[str]] = None, **kwargs):
        self._addr = address
        self._port = port
        self._notify_port = notify_port
        self._devices = devices

    def listen(self):
        api_provider = TikTorchServer(devices=self._devices)
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
