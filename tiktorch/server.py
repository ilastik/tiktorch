import argparse
import logging
import io
import os
import zmq

import torch
from datetime import datetime
import socket
import shutil
import tempfile


from torch import multiprocessing as mp

import tiktorch.utils as utils
from tiktorch.rpc import Server, Shutdown, TCPConnConf, RPCFuture
from tiktorch.rpc.mp import MPServer, MPClient

from tiktorch.types import NDArray, NDArrayBatch
from tiktorch.tiktypes import TikTensor, TikTensorBatch
from tiktorch.handler import IHandler, run as run_handler
from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl
from typing import Iterable


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
            raise Exception(f"Transition from {self._current} to {state} not allowed")

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

    def __init__(self):
        logger = logging.getLogger("TikTorchServer.__init__")
        # Privates
        self._handler: MPClient = None
        self._log_directory = None
        # Publics
        self.state = State()

    def init(self):
        logger = logging.getLogger("TikTorchServer.init")
        logger.info("Setting up ZMQ")
        # Init ZMQ
        logger.info("Setting up ZMQ Context...")
        self._zmq_context = zmq.Context()
        logger.info("Setting up ZMQ Socket...")
        self._zmq_socket = self._zmq_context.socket(zmq.PAIR)
        logger.info("Binding to socket tcp://%s:%s", self.addr, self.port)
        self._zmq_socket.bind(f"tcp://{self.addr}:{self.port}")
        logger.info("Waiting for init data...")

    @property
    def log_directory(self):
        return self._log_directory

    def load_model(self, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes):
        if self._handler is not None:
            pass

        server_conn, handler_conn = mp.Pipe()
        p = mp.Process(
            target=run_handler,
            kwargs={
                "conn": handler_conn,
                "config": config,
                "model_file": model_file,
                "model_state": model_state,
                "optimizer_state": optimizer_state,
            },
        )
        p.start()
        self._handler = MPClient(IHandler(), server_conn)


    @property
    def handler(self):
        assert self._handler
        return self._handler

    def get(self, tag, default=None, assert_exist=False):
        if assert_exist:
            assert tag in self._config, f"Tag '{tag}' not found in configuration."
        return self._config.get(tag, default)

    def forward(self, batch: NDArrayBatch) -> RPCFuture[NDArrayBatch]:
        fut = self.handler.forward(data=TikTensorBatch(batch))
        return RPCFuture(fut)  # todo: fut -> rpcfut?

    def train(self, keys: Iterable, data: NDArrayBatch, labels: NDArrayBatch) -> None:
        torch_data = [torch.from_numpy(arr) for arr in data.as_numpy()]
        torch_labels = [torch.from_numpy(arr) for arr in labels.as_numpy()]
        logger.info("Received data and labels from chief.")
        logger.info("Sending to handler.")
        self.handler.train(torch_data, torch_labels)
        logger.info("Sent to handler.")

    def training_process_is_running(self) -> bool:
        return self.handler.training_process_is_alive()

    def pause(self) -> None:
        self.handler.pause_training()

    def resume(self) -> None:
        self.handler.resume_training()

    def set_hparams(self, params: dict) -> None:
        logger.info("Sending to handler.")
        self.handler.set_hparams(params)
        logger.info("Sent to handler.")

    def request_model_state_dict(self):
        logger = logging.getLogger("TikTorchServer.request_model_state_dict")
        logger.info("Requesting model state dict from handler....")
        self.handler.update_state()
        state_dict = io.BytesIO()
        torch.save(self.model.state_dict(), f=state_dict)
        logger.info("Sending state dict.")
        self._zmq_socket.send(state_dict.getvalue())
        logger.info("Sent state dict.")
        state_dict.close()

    def request_optimizer_state_dict(self):
        pass

    def dry_run(self, conf: dict) -> dict:
        assert "train" in conf
        assert "upper_bound" in conf
        logger = logging.getLogger("TikTorchServer.dry_run")
        logger.info("Initiating dry run...")
        valid_shape = self.handler.dry_run(conf["upper_bound"], conf["train"])
        return {"shape": valid_shape}

    def poll_training_process(self):
        logger = logging.getLogger("TikTorchServer.poll_training_process")
        logger.info("Polling...")
        # Check if training process is running, and send info back
        it_lives = self.handler.training_process_is_alive()
        logger.info("Poll successful. Sending response...")
        info = {"id": "POLL_TRAIN.INFO", "is_alive": it_lives}
        self.meta_send(info)
        logger.info("Poll response sent.")

    def ping(self) -> bytes:
        return b"pong"

    def shutdown(self):
        logger = logging.getLogger("TikTorchServer.shutdown")
        logger.info("Stopping training...")
        if self._handler:
            self._handler.stop_training()
        logger.info("Training has stopped")
        raise Shutdown()


class ServerProcess:
    def __init__(self, address: str, port: str, notify_port: str, device=None, **kwargs):
        self._addr = address
        self._port = port
        self._notify_port = notify_port
        self._device = device

    def listen(self):
        api_provider = TikTorchServer(device=self._device)
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
    srv = ServerProcess(address=args.addr, port=args.port)
    srv.listen()
