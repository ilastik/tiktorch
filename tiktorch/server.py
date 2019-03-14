import argparse
import logging
import io
import os
from importlib import util as imputils
import zmq

import numpy as np
import torch
from datetime import datetime
import socket
import shutil
import tempfile

import tiktorch.utils as utils
from tiktorch.rpc import Server, Shutdown, TCPConnConf
from tiktorch.device_handler import ModelHandler
from tiktorch.types import NDArray, NDArrayBatch

from .rpc_interface import INeuralNetworkAPI, IFlightControl
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

    def __init__(self, device=None):
        logger = logging.getLogger("TikTorchServer.__init__")
        # Privates
        self._build_directory = None
        self._handler: ModelHandler = None
        self._model = None
        self._log_directory = None
        # Set up queues
        if device is None:
            # The default behaviour is to select a GPU if one is availabe.
            # This can be overriden by providing device in the constructor.
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        logger.info(f"Using device: {self._device}")
        # Publics
        self.ilp_directory = None
        self.state = State()
        # self.init()
        self.tmp_dir = tempfile.mkdtemp()  # todo: get rid of tmp dir!

        self.binary_model_file = None
        self.binary_model_state = None
        self._config = None

    def __del__(self):
        shutil.rmtree(self.tmp_dir)

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
    def output_shape(self):
        return self.get("output_shape")

    @property
    def halo(self):
        """
        Returns the halo in dynamic base shape blocks
        """
        assert self.handler is not None
        halo_block = self.handler.halo_in_blocks
        base_shape = self.handler.dynamic_shape.base_shape
        return [shape * block for shape, block in zip(base_shape, halo_block)]

    @property
    def log_directory(self):
        if self._log_directory is None and self.ilp_directory is not None:
            # Make a log directory in the ilp_directory
            path = os.path.join(
                self.ilp_directory, "TikTorchLogs", f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}"
            )
            os.makedirs(path, exist_ok=True)
            self._log_directory = path
            return self._log_directory
        else:
            return self._log_directory

    @property
    def model(self):
        return self.handler.model

    @property
    def device(self):
        return self.handler.device

    @property
    def handler(self):
        assert self._handler
        return self._handler

    def _set_handler(self, model):
        assert self.get("input_shape") is not None
        # Pass
        self._handler = ModelHandler(
            model=model,
            device_names=self._device,
            channels=self.get("input_shape")[0],
            dynamic_shape_code=self.get("dynamic_input_shape"),
            log_directory=self.log_directory,
        )

    def get(self, tag, default=None, assert_exist=False):
        if assert_exist:
            assert tag in self._config, f"Tag '{tag}' not found in configuration."
        return self._config.get(tag, default)

    @staticmethod
    def define_patched_model(model_file_name, model_class_name, model_init_kwargs):
        # Dynamically import file.
        module_spec = imputils.spec_from_file_location("model", model_file_name)
        module = imputils.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        # Build model from file
        model: torch.nn.Module = getattr(module, model_class_name)(**model_init_kwargs)
        # Monkey patch
        model.__model_file_name = model_file_name
        model.__model_class_name = model_class_name
        model.__model_init_kwargs = model_init_kwargs
        return model

    def load_model(self, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes) -> None:
        # Dynamically import file.
        # hack to keep current impl: safe binary_model_file to tmp file
        # todo: load model.py without tmp directory
        self.binary_model_file = model_file
        self.binary_model_state = model_state
        self._config = config

        tmp_dir = self.tmp_dir
        model_file_path = os.path.join(tmp_dir, "model.py")
        with open(model_file_path, "wb") as f:
            f.write(self.binary_model_file)

        model_state_path = None
        if self.binary_model_state:
            model_state_path = os.path.join(tmp_dir, "state.nn")
            with open(model_state_path, "wb") as f:
                f.write(self.binary_model_state)
                # todo: implement like 'import io; f = io.BytesIO(binary_model_state) torch.load(f) # in same context

        # todo: optimizer state

        model = utils.define_patched_model(model_file_path, self.get("model_class_name"), self.get("model_init_kwargs"))
        # Load parameters
        try:
            state_dict = torch.load(model_state_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except:
            logger.warning(f"state.nn file not found in {model_state_path}, not loading weights!")
            # raise FileNotFoundError(f"Model weights could not be found at location '{state_path}'!")
        # Build handler
        self._set_handler(model)

    def forward(self, batch: NDArrayBatch) -> NDArrayBatch:
        # TODO: Use TikIO for batching
        tensors = [torch.from_numpy(a) for a in batch.as_numpy()]
        res = self.handler.forward(*tensors)
        logger.debug("Send forward result")
        return NDArrayBatch([NDArray(res.numpy())])

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
    def __init__(
        self,
        address: str,
        port: str,
        notify_port: str,
        device=None,
        **kwargs
    ):
        self._addr = address
        self._port = port
        self._notify_port = notify_port
        self._device = device

    def listen(self):
        api_provider = TikTorchServer(device=self._device)
        srv = Server(api_provider, TCPConnConf(
            self._addr, self._port, self._notify_port
        ))
        srv.listen()


if __name__ == "__main__":
    # Output pid for process tracking
    print(os.getpid(), flush=True)

    parsey = argparse.ArgumentParser()
    parsey.add_argument('--addr', type=str, default='127.0.0.1')
    parsey.add_argument('--port', type=str, default='29500')
    parsey.add_argument('--notify-port', type=str, default='29501')
    parsey.add_argument('--debug', type=bool, default=False)

    args = parsey.parse_args()

    logger.info("Starting server on %s:%s", args.addr, args.port)

    srv = ServerProcess(
        address=args.addr,
        port=args.port,
        notify_port=args.notify_port,
    )
    srv = ServerProcess(address=args.addr, port=args.port)
    srv.listen()
