import io
import logging
import os.path
import pickle
import tempfile
import torch.nn

from contextlib import closing
from importlib.util import spec_from_file_location, module_from_spec
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from torch.multiprocessing import Process, Pipe

from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from ..types import NDArrayBatch
from .constants import SHUTDOWN, SHUTDOWN_ANSWER
from .training import TrainingProcess
from .inference import InferenceProcess

logger = logging.getLogger(__name__)


class HandlerProcess(Process):
    """
    Process to orchestrate the interplay of training/validation and inference
    """

    def __init__(
        self, server_conn: Connection, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes
    ):
        """
        :param server_conn: Connection to communicate with server
        :param config: configuration dict
        :param model_file: bytes of file describing the neural network model
        :param model_state: binarized model state dict
        :param optimizer_state: binarized optimizer state dict
        """
        assert model_file
        for required in ["model_class_name"]:
            if required not in config:
                raise ValueError(f"{required} missing in config")

        super().__init__(name="HandlerProcess")
        self.server_conn = server_conn
        self.config = config

        with tempfile.TemporaryDirectory() as tempdir:
            module_name = "model.py"
            with open(os.path.join(tempdir, module_name), "w") as f:
                f.write(pickle.loads(model_file))

            module_spec = spec_from_file_location(module_name, os.path.join(tempdir, module_name))
            module = module_from_spec(module_spec)
            module_spec.loader.exec_module(module)

        self.model: torch.nn.Module = getattr(module, config["model_class_name"])(**config.get("model_init_kwargs", {}))
        # model._model_class_name = config['model_class_name']
        # model._model_init_kwargs = config['model_init_kwargs']

        if model_state:
            with closing(io.BytesIO(model_state)) as f:
                self.model.load_state_dict(torch.load(f))

        self.optimizer_state = optimizer_state  # to be deserialized in training process

    def run(self):
        # set up training process
        self.training_conn, handler_conn_training = Pipe()
        self.training_proc = TrainingProcess(
            handler_conn=handler_conn_training,
            config=self.config,
            model=self.model,
            optimizer_state=self.optimizer_state,
        )
        self.inference_conn, handler_conn_inference = Pipe()
        self.inference_proc = InferenceProcess(handler_conn=handler_conn_inference, model=self.model)

        try:
            while True:
                for call, kwargs in wait([self.server_conn, self.inference_conn, self.training_conn]):
                    meth = getattr(self, call, None)
                    if meth is None:
                        raise NotImplementedError(call)

                    meth(**kwargs)
        finally:
            self.shutdown()

    # general
    def shutdown(self):
        self.inference_conn.send((SHUTDOWN, {}))
        self.training_conn.send((SHUTDOWN, {}))

        shutdown_time = 60

        def shutdown_child(conn, proc):
            while proc.is_alive():
                # look for shutting down answer
                if conn.poll(timeout=2):
                    call, kwargs = conn.recv()
                    if call == SHUTDOWN_ANSWER:
                        # give child process extra time to shutdown
                        proc.join(timeout=shutdown_time)
                        continue

                logger.error(
                    "Failed to shutdown %s gracefully in %d seconds. Sending kill...", proc.name, shutdown_time
                )
                proc.kill()

            logger.debug("%s has shutdown", proc.name)

        shutdown_child(self.inference_conn, self.inference_proc)
        shutdown_child(self.training_conn, self.training_proc)

    def update_hparams(self, hparams: dict):
        pass

    # inference
    def forward(self, data: NDArrayBatch):
        raise NotImplementedError()

    # training
    def resume_train(self):
        pass

    def pause_train(self):
        pass

    def update_training_dataset(self):
        pass

    def request_state(self):
        # model state
        # optimizer state
        # current config dict
        pass

    # validation
    def update_validation_dataset(self):
        pass

    def validate(self):
        pass
