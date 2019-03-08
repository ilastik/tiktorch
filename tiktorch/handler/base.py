import io
import logging
import logging.config
import os.path
import pickle
import tempfile
import torch

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

# logging.basicConfig(level=logging.DEBUG)
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'default': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True
        },
    }
})



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
        self.model_file = model_file  # to be deserialized in 'load_model'
        self.model_state = model_state  # to be deserialized in 'load_model'
        self.optimizer_state = optimizer_state  # to be deserialized in training process

    def load_model(self):
        with tempfile.TemporaryDirectory() as tempdir:
            module_name = "usermodel.py"
            with open(os.path.join(tempdir, module_name), "w") as f:
                f.write(pickle.loads(self.model_file))

            module_spec = spec_from_file_location(module_name.replace(".py", ""), os.path.join(tempdir, module_name))
            module = module_from_spec(module_spec)
            module_spec.loader.exec_module(module)

        self.training_model: torch.nn.Module = getattr(module, self.config["model_class_name"])(
            **self.config.get("model_init_kwargs", {})
        )
        self.inference_model: torch.nn.Module = getattr(module, self.config["model_class_name"])(
            **self.config.get("model_init_kwargs", {})
        )

        if self.model_state:
            with closing(io.BytesIO(self.model_state)) as f:
                self.training_model.load_state_dict(torch.load(f))

    def run(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting")
        self._shutting_down = False
        self.load_model()
        # set up training process
        self.training_conn, handler_conn_training = Pipe()
        self.training_proc = TrainingProcess(
            handler_conn=handler_conn_training,
            config=self.config,
            model=self.training_model,
            optimizer_state=self.optimizer_state,
        )
        self.training_proc.start()
        self.inference_conn, handler_conn_inference = Pipe()
        self.inference_proc = InferenceProcess(handler_conn=handler_conn_inference, model=self.inference_model)
        self.inference_proc.start()

        try:
            while True:
                for ready in wait([self.server_conn, self.inference_conn, self.training_conn]):
                    call, kwargs = ready.recv()
                    meth = getattr(self, call, None)
                    if meth is None:
                        raise NotImplementedError(call)

                    meth(**kwargs)
        finally:
            self.shutdown()

    # general
    def shutdown(self):
        self._shutting_down = True
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
    def update_inference_model(self):
        with io.BytesIO() as bytes_io:
            torch.save(self.training_model.state_dict(), bytes_io)
            bytes_io.seek(0)
            self.inference_model.load_state_dict(torch.load(bytes_io))

        self.inference_model.eval()

    def forward(self, keys: Iterable, data: torch.Tensor) -> None:
        self.logger.debug('forward')
        self.update_inference_model()
        self.inference_conn.send(("forward", {"keys": keys, "data": data}))

    def forward_answer(self, keys: Iterable, data: torch.Tensor) -> None:
        self.logger.debug('forward_answer')
        if not self._shutting_down:
            self.server_conn.send((self.forward_answer.__name__, {"keys": keys, "data": data}))

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
