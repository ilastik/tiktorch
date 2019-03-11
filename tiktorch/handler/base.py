import importlib
import io
import logging
import logging.config
import os.path
import pickle
import shutil
import sys
import tempfile
import torch

from contextlib import closing
from functools import partial
from multiprocessing.connection import Connection, wait
from torch.multiprocessing import Process, Pipe, active_children

from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from ..types import NDArrayBatch
from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION, TRAINING, VALIDATION
from .training import TrainingProcess
from .inference import InferenceProcess

logging.basicConfig(level=logging.DEBUG)
# logging.config.dictConfig({
#     'version': 1,
#     'disable_existing_loggers': False,
#     'handlers': {
#         'default': {
#             'level': 'INFO',
#             'class': 'logging.StreamHandler',
#             'stream': 'ext://sys.stdout',
#         },
#     },
#     'loggers': {
#         '': {
#             'handlers': ['default'],
#             'level': 'DEBUG',
#             'propagate': True
#         },
#     }
# })


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
        assert hasattr(self, SHUTDOWN[0]), "make sure the 'shutdown' method has the correct name"
        assert hasattr(self, REPORT_EXCEPTION), "make sure the 'report_exception' method has the correct name"
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

        devices = self.config.get("devices", default=[])
        self.devices = {"training": [], "inference": [], "idle": devices}

    def load_model(self):
        self.tempdir = tempfile.mkdtemp()
        user_module_name = "usermodel"
        with open(os.path.join(self.tempdir, user_module_name + ".py"), "w") as f:
            f.write(pickle.loads(self.model_file))

        sys.path.insert(0, self.tempdir)
        user_module = importlib.import_module(user_module_name)

        self.training_model: torch.nn.Module = getattr(user_module, self.config["model_class_name"])(
            **self.config.get("model_init_kwargs", {})
        )
        self.inference_model: torch.nn.Module = getattr(user_module, self.config["model_class_name"])(
            **self.config.get("model_init_kwargs", {})
        )
        self.logger.debug("created user models")

        if self.model_state:
            with closing(io.BytesIO(self.model_state)) as f:
                self.training_model.load_state_dict(torch.load(f))

            self.logger.info("restored model state")

    def run(self):
        self._shutting_down = False
        self.logger = logging.getLogger(self.name)
        self.logger.info("Starting")
        try:
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
            self.inference_proc = InferenceProcess(
                handler_conn=handler_conn_inference, config=self.config, model=self.inference_model
            )
            self.inference_proc.start()

            while not self._shutting_down:
                self.logger.debug("loop")
                for ready in wait([self.server_conn, self.inference_conn, self.training_conn]):
                    call, kwargs = ready.recv()
                    self.logger.debug("got call %s", call)
                    meth = getattr(self, call, None)
                    if meth is None:
                        raise NotImplementedError(call)

                    meth(**kwargs)
        except Exception as e:
            self.logger.error(str(e))
            self.server_conn.send(("report_shutdown", {"name": self.name, "exception": e}))
            self.shutdown()

    # general
    def active_children(self):
        self.server_conn.send([child_proc.name for child_proc in active_children()])

    def report_exception(self, proc_name: str, exception: Exception):
        self.logger.error("Received exception report from %s: %s", proc_name, exception)
        if proc_name == TrainingProcess.name:
            # todo: restart training proess
            pass
        elif proc_name == InferenceProcess.name:
            # todo: restart inference process
            pass
        else:
            raise NotImplementedError("Did not expect exception report form %s", proc_name)

    def shutdown(self):
        logger = logging.getLogger(self.name + ".shutdown")
        logger.debug("Shutting down...")
        self._shutting_down = True
        try:
            self.inference_conn.send(SHUTDOWN)
        except Exception as e:
            logger.error(e)
        try:
            self.training_conn.send(SHUTDOWN)
        except Exception as e:
            logger.error(e)

        self.server_conn.send(SHUTDOWN_ANSWER)

        shutdown_time = 20

        def shutdown_child(conn, proc):
            while proc.is_alive():
                # look for shutting down answer
                if conn.poll(timeout=shutdown_time):
                    answer = conn.recv()
                    if answer == SHUTDOWN_ANSWER:
                        # give child process extra time to shutdown
                        proc.join(timeout=shutdown_time)
                else:
                    logger.error("Failed to shutdown %s gracefully. Sending kill...", proc.name)
                    proc.kill()
                    proc.join(timeout=shutdown_time)

            logger.debug("%s has shutdown", proc.name)

        try:
            shutdown_child(self.inference_conn, self.inference_proc)
        except Exception as e:
            logger.error(str(e))
        try:
            shutdown_child(self.training_conn, self.training_proc)
        except Exception as e:
            logger.error(str(e))

        shutil.rmtree(self.tempdir)
        logger.debug("Shutdown complete")

    def update_hparams(self, hparams: dict):
        pass

    def generic_relay_to_server(self, method_name: str, **kwargs):
        if not self._shutting_down:
            self.server_conn.send((method_name, kwargs))

    def __getattr__(self, method_name):
        if method_name in ["forward_answer"]:
            return partial(self.generic_relay_to_server, method_name=method_name)
        else:
            raise AttributeError(method_name)

    def report_idle(self, proc_name):
        # todo: report idle
        if proc_name == TrainingProcess.name:
            pass
        elif proc_name == InferenceProcess.name:
            pass
        else:
            raise NotImplementedError(proc_name)

    # inference
    def update_inference_model(self):
        with io.BytesIO() as bytes_io:
            torch.save(self.training_model.state_dict(), bytes_io)
            bytes_io.seek(0)
            self.inference_model.load_state_dict(torch.load(bytes_io))

        self.inference_model.eval()

    def forward(self, keys: Iterable, data: torch.Tensor) -> None:
        self.logger.debug("forward")
        self.update_inference_model()
        # todo: update inference devices
        self.inference_conn.send(("forward", {"keys": keys, "data": data}))

    # training
    def resume_training(self):
        device = 'cpu'
        if self.devices['training']:
            raise NotImplementedError('gpu training')
            # todo: update training devices
            os.environ['CUDA_VISIBLE_DEVICES'] = self.devices['training']
            device = 'gpu'

        self.training_conn.send((self.resume_training.__name__, {'device': 'cpu'}))

    def pause_training(self):
        self.training_conn.send((self.pause_training.__name__, {}))

    def pause_training_answer(self):
        self.devices['idle'] = self.devices['training']
        self.devices['training'] = []

    def update_training_dataset(self, keys: Iterable, data: torch.Tensor):
        self.training_conn.send(('update_dataset', {'name': TRAINING, 'keys': keys, 'data': data}))

    def request_state(self):
        # model state
        # optimizer state
        # current config dict
        pass

    # validation
    def update_validation_dataset(self, keys: Iterable, data: torch.Tensor):
        self.training_conn.send(('update_dataset', {'name': VALIDATION, 'keys': keys, 'data': data}))

    def validate(self):
        pass
