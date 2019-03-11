import io
import logging
import time
import torch.nn, torch.optim

from contextlib import closing
from copy import deepcopy
from multiprocessing.connection import Connection
from queue import Empty, Full
from torch.utils.data import DataLoader
from torch.multiprocessing import Process
from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from inferno.trainers import Trainer

from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION
from .datasets import DynamicDataset

logging.basicConfig(level=logging.INFO)
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


TRAINING = "train"  # same name as used in inferno
VALIDATION = "validate"  # same name as used in inferno


class TrainingProcess(Process):
    """
    Process to run an inferno trainer instance to train a neural network. This instance is used for validation as well.
    """

    name = "TrainingProcess"

    def __init__(self, handler_conn: Connection, config: dict, model: torch.nn.Module, optimizer_state: bytes):
        assert hasattr(self, SHUTDOWN[0])
        super().__init__(name=self.name)
        logger = logging.getLogger(__name__)
        self.handler_conn = handler_conn
        self.is_training = False
        self.max_num_iterations = 0

        self.datasets = {TRAINING: DynamicDataset(), VALIDATION: DynamicDataset()}
        self.loader_kwargs = {
            TRAINING: {"dataset": self.datasets[TRAINING]},
            VALIDATION: {"dataset": self.datasets[VALIDATION]},
        }

        # setup inferno trainer configuration
        trainer_config = {
            "logger_config": {"name": "InfernoTrainer"},
            "max_num_iterations": self.max_num_iterations,
            "max_num_epochs": "inf",
        }
        # Some keys may not be set by the user
        if "max_num_iterations" in config:
            raise ValueError(
                "max_num_iterations is reserved for internal use."
                "The user should set max_num_iterations_per_update instead"
            )

        # Catch all forbidden keys we might have forgotten to implement an Exception for
        for key in trainer_config.keys():
            if key in config:
                raise ValueError("%s reserved for internal use. Do not specify in config!")

        # Some keys are required from the user
        if "optimizer_config" not in config:
            raise ValueError("Missing optimizer configuration!")

        trainer_config.update(deepcopy(config))
        if "max_num_iterations_per_update" not in config:
            config["max_num_iterations_per_update"] = 10

        optimizer = False
        if optimizer_state:
            try:
                optimizer = self._create_optimizer_from_binary_state(
                    trainer_config["optimizer_config"]["method"], optimizer_state
                )
                del trainer_config["optimizer_config"]
            except Exception as e:
                logger.warning(
                    "Could not load optimizer state due to %s.\nCreating new optimizer from %s",
                    e,
                    config["optimizer_config"],
                )

        self.trainer = Trainer.build(
            model=model, **trainer_config
        )  # todo: configure (create/load) inferno trainer fully
        if optimizer:
            self.trainer.optimizer = optimizer

    # internal
    def _create_optimizer_from_binary_state(self, name, binary_state):
        optimizer: torch.optim.Optimizer = getattr(torch.optim, name)
        with closing(io.BytesIO(binary_state)) as f:
            optimizer.load_state_dict(torch.load(f))

        return optimizer

    def run(self):
        self._shutting_down = False
        self.logger = logging.getLogger(self.name)
        self.logger.info("Starting")

        def handle_incoming_msgs():
            try:
                call, kwargs = self.handler_conn.recv()
                meth = getattr(self, call, None)
                if meth is None:
                    raise NotImplementedError(call)

                meth(**kwargs)
            except Empty:
                pass

        # listen periodically while training/validation is running
        self.trainer.register_callback(handle_incoming_msgs, trigger="end_of_training_iteration")
        self.trainer.register_callback(handle_incoming_msgs, trigger="end_of_validation_iteration")

        try:
            while not self._shutting_down:
                handle_incoming_msgs()
                time.sleep(0.01)
        except Exception as e:
            self.logger.error(e)
            self.handler_conn.send((REPORT_EXCEPTION, {"proc_name": self.name, "exception": e}))
            self.shutdown()

    def shutdown(self):
        assert not self._shutting_down
        self._shutting_down = True
        self.handler_conn.send(SHUTDOWN_ANSWER)
        self.logger.debug("Shutdown complete")

    def resume_training(self) -> None:
        if not self.is_training:
            self.is_training = True
            self.trainer.set_max_num_iterations(self.max_num_iterations)
            self.trainer.fit()

    def pause_training(self) -> None:
        self.is_training = False
        self.trainer.set_max_num_iterations(0)

    def update_dataset(self, name: str, keys: Iterable, values: Iterable) -> None:
        assert name in (TRAINING, VALIDATION)
        self.datasets[name].update(keys=keys, values=values)
        if name == TRAINING:
            self.max_num_iterations += self.config["max_num_iterations_per_update"] * len(keys)

    def update_hparams(self, name: str, hparams: dict):
        assert name in (TRAINING, VALIDATION)
        update_loader = {}
        update_loader[name] = False
        for key, value in hparams.items():
            if key in "batch_size":
                update_loader[name] = True
                self.loader_kwargs[name][key] = value
            else:
                raise NotImplementedError(f"How to set {key} as a hyper parameter?")

        if update_loader:  # need to bind new data loader
            self.trainer.bind_loader(name, DataLoader(**self.loader_kwargs[name]))

    def request_state(self):
        # model state
        # optimizer state
        # current config dict
        raise NotImplementedError
