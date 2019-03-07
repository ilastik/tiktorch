import logging
import time

from queue import Empty, Full
from torch.utils.data import DataLoader
from torch.multiprocessing import Process
from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from inferno.trainers import Trainer

from .base import ProcessComm, SHUTDOWN_ANSWER
from .datasets import DynamicDataset

logger = logging.getLogger(__name__)
TRAINING = "train"  # same name as used in inferno
VALIDATION = "validate"  # same name as used in inferno


class TrainingProcess(Process):
    """
    Process to run an inferno trainer instance to train a neural network. This instance is used for validation as well.
    """

    def __init__(self, handler_comm: ProcessComm):
        """
        :param from_handler_queue: downstream communication
        :param to_handler_queue: upstream communication
        """
        super().__init__(name="TrainingProcess")
        self.handler_comm = handler_comm
        self.keep_training = None

        self.datasets = {TRAINING: DynamicDataset(), VALIDATION: DynamicDataset()}
        self.loader_kwargs = {
            TRAINING: {"dataset": self.datasets[TRAINING]},
            VALIDATION: {"dataset": self.datasets[VALIDATION]},
        }

    def run(self):
        def handle_incoming_msgs():
            try:
                call, kwargs = self.handler_comm.recv()
                meth = getattr(self, call, None)
                if meth is None:
                    raise NotImplementedError(call)

                meth(**kwargs)
            except Empty:
                pass

        self.trainer = Trainer.build(logger_config={'name': 'InfernoTrainer'})  # todo: configure (create/load) inferno trainer

        # listen periodically while training/validation is running
        self.trainer.register_callback(handle_incoming_msgs, trigger="end_of_training_iteration")
        self.trainer.register_callback(handle_incoming_msgs, trigger="end_of_validation_iteration")

        try:
            while True:
                handle_incoming_msgs()
                time.sleep(0.01)
        finally:
            self.shutdown()

    def shutdown(self):
        try:
            self.handler_comm.send(SHUTDOWN_ANSWER, timeout=2)
        except Full:
            pass
        # todo: save!?!

    def resume_train(self) -> None:
        if self.keep_training is None:  # start training
            pass

        self.keep_training = True

    def pause_train(self) -> None:
        self.keep_training = False

    def update_dataset(self, name: str, keys: Iterable, values: Iterable) -> None:
        assert name in (TRAINING, VALIDATION)
        self.datasets[name].update(keys=keys, values=values)

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
