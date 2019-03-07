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

        self.training_dataset = DynamicDataset()
        self.training_loader_kwargs = {"dataset": self.training_dataset}
        self.validation_dataset = DynamicDataset()
        self.validation_loader_kwargs = {"dataset": self.validation_dataset}

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

        self.trainer = Trainer.build()  # todo: configure (create/load) inferno trainer

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

    def update_training_dataset(self, keys: Iterable, values: Iterable) -> None:
        self.training_dataset.update(keys=keys, values=values)

    def update_validation_dataset(self, keys: Iterable, values: Iterable) -> None:
        self.validation_dataset.update(keys=keys, values=values)

    def update_hparams(self, hparams: dict):
        update_training_data_loader = False
        # update_validation_data_loader = False  # todo: validation data loader
        for key, value in hparams.items():
            if key in "batch_size":
                update_training_data_loader = True
                self.training_loader_kwargs[key] = value
            else:
                raise NotImplementedError(f"How to set {key} as a hyper parameter?")

        if update_training_data_loader:  # need to bind new training data loader
            new_training_loader = DataLoader(**self.training_loader_kwargs)

    def request_state(self):
        # model state
        # optimizer state
        # current config dict
        raise NotImplementedError
