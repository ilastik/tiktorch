import logging
import time

from torch.multiprocessing import Queue, Process
from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from inferno.trainers import Trainer

from .datasets import DynamicDataset


logger = logging.getLogger(__name__)


class TrainingProcess(Process):
    """
    Process to run an inferno trainer instance to train a neural network. This instance is used for validation as well.
    """

    def __init__(self, from_handler_queue: Queue, to_handler_queue: Queue):
        """
        :param from_handler_queue: downstream communication
        :param to_handler_queue: upstream communication
        """
        super().__init__(name="TrainingProcess")
        self.from_handler_queue = from_handler_queue
        self.to_handler_queue = to_handler_queue
        self.keep_training = False

        self.training_dataset = DynamicDataset()
        self.validation_dataset = DynamicDataset()

    def run(self):
        def handle_incoming_msgs():
            if not self.from_handler_queue.empty():
                call, kwargs = self.from_handler_queue.get()
                meth = getattr(self, call, None)
                if meth is None:
                    raise NotImplementedError(call)

                meth(**kwargs)

        self.trainer = Trainer.build()  # todo: configure (create/load) inferno trainer

        # listen periodically while training/validation is running
        self.trainer.register_callback(handle_incoming_msgs, trigger="end_of_training_iteration")
        self.trainer.register_callback(handle_incoming_msgs, trigger="end_of_validation_iteration")

        while True:
            handle_incoming_msgs()
            time.sleep(0.01)

    def resume_train(self) -> None:
        self.keep_training = True

    def pause_train(self) -> None:
        self.keep_training = False

    def update_training_dataset(self, keys: Iterable, values: Iterable) -> None:
        self.training_dataset.update(keys=keys, values=values)

    def update_validation_dataset(self, keys: Iterable, values: Iterable) -> None:
        self.validation_dataset.update(keys=keys, values=values)

    def request_state(self):
        # model state
        # optimizer state
        # current config dict
        raise NotImplementedError
