import functools
import inspect
import logging
import time

from torch.multiprocessing import Queue, Process
from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from inferno.trainers import Trainer

from .datasets import DynamicDataset


logger = logging.getLogger(__name__)


class InferenceProcess(Process):
    """
    Process for neural network inference
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

                if 'callable' in inspect.getfullargspec()['args']:
                    meth = functools.partial(meth, callable=handle_incoming_msgs())

                meth(**kwargs)

        while True:
            handle_incoming_msgs()
            time.sleep(0.01)

    def forward(self, data, callback : Callable):
        pass
