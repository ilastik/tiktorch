import logging
import time

from queue import Empty, Full
from torch.multiprocessing import Process
from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from .base import ProcessComm
from .datasets import DynamicDataset
from ..types import NDArrayBatch


logger = logging.getLogger(__name__)


class InferenceProcess(Process):
    """
    Process for neural network inference
    """

    def __init__(self, handler_comm: ProcessComm) -> None:
        """
        :param from_handler_queue: downstream communication
        :param to_handler_queue: upstream communication
        """
        super().__init__(name="TrainingProcess")
        self.handler_comm = handler_comm
        self.keep_training = False

        self.training_dataset = DynamicDataset()
        self.validation_dataset = DynamicDataset()

    # internal
    def handle_incoming_msgs(self) -> None:
        try:
            call, kwargs = self.handler_comm.recv()
            meth = getattr(self, call, None)
            if meth is None:
                raise NotImplementedError(call)

            meth(**kwargs)
        except Empty:
            pass

    def run(self) -> None:
        while True:
            self.handle_incoming_msgs()
            time.sleep(0.01)

    def forward(self, data: NDArrayBatch) -> NDArrayBatch:
        """
        :param data: input data to neural network
        :return: predictions
        """
        callback = self.handle_incoming_msgs()
