import logging
import time
import torch.nn

from multiprocessing.connection import Connection
from queue import Empty, Full
from torch.multiprocessing import Process
from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from .constants import SHUTDOWN, SHUTDOWN_ANSWER

logging.basicConfig(level=logging.DEBUG)

class InferenceProcess(Process):
    """
    Process for neural network inference
    """

    def __init__(self, handler_conn: Connection, model: torch.nn.Module) -> None:
        """
        :param from_handler_queue: downstream communication
        :param to_handler_queue: upstream communication
        """
        super().__init__(name="TrainingProcess")
        self.handler_conn = handler_conn
        self.model = model

    # internal
    def handle_incoming_msgs(self) -> None:
        try:
            call, kwargs = self.handler_conn.recv()
            meth = getattr(self, call, None)
            if meth is None:
                raise NotImplementedError(call)

            meth(**kwargs)
        except Empty:
            pass

    def run(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.info('Starting')
        while True:
            self.handle_incoming_msgs()
            time.sleep(0.01)

    def forward(self, keys: Iterable, data: torch.Tensor, answer_to: Connection = None) -> None:
        """
        :param data: input data to neural network
        :return: predictions
        """
        callback = self.handle_incoming_msgs()
        answer = ('forward_answer', {keys: keys, data: self.model(data)})

        if answer_to:
            answer_to.send(answer)
        else:
            self.handler_conn.send(answer)
