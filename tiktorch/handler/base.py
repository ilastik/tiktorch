import logging
import torch

from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from torch.multiprocessing import Queue, Process
from queue import Empty, Full

from ..types import NDArrayBatch

logger = logging.getLogger(__name__)


class HandlerProcess(Process):
    """
    Process to orchestrate the interplay of training/validation and inference
    """
    def __init__(self, from_server_queue : Queue, to_server_queue : Queue, timeout : int = 300):
        """
        :param from_server_queue: downstream communication
        :param to_server_queue: upstream communication
        :param timeout: log a warning if no message has been received for this many seconds
        """
        self.from_server_queue = from_server_queue
        self.to_server_queue = to_server_queue
        self.timeout = timeout
        super().__init__(name='HandlerProcess')

    def run(self):

        while(True):
            try:
                call, kwargs = self.from_server_queue.get(block=True, timeout=self.timeout)
                meth = getattr(self, call, None)
                if meth is None:
                    raise NotImplementedError(call)

                meth(**kwargs)
            except Empty:
                logger.warning("No message received in the last %d seconds", self.timeout)
            except Full:
                # todo: handle full queue
                raise

    # general
    def shutdown(self):
        pass

    # inference
    def forward(self, data : NDArrayBatch):
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
