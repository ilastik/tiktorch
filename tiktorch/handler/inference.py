import logging
import time
import torch.nn

from multiprocessing.connection import Connection
from torch.multiprocessing import Process
from typing import Any, List, Generic, Iterator, Iterable, Sequence, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION

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


class InferenceProcess(Process):
    """
    Process for neural network inference
    """

    name = "InferenceProcess"

    def __init__(self, handler_conn: Connection, config: dict, model: torch.nn.Module) -> None:
        """
        :param from_handler_queue: downstream communication
        :param to_handler_queue: upstream communication
        """
        assert hasattr(self, SHUTDOWN[0]), "missing shutdown method"
        super().__init__(name=self.name)
        self.handler_conn = handler_conn
        self.config = config
        self.model = model

    # internal
    def handle_incoming_msgs(self) -> None:
        if self.handler_conn.poll():
            call, kwargs = self.handler_conn.recv()
            meth = getattr(self, call, None)
            if meth is None:
                raise NotImplementedError(call)

            meth(**kwargs)

    def run(self) -> None:
        self.logger = logging.getLogger(self.name)
        self.logger.info("Starting")
        self._shutting_down = False
        try:
            while not self._shutting_down:
                self.handle_incoming_msgs()
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

    def forward(self, keys: Sequence, data: torch.Tensor) -> None:
        """
        :param data: input data to neural network
        :return: predictions
        """
        batch_size: int = self.config.get("inference_batch_size", None)
        if batch_size is None:
            batch_size = 1
            increase_batch_size = True
        else:
            increase_batch_size = False

        start = 0
        last_batch_size = batch_size

        def create_end_generator(start, batch_size):
            return iter(range(start + batch_size, len(keys) + batch_size - 1, batch_size))

        end_generator = create_end_generator(start, batch_size)
        while start < len(keys):
            self.handle_incoming_msgs()
            end = next(end_generator)
            try:
                pred = self.model(data[start:end]).detach()
            except Exception as e:
                if batch_size > last_batch_size:
                    self.logger.info(
                        "forward pass with batch size %d threw exception %s. Using previous batch size %d again.",
                        batch_size,
                        e,
                        last_batch_size,
                    )
                    batch_size = last_batch_size
                    increase_batch_size = False
                else:
                    last_batch_size = batch_size
                    batch_size //= 2
                    if batch_size == 0:
                        self.logger.error("Forward pass failed. Processed %d/%d", start, len(keys))
                        break

                    increase_batch_size = True
                    self.logger.info(
                        "forward pass with batch size %d threw exception %s. Trying again with smaller batch_size %d",
                        last_batch_size,
                        e,
                        batch_size,
                    )
                end_generator = create_end_generator(start, batch_size)
            else:
                self.handler_conn.send(("forward_answer", {"keys": keys[start:end], "data": pred}))
                start = end
                last_batch_size = batch_size
                if increase_batch_size:
                    batch_size += 1
                    end_generator = create_end_generator(start, batch_size)
