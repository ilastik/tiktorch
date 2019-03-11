import logging
import time
import torch.nn

from multiprocessing.connection import Connection
from torch.multiprocessing import Process
from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

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
        assert hasattr(self, SHUTDOWN[0])
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

    def forward(self, keys: Iterable, data: torch.Tensor) -> None:
        """
        :param data: input data to neural network
        :return: predictions
        """
        callback = self.handle_incoming_msgs()
        self.handler_conn.send(("forward_answer", {"keys": keys, "data": self.model(data).detach()}))
