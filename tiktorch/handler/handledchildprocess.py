import logging.config

from functools import partial
from multiprocessing.connection import Connection
from torch.multiprocessing import Process

from typing import Callable

# logging.basicConfig(level=logging.DEBUG)
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {"default": {"level": "INFO", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"}},
        "loggers": {"": {"handlers": ["default"], "level": "DEBUG", "propagate": True}},
    }
)


class HandledChildProcess(Process):
    def __init__(self, handler_conn: Connection, name: str) -> None:
        self.handler_conn = handler_conn
        self.devices = []
        self.idle = False
        super().__init__(name=name)

    @property
    def handle_incoming_msgs_callback(self) -> Callable:
        return partial(self.handle_incoming_msgs, timeout=0, callback=True)

    def handle_incoming_msgs(self, timeout: float, callback: bool = False) -> None:
        """
        :param timeout: wait for at most 'timeout' seconds for incoming messages
        :param callback: do not send idle message when using as callback
        """
        if callback:
            assert timeout == 0, "Do not wait for messages when using as callback"

        if self.handler_conn.poll(timeout=timeout):
            self.idle = False
            call, kwargs = self.handler_conn.recv()
            meth = getattr(self, call, None)
            if meth is None:
                raise NotImplementedError(call)

            meth(**kwargs)
        elif not callback and not self.idle:
            self.idle = True
            # todo: make sure devices are actually free
            self.handler_conn.send(("report_idle", {"proc_name": self.name, "devices": self.devices}))
