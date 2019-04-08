from typing import Optional

import zmq


class IConnConf:
    _ctx: zmq.Context
    _timeout: Optional[int]

    def get_conn_str(self) -> str:
        """
        :returns str: valid connection string for zmq.Socket
        """
        raise NotImplementedError()

    def get_pubsub_conn_str(self) -> str:
        """
        :returns str: valid connection string for PUB/SUB zmq.Socket
        """
        raise NotImplementedError()

    def get_ctx(self) -> zmq.Context:
        """
        :returns zmq.Context: same ctx for each class instance
        """
        return self._ctx

    def get_timeout(self) -> int:
        """
        timeout in seconds to use with zmq
        -1 indefinite
        >= 0 ms
        """
        if self._timeout is None:
            return -1
        else:
            return self._timeout


class InprocConnConf(IConnConf):
    def __init__(self, name: str, pubsub: str, ctx: zmq.Context, timeout: Optional[int] = None) -> None:
        """
        Inproc config is dependent on sharing *same context instance*
        """
        self._ctx = ctx
        self._timeout = timeout
        self.name = name
        self.pubsub = pubsub

    def get_conn_str(self) -> str:
        return f"inproc://{self.name}"

    def get_pubsub_conn_str(self) -> str:
        return f"inproc://{self.pubsub}"


class TCPConnConf(IConnConf):
    def __init__(
        self, addr: str, port: str, pubsub_port: str, timeout: Optional[int] = None, ctx: Optional[zmq.Context] = None
    ) -> None:
        self.port = port
        self.addr = addr
        self._timeout = timeout
        self._ctx = ctx or zmq.Context.instance()
        self.pubsub_port = pubsub_port

    def get_conn_str(self) -> str:
        return f"tcp://{self.addr}:{self.port}"

    def get_pubsub_conn_str(self) -> str:
        return f"tcp://{self.addr}:{self.pubsub_port}"
