import inspect
import logging
import threading
import enum

from concurrent.futures import Future
from uuid import uuid4
from typing import (
    Any, List, Generic, Iterator, TypeVar, Mapping, Callable, Dict, Optional, Tuple
)

import zmq

from .serialization import serialize, deserialize


logger = logging.getLogger(__name__)


@enum.unique
class State(enum.Enum):
    Return = b'0'
    Error = b'1'


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
    def __init__(
        self,
        name: str,
        pubsub: str,
        ctx: zmq.Context,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Inproc config is dependent on sharing *same context instance*
        """
        self._ctx = ctx
        self._timeout = timeout
        self.name = name
        self.pubsub = pubsub

    def get_conn_str(self) -> str:
        return f'inproc://{self.name}'

    def get_pubsub_conn_str(self) -> str:
        return f'inproc://{self.pubsub}'


class TCPConnConf(IConnConf):
    def __init__(
        self,
        addr: str,
        port: str,
        pubsub_port: str,
        timeout: Optional[int] = None,
        ctx: Optional[zmq.Context] = None,
    ) -> None:
        self.port = port
        self.addr = addr
        self._timeout = timeout
        self._ctx = ctx or zmq.Context.instance()
        self.pubsub_port = pubsub_port

    def get_conn_str(self) -> str:
        return f'tcp://{self.addr}:{self.port}'

    def get_pubsub_conn_str(self) -> str:
        return f'tcp://{self.addr}:{self.pubsub_port}'


def serialize_args(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None
) -> Iterator[zmq.Frame]:

    kwargs = kwargs or {}
    ismethod = inspect.ismethod(func)

    sig = inspect.signature(func)

    # Ensure that we only handle class methods
    if not ismethod:
        raise ValueError(f'{func} should be bound instance method')

    bound_args = sig.bind(*args, **kwargs).arguments

    for arg in sig.parameters.values():
        type_ = arg.annotation
        call_arg = bound_args[arg.name]
        yield from serialize(type_, call_arg)


def deserialize_args(
    func: Callable,
    frames: Iterator[zmq.Frame],
) -> List[Any]:
    sig = inspect.signature(func)
    args = []

    for arg in sig.parameters.values():
        type_ = arg.annotation
        args.append(deserialize(type_, frames))

    return args


def serialize_return(
    func: Callable,
    value: Any,
) -> Iterator[zmq.Frame]:
    sig = inspect.signature(func)
    return serialize(sig.return_annotation, value)


def deserialize_return(
    func: Callable,
    frames: Iterator[zmq.Frame],
) -> Any:
    sig = inspect.signature(func)
    return deserialize(sig.return_annotation, frames)


def get_exposed_methods(obj):
    exposed = getattr(obj, '__exposedmethods__', None)

    if not exposed:
        raise ValueError(f"Class doesn't provide public API")

    exposed_methods = {}

    for attr_name in exposed:
        attr = getattr(obj, attr_name)
        if callable(attr):
            exposed_methods[attr_name] = attr

    return exposed_methods


class RPCInterfaceMeta(type):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        exposed = {
            name
            for name, value in namespace.items()
            if getattr(value, "__exposed__", False)
        }

        for base in bases:
            if issubclass(base, RPCInterface):
                 exposed ^= getattr(base, "__exposedmethods__", set())

        cls.__exposedmethods__ = frozenset(exposed)
        return cls


class RPCInterface(metaclass=RPCInterfaceMeta):
    pass


def exposed(method):
    method.__exposed__ = True
    return method


class MethodDispatcher:
    def __init__(self, method_name, method, client):
        self._method_name = method_name
        self._client = client
        self._method = method
        self._id = uuid4().bytes

    def _raise(self, frames):
        msg = next(frames, None)
        raise Exception(msg)

    def __call__(self, *args, **kwargs) -> Any:
        method_name = self._method_name.encode('ascii')
        frames = [method_name, self._id, *serialize_args(self._method, args, kwargs)]
        return_frames = self._client.dispatch(frames)
        ctrl_frm, *rest = return_frames

        it = iter(rest)
        if ctrl_frm.bytes == State.Error.value:
            self._raise(it)
        elif ctrl_frm.bytes == State.Return.value:
            return deserialize_return(self._method, it)


class Client:
    def __init__(
        self,
        api: type,
        conn_conf: IConnConf,
    ) -> None:
        self._methods_by_name = get_exposed_methods(api)
        self._conn_conf = conn_conf

        self._local = threading.local()
        self._poller = zmq.Poller()
        self._timeout = conn_conf.get_timeout()
        self._futures = {}

    @property
    def _socket(self):
        sock = getattr(self._local, 'sock', None)

        if sock is None:
            ctx = self._conn_conf.get_ctx()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.RCVTIMEO = self._timeout
            sock.connect(self._conn_conf.get_conn_str())
            setattr(self._local, 'sock', sock)

        return sock

    def __getattr__(self, name) -> Any:
        method = self._methods_by_name.get(name)
        if method is None:
            raise AttributeError(name)
        return MethodDispatcher(name, method, self)

    def dispatch(self, frames: List[zmq.Frame]):
        evt = self._socket.poll(self._timeout, flags=zmq.POLLOUT)
        if not evt:
            raise TimeoutError()

        frames = self._socket.send_multipart(frames, copy=False)

        evt = self._socket.poll(self._timeout, flags=zmq.POLLIN)
        if not evt:
            raise TimeoutError()

        return self._socket.recv_multipart(copy=False)

    def __dir__(self):
        own_methods = self.__dict__.keys()
        iface_methods = self._methods_by_name.keys()
        return iface_methods ^ own_methods


class Shutdown(Exception):
    pass


class TimeoutError(Exception):
    pass


class Server:
    def __init__(
        self,
        api: RPCInterface,
        conn_conf: IConnConf,
    ) -> None:
        self._api = api

        ctx = conn_conf.get_ctx()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(conn_conf.get_conn_str())

        pub = ctx.socket(zmq.PAIR)
        pub.setsockopt(zmq.LINGER, 2000)
        pub.bind(conn_conf.get_pubsub_conn_str())

        self._futures = {}

        self._socket = sock
        self._pub_sock = pub
        self._method_by_name = get_exposed_methods(api)

    def _done_callback(self, fut: Future) -> None:
        # TODO: handle exceptions
        result = fut.result()
        d = [fut.id.encode('ascii'), result]
        print('done callback', d)
        self._pub_sock.send_multipart(d)
        print('sent multipart')

    def _call(self, func: Callable, id_: bytes, frames: List[zmq.Frame]) -> Iterator[zmq.Frame]:
        frames_it = iter(frames)
        args = deserialize_args(func, frames_it)
        ret = func(*args)

        if isinstance(ret, Future):
            ret.id = uuid4().hex
            self._futures[ret] = func
            ret.add_done_callback(self._done_callback)

        return serialize_return(func, ret)

    def listen(self):
        while True:
            frames = self._socket.recv_multipart(copy=False)
            method_frm, method_id_frm, *args = frames

            method_name = method_frm.bytes
            method_id = method_id_frm.bytes
            logger.debug("Invoking %s method with req id %s", method_name, method_id)

            method = self._method_by_name.get(method_name.decode('ascii'))

            try:
                if method is None:
                    raise Exception(f'Unknown method {method_name}')

                resp_frames = [State.Return.value, *self._call(method, method_id, args)]

            except Shutdown:
                self._socket.send(b'')
                self._socket.close()
                break

            except Exception as e:
                logger.exception('Exception during method %s call', method)
                # TODO: Better exception serialization
                self._socket.send_multipart([
                    State.Error.value, str(e).encode('ascii')
                ])

            # TODO: rearm socket after timeout to avoid being stuck in sending state
            # in case of client missing (dead)
            else:
                self._socket.send_multipart(list(resp_frames), copy=False)
