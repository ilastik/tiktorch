import inspect
import logging
import enum

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
    if not isinstance(obj, type):
        cls = type(obj)
    else:
        cls = obj

    iface = None

    for base in inspect.getmro(cls):
        if issubclass(base, RPCInterface) and base is not RPCInterface:
            iface = base

    if iface is None:
        raise ValueError(f"Class doesn't provide public API")

    public_attrs = [attr for attr in dir(iface) if not attr.startswith('_')]
    exposed_methods = {}

    for attr_name in public_attrs:
        attr = getattr(obj, attr_name)
        if callable(attr):
            exposed_methods[attr_name] = attr

    return exposed_methods


class MethodDispatcher:
    def __init__(self, method_name, method, client):
        self._method_name = method_name
        self._client = client
        self._method = method

    def _raise(self, frames):
        msg = next(frames, None)
        raise Exception(msg)

    def __call__(self, *args, **kwargs) -> Any:
        method_name = self._method_name.encode('ascii')
        frames = [method_name, *serialize_args(self._method, args, kwargs)]
        return_frames = self._client.dispatch(frames)
        ctrl_frm, *rest = return_frames

        it = iter(rest)
        if ctrl_frm.bytes == State.Error.value:
            self._raise(it)
        elif ctrl_frm.bytes == State.Return.value:
            return deserialize_return(self._method, it)


class RPCInterface:
    pass


class Client:
    def __init__(self, api: type, socket: zmq.Socket) -> None:
        self._methods_by_name = get_exposed_methods(api)
        self._socket = socket

    def __getattr__(self, name) -> Any:
        method = self._methods_by_name.get(name)
        if method is None:
            raise AttributeError(name)
        return MethodDispatcher(name, method, self)

    def dispatch(self, frames: List[zmq.Frame]):
        # TODO: Timeout
        frames = self._socket.send_multipart(frames, copy=False)
        return self._socket.recv_multipart(copy=False)

    def __dir__(self):
        own_methods = self.__dict__.keys()
        iface_methods = self._methods_by_name.keys()
        return iface_methods ^ own_methods


class Shutdown(Exception):
    pass


class Server:
    def __init__(self, api, socket: zmq.Socket) -> None:
        self._api = api
        self._socket = socket
        self._method_by_name = get_exposed_methods(api)

    def _call(self, func: Callable, frames: List[zmq.Frame]) -> Iterator[zmq.Frame]:
        frames_it = iter(frames)
        args = deserialize_args(func, frames_it)
        ret = func(*args)
        return serialize_return(func, ret)

    def listen(self):
        while True:
            frames = self._socket.recv_multipart(copy=False)
            method_frm, *args = frames

            method_name = method_frm.bytes
            method = self._method_by_name.get(method_name.decode('ascii'))

            try:
                if method is None:
                    raise Exception(f'Unknown method {method_name}')

                resp_frames = [State.Return.value, *self._call(method, args)]

            except Shutdown:
                self._socket.send(b'')
                break

            except Exception as e:
                logger.exception('Exception during method %s call', method)
                # TODO: Better exception serialization
                self._socket.send_multipart([
                    State.Error.value, str(e).encode('ascii')
                ])

            else:
                self._socket.send_multipart(list(resp_frames), copy=False)

