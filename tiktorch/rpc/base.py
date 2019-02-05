import inspect
import logging

from typing import Any, List, Generic, Iterator, TypeVar, Mapping, Callable, Dict

import zmq

from .serialization import serialize, deserialize


def serialize_args(
    func: Callable,
    args: List[Any],
    kwargs: Dict[str, Any]
) -> Iterator[zmq.Frame]:

    spec = inspect.getfullargspec(func)
    bound_args = inspect.getcallargs(func, None, *args, **kwargs)

    for arg_name in spec.args:
        if arg_name == 'self':
            # TODO: Better way to skip self attribute (self naming is a convention)
            continue
        type_ = spec.annotations[arg_name]
        call_arg = bound_args[arg_name]
        yield from serialize(type_, call_arg)


def deserialize_return(
    func: Callable,
    frames: Iterator[zmq.Frame],
) -> Any:
    sig = inspect.signature(func)
    return deserialize(sig.return_annotation, frames)


def deserialize_args(
    func: Callable,
    frames: Iterator[zmq.Frame],
) -> List[Any]:
    spec = inspect.getfullargspec(func)
    args = []

    for arg_name in spec.args:
        if arg_name == 'self':
            # TODO: Better way to skip self attribute (self naming is a convention)
            continue
        type_ = spec.annotations[arg_name]
        args.append(deserialize(type_, frames))

    return args


def serialize_return(
    func: Callable,
    value: Any,
) -> Iterator[zmq.Frame]:
    sig = inspect.signature(func)
    return serialize(sig.return_annotation, value)

logger = logging.getLogger(__name__)


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
            exposed_methods[attr_name.encode('ascii')] = attr

    return exposed_methods


class MethodDispatcher:
    def __init__(self, method_name, method, client):
        self._method_name = method_name
        self._client = client
        self._method = method

    def __call__(self, *args, **kwargs) -> Any:
        method_name = self._method_name.encode('ascii')
        frames = [method_name, *serialize_args(self._method, args, kwargs)]
        return_frames = self._client.dispatch(frames)
        it = iter(return_frames)
        return deserialize_return(self._method, it)


class RPCInterface:
    pass


class Client:
    def __init__(self, api: type, socket: zmq.Socket) -> None:
        self._methods_by_name = get_exposed_methods(api)
        self._socket = socket

    def __getattr__(self, name) -> Any:
        method = self._methods_by_name.get(name.encode('ascii'))
        if method is None:
            raise AttributeError(name)
        return MethodDispatcher(name, method, self)

    def dispatch(self, frames: List[zmq.Frame]):
        # TODO: Timeout
        frames = self._socket.send_multipart(frames, copy=False)
        return self._socket.recv_multipart(copy=False)


class Shutdown(Exception):
    pass


class Server:
    def __init__(self, api, socket: zmq.Socket) -> None:
        self._api = api
        self._socket = socket
        self._method_by_name = get_exposed_methods(api)

    def _call(self, func: Callable, frames: List[zmq.Frame]) -> List[zmq.Frame]:
        frames_it = iter(frames)
        args = deserialize_args(func, frames_it)
        ret = func(*args)
        return serialize_return(func, ret)

    def listen(self):
        while True:
            frames = self._socket.recv_multipart(copy=False)
            method_frm, *args = frames

            method_name = method_frm.bytes
            method = self._method_by_name.get(method_name)

            if method is None:
                raise Exception(f'Unknown method {method_name}')

            try:
                resp_frames = self._call(method, args)
            except Shutdown:
                self._socket.send(b'')
                break

            self._socket.send_multipart(list(resp_frames))
