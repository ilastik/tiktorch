import enum
import inspect
import logging
import queue
import threading
from concurrent.futures import Future
from functools import partial
from typing import Any, Callable, Dict, Generic, Iterator, List, Mapping, Optional, Set, Tuple, TypeVar
from uuid import uuid4

import zmq

from .connections import IConnConf
from .exceptions import CallException, Canceled, Shutdown, Timeout
from .interface import RPCInterface, get_exposed_methods
from .serialization import deserialize, serialize
from .types import RPCFuture, isfutureret

logger = logging.getLogger(__name__)


T = TypeVar("T")


@enum.unique
class Mode(enum.Enum):
    Normal = b"0"
    Shutdown = b"1"


@enum.unique
class State(enum.Enum):
    Return = b"0"
    Error = b"1"
    Ack = b"2"


def isfuture(obj):
    return isinstance(obj, (RPCFuture, Future))


def serialize_args(
    func: Callable, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None
) -> Iterator[zmq.Frame]:

    kwargs = kwargs or {}
    ismethod = inspect.ismethod(func)

    sig = inspect.signature(func)

    # Ensure that we only handle class methods
    if not ismethod:
        raise ValueError(f"{func} should be bound instance method")

    bound_args = sig.bind(*args, **kwargs).arguments

    for arg in sig.parameters.values():
        type_ = arg.annotation
        call_arg = bound_args[arg.name]
        yield from serialize(call_arg)


def deserialize_args(func: Callable, frames: Iterator[zmq.Frame]) -> List[Any]:
    sig = inspect.signature(func)
    args = []

    for arg in sig.parameters.values():
        type_ = arg.annotation
        args.append(deserialize(frames))

    return args


def serialize_return(func: Callable, value: Any) -> Iterator[zmq.Frame]:
    return serialize(value)
    # sig = inspect.signature(func)

    # if sig.return_annotation and issubclass(sig.return_annotation, RPCFuture):
    #     type_, *rest = typing_inspect.get_args(sig.return_annotation)
    #     if rest:
    #         raise ValueError("Tuple returns are not supported")
    #     return serialize(type_, value)

    # else:
    #     return serialize(sig.return_annotation, value)


def deserialize_return(func: Callable, frames: Iterator[zmq.Frame]) -> Any:
    sig = inspect.signature(func)

    # if sig.return_annotation and issubclass(sig.return_annotation, RPCFuture):
    #     type_, *rest = typing_inspect.get_args(sig.return_annotation)
    #     if rest:
    #         raise ValueError("Tuple returns are not supported")

    return deserialize(frames)

    # else:
    #     return deserialize(frames)


class Result:
    __slots__ = ("_value", "_exc")

    def __init__(self, value: Optional[Any] = None, exc: Optional[Exception] = None) -> None:
        self._value = value
        self._exc = exc

    def result(self) -> Any:
        if self._exc:
            raise self._exc

        return self._value

    def to_future(self, future: RPCFuture) -> None:
        if self._exc:
            future.set_exception(self._exc)

        future.set_result(self._value)


class Ack:
    __slots__ = "_exc"

    def __init__(self, exc: Optional[Exception] = None):
        self._exc = exc

    def to_future(self, future: RPCFuture) -> None:
        if self._exc:
            future.set_exception(self._exc)


def deserialize_result(method: Callable, frames: Iterator[zmq.Frame]):

    ctrl_frm = next(frames)

    if ctrl_frm.bytes == State.Error.value:
        msg = next(frames, None)
        return Result(exc=CallException(msg))

    elif ctrl_frm.bytes == State.Return.value:
        value = deserialize_return(method, frames)
        return Result(value=value)

    elif ctrl_frm.bytes == State.Ack.value:
        raise Exception("Expected return value. But received Future")

    raise Exception("Unexpected control frame %s" % ctrl_frm)


def deserialize_ack(frames: Iterator[zmq.Frame]):
    ctrl_frm = next(frames)

    if ctrl_frm.bytes == State.Error.value:
        msg = next(frames, None)
        return Ack(exc=CallException(msg))

    elif ctrl_frm.bytes == State.Ack.value:
        return Ack()

    raise Exception("Unexpected control frame %s" % ctrl_frm)


class MethodDispatcher:
    def __init__(self, method_name, method, client):
        self._method_name = method_name
        self._client = client
        self._method = method
        self._id = uuid4().hex.encode("ascii")

    def __call__(self, *args, **kwargs) -> Any:
        logger.debug("[id: %s] Send call %s", self._id, self._method_name)
        method_name = self._method_name.encode("utf-8")
        frames = [method_name, self._id, *serialize_args(self._method, args, kwargs)]
        is_future = isfutureret(self._method)
        if is_future:
            logger.debug("[id: %s] Created future", self._id)
            fut = self._client.create_future(self._id, self._method)

        # temporal dep,
        # postbox (future) should be created before address is known by remote
        return_frames = iter(self._client.dispatch(frames))

        if is_future:
            ack = deserialize_ack(return_frames)
            ack.to_future(fut)
            return fut
        else:
            res = deserialize_result(self._method, return_frames)
            return res.result()


class Client:
    def __init__(self, api: RPCInterface, conn_conf: IConnConf) -> None:

        self._methods_by_name = get_exposed_methods(api)
        self._conn_conf = conn_conf

        self._name = api.__class__.__name__
        self._local = threading.local()
        self._poller = zmq.Poller()
        self._timeout = conn_conf.get_timeout()
        self._futures = {}
        self._shutdown = threading.Event()
        self._listener = None
        self._ctx = self._conn_conf.get_ctx()

    def create_future(self, id_, method):
        if self._listener is None:
            self._start_listener()

        fut_timeout = self._timeout if self._timeout != -1 else None  # infinite timeout: zmq: -1, Future: None
        f = RPCFuture(timeout=fut_timeout)
        self._futures[id_] = (f, method)
        return f

    def _start_listener(self):
        def _listen():
            ctx = self._ctx
            sock = ctx.socket(zmq.PAIR)
            sock.setsockopt(zmq.LINGER, 2000)
            sock.RCVTIMEO = self._timeout
            sock.SNDTIMEO = self._timeout
            sock.connect(self._conn_conf.get_pubsub_conn_str())

            while True:
                events = sock.poll(flags=zmq.POLLIN, timeout=500)

                if events == zmq.POLLIN:
                    id_frm, *return_frames = sock.recv_multipart(copy=False)
                    id_ = id_frm.bytes
                    logger.debug("[id: %s] Recieved return", id_)
                    fut, method = self._futures.pop(id_, (None, None))
                    if fut is not None:
                        try:
                            result = deserialize_result(method, iter(return_frames))
                        except Exception as e:
                            fut.set_exception(e)
                        else:
                            result.to_future(fut)

                if self._shutdown.is_set():
                    sock.close()
                    break

        self._listener = threading.Thread(target=_listen, name=f"ClientNotificationsThread[{self._name}]")
        self._listener.daemon = True
        self._listener.start()

    @property
    def _socket(self):
        sock = getattr(self._local, "sock", None)

        if sock is None:
            ctx = self._conn_conf.get_ctx()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 2000)
            sock.RCVTIMEO = self._timeout
            sock.SNDTIMEO = self._timeout
            sock.connect(self._conn_conf.get_conn_str())
            setattr(self._local, "sock", sock)

        return sock

    def __getattr__(self, name) -> Any:
        method = self._methods_by_name.get(name)
        if method is None:
            raise AttributeError(name)
        return MethodDispatcher(name, method, self)

    def dispatch(self, frames: List[zmq.Frame]):
        evt = self._socket.poll(self._timeout, flags=zmq.POLLOUT)
        if not evt:
            raise Timeout()

        frames = self._socket.send_multipart(frames, copy=False)

        evt = self._socket.poll(1000 * self._timeout, flags=zmq.POLLIN)
        if not evt:
            raise Timeout()

        mode_frm, *resp = self._socket.recv_multipart(copy=False)

        if mode_frm.bytes == Mode.Shutdown.value:
            self._socket.close()

            for f, _ in list(self._futures.values()):
                f.set_exception(Shutdown())

            self._shutdown.set()

            raise Shutdown()

        return resp

    def __dir__(self):
        own_methods = self.__dict__.keys()
        iface_methods = self._methods_by_name.keys()
        return iface_methods ^ own_methods


class Server:
    def __init__(self, api: RPCInterface, conn_conf: IConnConf) -> None:

        self._api = api

        self._ctx = ctx = conn_conf.get_ctx()
        self._conn_conf = conn_conf
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.RCVTIMEO = 2000
        sock.SNDTIMEO = 2000
        sock.bind(conn_conf.get_conn_str())

        self._futures: Set[Future[Any]] = set()

        self._socket = sock
        self._method_by_name = get_exposed_methods(api)
        self._pub_lock = threading.Lock()
        self._results_queue = queue.Queue()
        self._shutdown_event = threading.Event()
        self._result_sender = None

    def _start_result_sender(self):
        def _sender():
            pub = self._ctx.socket(zmq.PAIR)
            pub.setsockopt(zmq.LINGER, 0)
            pub.RCVTIMEO = 2000
            pub.SNDTIMEO = 2000
            pub.bind(self._conn_conf.get_pubsub_conn_str())

            while True:
                try:
                    result = self._results_queue.get(timeout=0.5)
                    pub.send_multipart(result)
                except queue.Empty:
                    pass

                if self._shutdown_event.is_set():
                    pub.close()
                    break

        t = threading.Thread(target=_sender, name="ResultSender")
        t.start()
        return t

    def _make_done_callback(self, id_: bytes, func: Callable[..., Any]) -> Callable[[Future], None]:
        logger.debug("[id: %s]. Created done callback", id_)
        if self._result_sender is None:
            self._result_sender = self._start_result_sender()

        def _done_callback(fut: Future) -> None:
            self._futures.discard(fut)

            try:
                result = fut.result()
                resp = [id_, State.Return.value, *serialize_return(func, result)]
            except Exception as e:
                logger.error("[id: %s]. Future expection", id_, exc_info=1)
                resp = [id_, State.Error.value, str(e).encode("utf-8")]

            logger.debug("[id: %s]. Sending result", id_)
            self._results_queue.put(resp)

        return _done_callback

    def _call(self, func: Callable, id_: bytes, frames: List[zmq.Frame]) -> List[zmq.Frame]:
        frames_it = iter(frames)
        args = deserialize_args(func, frames_it)

        logger.debug("[id: %s]. Invoking method %s", id_, func)
        ret = func(*args)
        logger.debug("[id: %s]. Return value", id_)

        if isfuture(ret):
            logger.debug("[id: %s]. Handling future", id_)

            self._futures.add(ret)
            ret.add_done_callback(self._make_done_callback(id_, func))

            return [State.Ack.value]

        return [State.Return.value, *serialize_return(func, ret)]

    def listen(self):
        while True:
            try:
                frames = self._socket.recv_multipart(copy=False)
            except zmq.Again:
                continue
            method_frm, method_id_frm, *args = frames

            method_name = method_frm.bytes
            method_id = method_id_frm.bytes
            logger.debug("Invoking %s method", method_name)

            method = self._method_by_name.get(method_name.decode("utf-8"))

            try:
                if method is None:
                    raise Exception(f"Unknown method {method_name}")

                resp_frames = self._call(method, method_id, args)

            except Shutdown:
                self._shutdown_event.set()
                if self._result_sender:
                    self._result_sender.join()
                for f in list(self._futures):
                    f.cancel()
                self._socket.send_multipart([Mode.Shutdown.value])
                self._socket.close()
                break

            except Exception as e:
                logger.exception("Exception during method %s call", method)
                # TODO: Better exception serialization
                self._socket.send_multipart([Mode.Normal.value, State.Error.value, str(e).encode("ascii")])
            # TODO: rearm socket after timeout to avoid being stuck in sending state
            # in case of client missing (dead)
            else:
                self._socket.send_multipart([Mode.Normal.value, *resp_frames], copy=False)
