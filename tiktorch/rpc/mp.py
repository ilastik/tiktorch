import dataclasses
import logging
import queue
import threading
from concurrent.futures import Future
from functools import wraps
from multiprocessing.connection import Connection
from threading import Event, Thread
from typing import Any, List, Optional, Type, TypeVar
from uuid import uuid4

from bioimageio.core.resource_io import nodes

from ..server.session import IRPCModelSession
from .exceptions import Shutdown
from .interface import get_exposed_methods
from .types import RPCFuture, isfutureret

logger = logging.getLogger(__name__)


T = TypeVar("T")


class Result:
    __slots__ = ("_value", "_err")

    def __init__(self, *, value=None, err=None):
        self._value = value
        self._err = err

    @property
    def value(self):
        if self.is_err:
            raise self._err

        return self._value

    @property
    def error(self):
        return self._err

    @property
    def is_ok(self):
        return self._err is None

    @property
    def is_err(self):
        return not self.is_ok

    @classmethod
    def OK(cls, value):
        return cls(value=value)

    @classmethod
    def Error(cls, err):
        return cls(err=err)

    def to_future(self, fut):
        if self.is_err:
            fut.set_exception(self.error)
        else:
            fut.set_result(self.value)


class MPMethodDispatcher:
    def __init__(self, method_name, client):
        self._method_name = method_name
        self._client = client

    def sync(self, *args, **kwargs):
        f = self(*args, **kwargs)
        return f.result()

    def __call__(self, *args, **kwargs) -> Any:
        return self._client._invoke(self._method_name, *args, **kwargs)


def create_client_api(iface_cls: Type[T], conn: Connection, timeout=None) -> T:
    client = MPClient(iface_cls.__name__, conn, timeout)

    def _make_method(method):
        class MethodWrapper:
            @wraps(method)
            def async_(self, *args, **kwargs):
                return client._invoke(method.__name__, *args, **kwargs)

            if isfutureret(method):

                @wraps(method)
                def __call__(self, *args, **kwargs) -> Any:
                    return self.async_(*args, **kwargs)

            else:

                @wraps(method)
                def __call__(self, *args, **kwargs) -> Any:
                    fut = client._invoke(method.__name__, *args, **kwargs)
                    return fut.result(timeout=timeout)

        return MethodWrapper()

    class _Api:
        pass

    exposed_methods = get_exposed_methods(iface_cls)
    for method_name, method in exposed_methods.items():
        setattr(_Api, method_name, _make_method(method))

    return _Api()


@dataclasses.dataclass(frozen=True)
class BioModelClient:
    name: str
    api: IRPCModelSession
    input_specs: List[nodes.InputTensor]
    output_specs: List[nodes.OutputTensor]
    devices: List[str]


class MPClient:
    def __init__(self, name, conn: Connection, timeout: int):
        self._conn = conn
        self._request_by_id = {}
        self._name = name
        self._shutdown_event = Event()
        self._logger = None
        self._start_poller()
        self._timeout = timeout
        self._send_lock = threading.Lock()

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        return self._logger

    def _new_id(self) -> str:
        return uuid4().hex

    def _start_poller(self):
        def _poller():
            while True:
                if self._conn.poll(timeout=0.3):
                    try:
                        msg = self._conn.recv()
                    except Exception as exc:
                        self.logger.warning("Communication channel closed. Shutting Down.")
                        self._shutdown(exc)
                    else:
                        # signal
                        if isinstance(msg, Signal):
                            if msg.payload == b"shutdown":
                                self.logger.debug("[signal] Shutdown")
                                self._shutdown(Shutdown())

                        # method
                        elif isinstance(msg, MethodReturn):
                            fut = self._request_by_id.pop(msg.id, None)
                            self.logger.debug("[id:%s] Recieved result", msg.id)

                            if fut is not None:
                                msg.result.to_future(fut)
                            else:
                                self.logger.debug("[id:%s] Discarding result", msg.id)

                if self._shutdown_event.is_set():
                    break

        self._poller = Thread(target=_poller, name=f"ClientPoller[{self._name}]")
        self._poller.daemon = True
        self._poller.start()

    def _cancellation_cb(self, fut):
        if fut.cancelled():
            with self._send_lock:
                self._conn.send(Cancellation(fut.id))

    def _make_future(self):
        f = RPCFuture(timeout=self._timeout)
        f.add_done_callback(self._cancellation_cb)
        return f

    def _invoke(self, method_name, *args, **kwargs):
        # request id, method, args, kwargs
        if self._shutdown_event.is_set():
            raise Exception("Cannot connect to server")

        id_ = self._new_id()
        self.logger.debug("[id:%s] %s call '%s' method", id_, self._name, method_name)
        self._request_by_id[id_] = f = self._make_future()
        f.id = id_
        with self._send_lock:
            self._conn.send(MethodCall(id_, method_name, args, kwargs))
        return f

    def _shutdown(self, exc):
        self._shutdown_event.set()
        for fut in self._request_by_id.values():
            fut.set_exception(exc)


class Message:
    def __init__(self, id_):
        self.id = id_


class Signal:
    def __init__(self, payload):
        self.payload = payload


class MethodCall(Message):
    def __init__(self, id_, method_name, args, kwargs):
        super().__init__(id_)
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs


class Cancellation(Message):
    pass


class MethodReturn(Message):
    def __init__(self, id_, result: Result):
        super().__init__(id_)
        self.result = result


class Stop(Exception):
    pass


class FutureStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._future_by_id = {}
        self._id_by_future = {}

    def put(self, id_: str, fut: Future):
        with self._lock:
            self._future_by_id[id_] = fut
            self._id_by_future[fut] = id_

    def pop_id(self, id_: str) -> Optional[Future]:
        with self._lock:
            if id_ not in self._future_by_id:
                return None

            fut = self._future_by_id.pop(id_, None)
            del self._id_by_future[fut]
            return fut

    def pop_future(self, fut: Future) -> Optional[str]:
        with self._lock:
            if fut not in self._id_by_future:
                return None

            id_ = self._id_by_future.pop(fut, None)
            del self._future_by_id[id_]
            return id_


class MPServer:
    _sentinel = object()

    def __init__(self, api, conn: Connection):
        self._api = api
        self._futures = FutureStore()
        self._logger = None
        self._conn = conn
        self._results_queue = queue.Queue()
        self._start_result_sender(conn)

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        return self._logger

    def _send(self, msg):
        self._results_queue.put(msg)

    def _start_result_sender(self, conn):
        def _sender():
            while True:
                try:
                    result = self._results_queue.get()
                    if result is self._sentinel:
                        break

                    conn.send(result)
                except Exception:
                    self.logger.exception("Error in result sender")

        t = threading.Thread(target=_sender, name="MPResultSender")
        t.start()

    def _send_result(self, fut):
        if fut.cancelled():
            return

        id_ = self._futures.pop_future(fut)

        if id_:
            self.logger.debug("[id: %s] Sending result", id_)
            try:
                self._send(MethodReturn(id_, Result.OK(fut.result())))
            except Exception as e:
                self._send(MethodReturn(id_, Result.Error(e)))
        else:
            self.logger.warning("Discarding result for future %s", fut)

    def _make_future(self):
        f = RPCFuture()
        f.add_done_callback(self._send_result)
        return f

    def _call_method(self, call: MethodCall):
        self.logger.debug("[id: %s] Recieved '%s' method call", call.id, call.method_name)
        fut = self._make_future()
        self._futures.put(call.id, fut)

        try:
            meth = getattr(self._api, call.method_name)
            res = meth(*call.args, **call.kwargs)

        except Exception as e:
            fut.set_exception(e)

        else:
            if isinstance(res, Shutdown):
                fut.set_result(Shutdown())
                raise Stop()

            if isinstance(res, Future):
                fut.attach(res)
            else:
                fut.set_result(res)

        return fut

    def _cancel_request(self, cancel: Cancellation):
        self.logger.debug("[id: %s] Recieved cancel request", cancel.id)
        fut = self._futures.pop_id(cancel.id)
        if fut:
            fut.cancel()
            self.logger.debug("[id: %s] Cancelled", cancel.id)

    def listen(self):
        while True:
            ready = self._conn.poll(timeout=1)
            if not ready:
                continue

            try:
                msg = self._conn.recv()

                if isinstance(msg, MethodCall):
                    try:
                        self._call_method(msg)
                    except Stop:
                        self.logger.debug("[id: %s] Shutdown", msg.id)
                        self._send(Signal(b"shutdown"))
                        self._send(self._sentinel)
                        break

                elif isinstance(msg, Cancellation):
                    self._cancel_request(msg)

            except EOFError:
                self.logger.error("Broken Pipe")
                break
            except Exception:
                self.logger.error("Error in main loop", exc_info=1)
