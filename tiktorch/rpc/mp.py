import logging
import multiprocessing as mp
import types

from uuid import uuid4
from threading import Thread
from concurrent.futures import Future
from multiprocessing.connection import Connection
from threading import Event
from typing import Any, Type, TypeVar
from functools import wraps

from .exceptions import Shutdown
from .interface import RPCInterface, get_exposed_methods
from .types import RPCFuture


logger = logging.getLogger(__name__)


T = TypeVar('T')

class Result:
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


class MPMethodDispatcher:
    def __init__(self, method_name, client):
        self._method_name = method_name
        self._client = client

    def sync(self, *args, **kwargs):
        f = self(*args, **kwargs)
        return f.result()

    def __call__(self, *args, **kwargs) -> Any:
        return self._client._invoke(self._method_name, *args, **kwargs)

import inspect

def create_client(iface_cls: Type[T], conn: Connection) -> T:
    client = MPClient(iface_cls(), conn)
    exposed = get_exposed_methods(iface_cls)

    def _make_method(method):
        sig = inspect.signature(method)
        is_future_ret = issubclass(sig.return_annotation, Future)

        class MethodWrapper:
            @wraps(method)
            def async_(self, *args, **kwargs):
                return client._invoke(method.__name__, *args, **kwargs)

            if is_future_ret:
                @wraps(method)
                def __call__(self, *args, **kwargs) -> Any:
                    return self.async_(*args, **kwargs)
            else:
                @wraps(method)
                def __call__(self, *args, **kwargs) -> Any:
                    fut = client._invoke(method.__name__, *args, **kwargs)
                    return fut.result(timeout=10)

        return MethodWrapper()

    class _Client(iface_cls):
        pass

    for method_name, method in get_exposed_methods(iface_cls).items():
        setattr(_Client, method_name, _make_method(method))

    return _Client()


class MPClient:
    def __init__(self, api, conn: Connection):
        self._conn = conn
        self._requests = {}
        self._api = api
        self._methods_by_name = get_exposed_methods(api)

        self._start_poller()
        self._shutdown_event = Event()
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        return self._logger

    def __getattr__(self, name) -> Any:
        method = self._methods_by_name.get(name)
        if method is None:
            raise AttributeError(name)
        return MPMethodDispatcher(name, self)

    def _new_id(self) -> str:
        return uuid4().hex

    def _start_poller(self):

        def _poller():
            while True:
                if self._conn.poll(timeout=1):
                    id_, res = self._conn.recv()

                    # signal
                    if id_ is None:
                        if res == b'shutdown':
                            self.logger.debug('[signal] Shutdown')
                            self._shutdown()

                    # method
                    else:
                        fut = self._requests.pop(id_, None)
                        self.logger.debug('[id:%s] Recieved result', id_)

                        if fut is not None:
                            res.to_future(fut)

                if self._shutdown_event.is_set():
                    break

        self._poller = Thread(target=_poller, name='ClientPoller')
        self._poller.start()

    def _invoke(self, method_name, *args, **kwargs):
        # request id, method, args, kwargs
        id_ = self._new_id()
        self.logger.debug("[id:%s] Call '%s' method", id_, method_name)
        self._requests[id_] = Future()
        self._conn.send([id_, method_name, args, kwargs])
        return self._requests[id_]

    def _shutdown(self):
        self._shutdown_event.set()


class MPServer:
    def __init__(self, api, conn: Connection):
        self._conn = conn
        self._api = api
        self._futures = {}
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        return self._logger

    def _send_result(self, fut):
        id_ = self._futures.pop(fut, None)
        self.logger.debug('[id: %s] Sending result', id_)
        if id_:
            self._conn.send([id_, Result.OK(fut.result())])

    def listen(self):
        while True:
            ready = self._conn.poll(timeout=1)
            if not ready:
                continue

            try:
                id_, method_name, args, kwargs = self._conn.recv()
                self.logger.debug("[id: %s] Recieved '%s' method call", id_, method_name)
                # TODO: handle signals

                meth = getattr(self._api, method_name)
                res = meth(*args, **kwargs)
            except Shutdown:
                self._conn.send([id_, Result.OK(None)])
                self._conn.send([None, b'shutdown'])
                break
            except Exception as e:
                self._conn.send([id_, Result.Error(e)])
            else:
                if isinstance(res, Future):
                    self._futures[res] = id_
                    res.add_done_callback(self._send_result)
                else:
                    self._conn.send([id_, Result.OK(res)])
