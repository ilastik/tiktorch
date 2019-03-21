import multiprocessing as mp

from uuid import uuid4
from threading import Thread
from concurrent.futures import Future
from multiprocessing.connection import Connection
from threading import Event
from typing import Any

from .exceptions import Shutdown
from .interface import RPCInterface, get_exposed_methods


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

    def __call__(self, *args, **kwargs) -> Any:
        return self._client._invoke(self._method_name, *args, **kwargs)


class MPClient:
    def __init__(self, api, conn: Connection):
        self._conn = conn
        self._requests = {}
        self._api = api
        self._methods_by_name = get_exposed_methods(api)

        self._start_poller()
        self._shutdown_event = Event()

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
                            self._shutdown()

                    # method
                    else:
                        fut = self._requests.pop(id_, None)

                        if fut is not None:
                            res.to_future(fut)

                if self._shutdown_event.is_set():
                    break

        self._poller = Thread(target=_poller, name='ClientPoller')
        self._poller.start()

    def _invoke(self, method_name, *args, **kwargs):
        # request id, method, args, kwargs
        id_ = self._new_id()
        self._requests[id_] = Future()
        self._conn.send([id_, method_name, args, kwargs])
        return self._requests[id_]

    def _shutdown(self):
        self._shutdown_event.set()


class MPServer:
    def __init__(self, api, conn: Connection):
        self._conn = conn
        self._api = api

    def listen(self):
        while True:
            ready = self._conn.poll(timeout=1)
            if not ready:
                continue

            try:
                id_, method_name, args, kwargs = self._conn.recv()
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
                self._conn.send([id_, Result.OK(res)])
