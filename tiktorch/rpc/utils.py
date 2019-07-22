from concurrent.futures import Future
from threading import Lock

from tiktorch.rpc.types import RPCFuture

_undef = object()


class BatchedExecutor:
    def __init__(self, batch_size=20):
        self._batch_size = batch_size
        self._lock = Lock()
        self._in_flight_count = 0
        self._pending = []

    def _is_valid_return(self, value):
        if not isinstance(value, Future):
            raise ValueError("Expected all submitted jobs to return Future")

    def _submit_new_request(self, res=_undef):
        with self._lock:
            if res is not _undef:
                self._in_flight_count -= 1

            if self._pending and self._in_flight_count < self._batch_size:
                fn, args, kwargs, user_fut = self._pending.pop()

                remote_fut = fn(*args, **kwargs)
                self._is_valid_return(remote_fut)
                remote_fut.add_done_callback(self._submit_new_request)

                user_fut.attach(remote_fut)
                self._in_flight_count += 1

        return res

    def submit(self, function, *args, **kwargs):
        f = RPCFuture()
        self._pending.append((function, args, kwargs, f))
        self._submit_new_request()
        return f
