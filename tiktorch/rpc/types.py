import inspect
import threading
from concurrent.futures import Future
from typing import Callable, Generic, Type, TypeVar, _GenericAlias

T = TypeVar("T")
S = TypeVar("S")


def _map_future(fut: Future, func: Callable[[T], S]) -> "RPCFuture[S]":
    new_fut: RPCFuture[S] = RPCFuture()

    def _do_map(f):
        try:
            res = func(f.result())
            new_fut.set_result(res)
        except Exception as e:
            new_fut.set_exception(e)

    fut.add_done_callback(_do_map)
    return new_fut


class _OneShotConnection:
    """
    Copies state from one completed future to another
    Used to propagate results and cancelations
    """

    __slots__ = ("_fired", "_lock", "_fut1", "_fut2")

    def __init__(self, fut1: Future, fut2: Future) -> None:
        self._fired = False
        self._lock = threading.Lock()

        self._fut1 = fut1
        self._fut1.add_done_callback(self._handle_fut1_completion)

        self._fut2 = fut2
        self._fut2.add_done_callback(self._handle_fut2_completion)

    def _copy(self, src: Future, dst: Future) -> None:
        gotit = self._lock.acquire(blocking=False)
        if not gotit:
            return

        try:
            if self._fired:
                return

            self._fired = True

            if src.cancelled():
                dst.cancel()
                return

            try:
                dst.set_result(src.result())
            except Exception as e:
                dst.set_exception(e)
        finally:
            self._lock.release()

    def _handle_fut1_completion(self, fut):
        assert fut == self._fut1
        self._copy(fut, self._fut2)

    def _handle_fut2_completion(self, fut):
        assert fut == self._fut2
        self._copy(fut, self._fut1)


class RPCFuture(Future, Generic[T]):
    def __init__(self, timeout=None):
        self._timeout = timeout
        super().__init__()

    def attach(self, future):
        _OneShotConnection(self, future)

    def result(self, timeout=None):
        return super().result(timeout or self._timeout)

    def exception(self, timeout=None):
        return super().exception(timeout or self._timeout)

    def map(self, func: Callable[[T], S]) -> "RPCFuture[S]":
        """
        Apply function and return new future
        Note: Function should return plain object not wrapped in future

        >>> fut = RPCFuture()
        >>> new_fut = fut.map(lambda val: val + 1)
        >>> fut.set_result(12)
        >>> new_fut.result()
        13

        >>> fut = RPCFuture()
        >>> new_fut = fut.map(lambda val: val / 0)
        >>> fut.set_result(12)
        >>> new_fut.result()
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero
        """
        return _map_future(self, func)


def _checkgenericfut(type_: Type) -> bool:
    if isinstance(type_, _GenericAlias):
        origin = getattr(type_, "__origin__", None)
        return origin and issubclass(origin, Future)
    return False


def isfutureret(func: Callable):
    sig = inspect.signature(func)
    ret = sig.return_annotation

    return (inspect.isclass(ret) and issubclass(sig.return_annotation, Future)) or _checkgenericfut(ret)
