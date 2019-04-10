import inspect

from concurrent.futures import Future
from typing import Generic, TypeVar, Callable, Tuple, List


T = TypeVar("T")
S = TypeVar("S")


class RPCFuture(Future, Generic[T]):
    def __init__(self, timeout=None):
        self._timeout = timeout
        super().__init__()

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
        new_fut: RPCFuture[S] = RPCFuture()

        def _do_map(f):
            try:
                res = func(f.result())
                new_fut.set_result(res)
            except Exception as e:
                new_fut.set_exception(e)

        self.add_done_callback(_do_map)
        return new_fut


def isfutureret(func: Callable):
    sig = inspect.signature(func)
    return inspect.isclass(sig.return_annotation) and issubclass(sig.return_annotation, Future)
