import inspect

from concurrent.futures import Future
from typing import Generic, TypeVar, Callable, NamedTuple, Tuple, List


T = TypeVar("T")


class RPCFuture(Future, Generic[T]):
    def __init__(self, timeout=None):
        self._timeout = timeout
        super().__init__()

    def result(self, timeout=None):
        return super().result(timeout or self._timeout)

    def exception(self, timeout=None):
        return super().exception(timeout or self._timeout)


def isfutureret(func: Callable):
    sig = inspect.signature(func)
    return sig.return_annotation is not None and issubclass(sig.return_annotation, Future)


class SetDeviceReturnType(NamedTuple):
    training_shape: Tuple[int, int, int, int, int]
    valid_shapes: List[Tuple[int, int, int, int, int]]
    shrinkage: Tuple[int, int, int, int, int]
