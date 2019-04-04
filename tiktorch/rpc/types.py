import inspect

from concurrent.futures import Future
from typing import Generic, TypeVar, Callable


T = TypeVar('T')


class RPCFuture(Future, Generic[T]):
    pass


def isfutureret(func: Callable):
    sig = inspect.signature(func)
    return (
        sig.return_annotation is not None
        and issubclass(sig.return_annotation, Future)
    )
