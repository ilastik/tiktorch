from concurrent.futures import Future
from typing import Generic, TypeVar


T = TypeVar('T')


class RPCFuture(Future, Generic[T]):
    pass
