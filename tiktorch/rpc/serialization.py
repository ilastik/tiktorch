from logging import getLogger
from typing import Any, List, Generic, Iterator, TypeVar, Type, Mapping, Callable

import zmq

T = TypeVar('T')
logger = getLogger(__name__)


class ISerializer(Generic[T]):
    """
    Serializer interface
    """
    @classmethod
    def deserialize(cls, frames: Iterator[zmq.Frame]) -> T:
        """
        Deserialize objects from zmq.Frame stream
        This function fully consume relevant part of iterator
        """
        raise NotImplementedError

    @classmethod
    def serialize(cls, obj: T) -> Iterator[zmq.Frame]:
        """
        Serialize object to zmq.Frame stream
        """
        raise NotImplementedError


def serializer_for(type_: ISerializer) -> Callable[[Type[T]], Type[T]]:
    """
    Register serializer for given type
    """
    def register(cls):
        _serializer_by_type[type_] = cls
        return cls

    return register


def serialize(type_: type, obj: Any) -> Iterator[zmq.Frame]:
    """
    Serialize single object of type *type_* to zmq.Frame stream
    """
    logger.debug("serialize(%r, %r)", type_, obj)
    serializer = _serializer_by_type.get(type_)
    if not serializer:
        raise NotImplementedError(f"Serialization protocol not implemented for {type_}")
    logger.debug("Using %r serializer", serializer)

    yield from serializer.serialize(obj)


def deserialize(type_: type, frames: Iterator[zmq.Frame]) -> Any:
    """
    Deserialize object of type *type_* from frames stream
    """
    logger.debug("deserialize(%r)", type_)
    serializer = _serializer_by_type.get(type_)
    if not serializer:
        raise NotImplementedError(f"Serialization protocol not implemented for {type_}")
    logger.debug("Using %r serializer", serializer)

    return serializer.deserialize(frames)


#: Registry of type serialziers
_serializer_by_type = {}  # Mapping[Type[T], ISerializer]


@serializer_for(None)
class NoneSerializer(ISerializer[None]):
    @classmethod
    def deserialize(cls, frames: Iterator[zmq.Frame]) -> None:
        """
        Deserialize objects from zmq.Frame stream
        This function fully consume relevant part of iterator
        """
        frame = next(frames)

        if frame.bytes == b'':
            return None

        raise ValueError(f"None frame shouldn't contain any data")

    @classmethod
    def serialize(cls, obj: None) -> Iterator[zmq.Frame]:
        """
        Serialize object to zmq.Frame stream
        """
        yield zmq.Frame()


@serializer_for(bytes)
class BytesSerializer(ISerializer[bytes]):
    @classmethod
    def deserialize(cls, frames: Iterator[zmq.Frame]) -> None:
        """
        Deserialize objects from zmq.Frame stream
        This function fully consume relevant part of iterator
        """
        frame = next(frames)

        return frame.bytes

    @classmethod
    def serialize(cls, obj: bytes) -> Iterator[zmq.Frame]:
        """
        Serialize object to zmq.Frame stream
        """
        yield zmq.Frame(obj)


@serializer_for(memoryview)
class MemoryViewSerializer(ISerializer[memoryview]):
    @classmethod
    def deserialize(cls, frames: Iterator[zmq.Frame]) -> memoryview:
        frame = next(frames)

        return frame.buffer

    @classmethod
    def serialize(cls, obj: memoryview) -> Iterator[zmq.Frame]:
        yield zmq.Frame(obj)
