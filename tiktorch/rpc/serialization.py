from logging import getLogger
from typing import Any, List, Generic, Iterator, TypeVar, Type, Mapping, Callable
from concurrent.futures import Future

import zmq

from zmq.utils import jsonapi


T = TypeVar('T')
logger = getLogger(__name__)


class ISerializer(Generic[T]):
    """
    Serializer interface
    """
    @classmethod
    def deserialize(cls, frames: 'FusedFrameIterator') -> T:
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


class SerializerRegistry:
    """
    Contains all registered serializers for types
    """
    def __init__(self):
        self._serializer_by_type: Dict[Any, ISerializer] = {}

    def register(self, type_: T) -> Callable[[ISerializer[T]], Any]:
        """
        Register serializer for given type
        """
        def _reg_fn(cls: ISerializer[T]) -> ISerializer[T]:
            self._serializer_by_type[type_] = cls
            return cls

        return _reg_fn

    def serialize(self, type_: type, obj: Any) -> Iterator[zmq.Frame]:
        """
        Serialize single object of type *type_* to zmq.Frame stream
        """
        logger.debug("serialize(%r, %r)", type_, obj)
        serializer = self._serializer_by_type.get(type_)
        if not serializer:
            raise NotImplementedError(f"Serialization protocol not implemented for {type_}")
        logger.debug("Using %r serializer", serializer)

        yield from serializer.serialize(obj)

    def deserialize(self, type_: type, frames: Iterator[zmq.Frame]) -> Any:
        """
        Deserialize object of type *type_* from frames stream
        """
        logger.debug("deserialize(%r)", type_)
        serializer = self._serializer_by_type.get(type_)
        if not serializer:
            raise NotImplementedError(f"Serialization protocol not implemented for {type_}")
        logger.debug("Using %r serializer", serializer)

        return serializer.deserialize(FusedFrameIterator(type_, frames))


class DeserializationError(Exception):
    pass


class FusedFrameIterator(Iterator[zmq.Frame]):
    """
    Iterate over frames and raise a meaningful error if frame stream is exhausted
    """
    def __init__(self, type_: T, iter_: Iterator[zmq.Frame]) -> None:
        self._iter = iter_
        self._type = type_

    def __iter__(self):
        return self

    def __next__(self) -> zmq.Frame:
        try:
            return next(self._iter)
        except StopIteration:
            raise DeserializationError(
                f'Not enought frames to deserialize {self._type}'
            ) from None



root_reg = SerializerRegistry()
serialize = root_reg.serialize
deserialize = root_reg.deserialize
register = root_reg.register
# TODO Remove
serializer_for = root_reg.register

@serializer_for(None)
class NoneSerializer(ISerializer[None]):
    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> None:
        """
        Deserialize objects from zmq.Frame stream
        This function fully consume relevant part of iterator
        """
        frame = next(frames)

        if frame.bytes == b'':
            return None

        raise DeserializationError(f"None frame shouldn't contain any data")

    @classmethod
    def serialize(cls, obj: None) -> Iterator[zmq.Frame]:
        """
        Serialize object to zmq.Frame stream
        """
        yield zmq.Frame()


@serializer_for(bytes)
class BytesSerializer(ISerializer[bytes]):
    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> bytes:
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
    def deserialize(cls, frames: FusedFrameIterator) -> memoryview:
        frame = next(frames)

        return frame.buffer

    @classmethod
    def serialize(cls, obj: memoryview) -> Iterator[zmq.Frame]:
        yield zmq.Frame(obj)


@serializer_for(dict)
class DictSerializer(ISerializer[dict]):
    @classmethod
    def serialize(cls, obj: dict) -> Iterator[zmq.Frame]:
        yield zmq.Frame(jsonapi.dumps(obj))

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> dict:
        frame = next(frames)
        return jsonapi.loads(frame.bytes)


@serializer_for(bool)
class BoolSerializer(ISerializer[bool]):
    @classmethod
    def serialize(cls, obj: bool) -> Iterator[zmq.Frame]:
        yield zmq.Frame(b'1' if obj else b'')

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> bool:
        frame = next(frames)
        return bool(frame.bytes)
