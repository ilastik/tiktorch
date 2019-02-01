from typing import Any, List, Generic, Iterator, TypeVar, Type, Mapping, Callable

import zmq

T = TypeVar('T')


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
    serializer = _serializer_by_type.get(type_)
    if not serializer:
        raise NotImplementedError(f"Serialization protocol not implemented for {type_}")

    yield from serializer.serialize(obj)


def deserialize(type_: type, frames: Iterator[zmq.Frame]) -> Any:
    """
    Deserialize object of type *type_* from frames stream
    """
    serializer = _serializer_by_type.get(type_)
    if not serializer:
        raise NotImplementedError(f"Serialization protocol not implemented for {type_}")

    return serializer.deserialize(frames)


#: Registry of type serialziers
_serializer_by_type = {}  # Mapping[Type[T], ISerializer]
