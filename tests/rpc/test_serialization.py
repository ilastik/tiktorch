import pytest

import json

from itertools import chain
from typing import Iterator

import zmq

from tiktorch.rpc.serialization import serializer_for, ISerializer, serialize, deserialize, DeserializationError
from tiktorch.rpc.serialization import _serializer_by_type


class Foo:
    def __init__(self, a):
        self.a = a

    def __eq__(self, other):
        return self.a == other.a


class Bar:
    def __init__(self, b: int, c: int):
        self.b = b
        self.c = c

    def __eq__(self, other):
        return (
            self.b == other.b
            and self.c == other.c
        )


@pytest.fixture(autouse=True)
def clean_registry():
    # Implementation detail, reset registry for each test
    _serializer_by_type.clear()


class FooSerializer(ISerializer[Foo]):
    @classmethod
    def serialize(cls, obj: Foo) -> Iterator[zmq.Frame]:
        obj_bytes = json.dumps(obj.__dict__).encode('ascii')
        yield zmq.Frame(obj_bytes)

    @classmethod
    def deserialize(cls, frames: Iterator[zmq.Frame]) -> Foo:
        frame = next(frames)
        dct = json.loads(frame.bytes.decode('ascii'))
        return Foo(**dct)


class BarSerializer(ISerializer[Bar]):
    @classmethod
    def serialize(cls, obj: Bar) -> Iterator[zmq.Frame]:
        yield zmq.Frame(obj.b.to_bytes(length=1, byteorder='little'))
        yield zmq.Frame(obj.c.to_bytes(length=1, byteorder='little'))

    @classmethod
    def deserialize(cls, frames: Iterator[zmq.Frame]) -> Foo:
        frame_b = next(frames)
        frame_c = next(frames)
        return Bar(
            b=int.from_bytes(frame_b, byteorder='little'),
            c=int.from_bytes(frame_c, byteorder='little'),
        )


def test_undefinined_serializer_raises():
    with pytest.raises(NotImplementedError):
        res = list(serialize(Foo, Foo(a=1)))

    with pytest.raises(NotImplementedError):
        res = list(deserialize(Foo, zmq.Frame(b'{"a": 1}')))


def test_defining_serializer():
    serializer_for(Foo)(FooSerializer)

    frame, *rest = list(serialize(Foo, Foo(a=42)))
    assert len(rest) == 0
    assert frame.bytes == b'{"a": 42}'

    foo = deserialize(Foo, iter([frame]))
    assert foo.a == 42


def test_multiframe_parser():
    serializer_for(Bar)(BarSerializer)

    frames = list(serialize(Bar, Bar(b=42, c=21)))
    assert len(frames) == 2
    assert frames[0].bytes == b'*'
    assert frames[1].bytes == b'\x15'


def test_deserializing_multiple_objects_from_frame_stream():
    serializer_for(Foo)(FooSerializer)
    serializer_for(Bar)(BarSerializer)

    expected_bar = Bar(b=42, c=21)
    expected_foo = Foo(a=1)
    bar_frames = serialize(Bar, expected_bar)
    foo_frames = serialize(Foo, expected_foo)
    tail = ('ok' for _ in range(1))
    # foo = deserialize(Foo, iter([frame]))

    frames = chain(bar_frames, foo_frames, tail)


    bar = deserialize(Bar, frames)
    assert bar == expected_bar
    assert bar is not expected_bar

    foo = deserialize(Foo, frames)
    assert foo == expected_foo
    assert foo is not expected_foo

    assert next(frames) == 'ok', "Iterator should consumed"


def test_deserializing_from_empty_iterator_should_raise():
    serializer_for(Foo)(FooSerializer)

    with pytest.raises(DeserializationError):
        foo = deserialize(Foo, iter([]))
