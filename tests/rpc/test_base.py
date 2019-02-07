# TODO: Timeout tests
from time import sleep
from threading import Thread
from functools import wraps

import pytest
import zmq

from tiktorch.rpc.base import (
    RPCInterface, get_exposed_methods,
    serialize_args, deserialize_args,
    deserialize_return, serialize_return,
    Server, Client, Shutdown
)


class Iface(RPCInterface):
    def foo(self):
        raise NotImplementedError


class API(Iface):
    def foo(self):
        pass

    def bar(self):
        pass


def dec(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class Foo:
    def func(self, data: dict, a: bytes) -> bytes:
        raise NotImplementedError

    @dec
    def func_dec(self, data: dict, a: bytes) -> bytes:
        raise NotImplementedError


def test_get_exposed_methods():
    api = API()

    assert get_exposed_methods(api) == {
        'foo': api.foo,
    }, "should return all bound methods"

    assert get_exposed_methods(API) == {
        'foo': API.foo,
    }, "should return all class functions in"


def test_serialize_deserialize_method_args():
    f = Foo()
    data = {'a': 1}
    a = b'hello'
    serialized = list(serialize_args(f.func, [data, a]))

    assert len(serialized) == 2
    for frame in serialized:
        assert isinstance(frame, zmq.Frame)

    deserialized = deserialize_args(f.func, iter(serialized))
    assert len(deserialized) == 2
    assert deserialized == [data, a]


def test_serialize_deserialize_method_return():
    f = Foo()
    serialized = list(serialize_return(f.func, b'bytes'))

    assert len(serialized) == 1
    deserialized = deserialize_return(f.func, iter(serialized))
    assert deserialized == b'bytes'


def test_serialize_deserialize_decorated_method_args():
    f = Foo()
    data = {'a': 1}
    a = b'hello'
    serialized = list(serialize_args(f.func_dec, [data, a]))

    assert len(serialized) == 2
    for frame in serialized:
        assert isinstance(frame, zmq.Frame)

    deserialized = deserialize_args(f.func_dec, iter(serialized))
    assert len(deserialized) == 2
    assert deserialized == [data, a]


def test_serialize_deserialize_decoratedd_method_return():
    f = Foo()
    serialized = list(serialize_return(f.func_dec, b'bytes'))

    assert len(serialized) == 1
    deserialized = deserialize_return(f.func_dec, iter(serialized))
    assert deserialized == b'bytes'


class IConcatRPC(RPCInterface):
    def concat(self, a: bytes, b: bytes) -> bytes:
        raise NotImplementedError

    def none_return(self) -> None:
        return None

    def shutdown(self) -> None:
        raise Shutdown()


class ConcatRPCSrv(IConcatRPC):
    def concat(self, a: bytes, b: bytes) -> bytes:
        return a + b

    def not_exposed(self) -> None:
        pass


def test_server():
    ctx = zmq.Context()

    def _target():
        api = ConcatRPCSrv()
        socket = ctx.socket(zmq.PAIR)
        socket.bind(f'inproc://test')
        srv = Server(api, socket)
        srv.listen()

    t = Thread(target=_target)
    t.start()

    socket = ctx.socket(zmq.PAIR)
    socket.connect(f'inproc://test')
    cl = Client(IConcatRPC(), socket)

    resp = cl.concat(b'foo', b'bar')
    assert resp == b'foobar'

    resp = cl.shutdown()
    t.join(timeout=2)
    assert not t.is_alive()


def test_client_dir():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    cl = Client(IConcatRPC(), socket)

    methods = dir(cl)

    assert 'concat' in methods
    assert 'shutdown' in methods

    assert 'not_exposed' not in methods


def test_method_returning_none():
    ctx = zmq.Context()

    def _target():
        api = ConcatRPCSrv()
        socket = ctx.socket(zmq.PAIR)
        socket.bind(f'inproc://test')
        srv = Server(api, socket)
        srv.listen()

    t = Thread(target=_target)
    t.start()

    socket = ctx.socket(zmq.PAIR)
    socket.connect(f'inproc://test')
    cl = Client(IConcatRPC(), socket)

    res = cl.none_return()


def test_error_doesnt_stop_server():
    class Foo:
        pass

    class SomeRPC(RPCInterface):
        def ping(self) -> bytes:
            return b'pong'

        def raise_exc(self) -> None:
            raise Exception('fail')

        def unknown_return_type(self) -> Foo:
            return Foo()

        def unknown_arg_type(self, f: Foo) -> bytes:
            return 'ok'

        def shutdown(self):
            raise Shutdown()


    ctx = zmq.Context()

    def _target():
        api = SomeRPC()
        socket = ctx.socket(zmq.PAIR)
        socket.bind(f'inproc://test')
        srv = Server(api, socket)
        srv.listen()

    t = Thread(target=_target)
    t.start()

    socket = ctx.socket(zmq.PAIR)
    # TODO: Check if socket is connected
    socket.connect('inproc://test')
    cl = Client(SomeRPC(), socket)

    assert cl.ping() == b'pong'
    assert t.is_alive()

    with pytest.raises(Exception):
        cl.raise_exc()

    assert t.is_alive()
    assert cl.ping() == b'pong'

    with pytest.raises(Exception):
        cl.unknown_return_type()

    assert t.is_alive()
    assert cl.ping() == b'pong'

