from time import sleep
from threading import Thread
from functools import wraps

import pytest
import zmq

from tiktorch.rpc.base import (
    RPCInterface, exposed, get_exposed_methods,
    serialize_args, deserialize_args,
    deserialize_return, serialize_return,
    Server, Client, Shutdown
)


class Iface(RPCInterface):
    @exposed
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
    @exposed
    def concat(self, a: bytes, b: bytes) -> bytes:
        raise NotImplementedError

    @exposed
    def none_return(self) -> None:
        return None

    @exposed
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
        srv = Server(api, 'inproc://test', ctx)
        srv.listen()

    t = Thread(target=_target)
    t.start()

    cl = Client(IConcatRPC(), f'inproc://test', ctx)

    resp = cl.concat(b'foo', b'bar')
    assert resp == b'foobar'

    resp = cl.shutdown()
    t.join(timeout=2)
    assert not t.is_alive()


def test_client_dir():
    cl = Client(IConcatRPC(), 'inproc://test')

    methods = dir(cl)

    assert 'concat' in methods
    assert 'shutdown' in methods

    assert 'not_exposed' not in methods


def test_method_returning_none():
    ctx = zmq.Context()

    def _target():
        api = ConcatRPCSrv()
        srv = Server(api, 'inproc://test', ctx)
        srv.listen()

    t = Thread(target=_target)
    t.start()

    cl = Client(IConcatRPC(), 'inproc://test', ctx)

    res = cl.none_return()
    cl.shutdown()


def test_error_doesnt_stop_server():
    class Foo:
        pass

    class SomeRPC(RPCInterface):
        @exposed
        def ping(self) -> bytes:
            return b'pong'

        @exposed
        def raise_exc(self) -> None:
            raise Exception('fail')

        @exposed
        def unknown_return_type(self) -> Foo:
            return Foo()

        @exposed
        def unknown_arg_type(self, f: Foo) -> bytes:
            return 'ok'

        @exposed
        def shutdown(self):
            raise Shutdown()


    ctx = zmq.Context()

    def _target():
        api = SomeRPC()
        srv = Server(api, 'inproc://test', ctx)
        srv.listen()

    t = Thread(target=_target)
    t.start()

    cl = Client(SomeRPC(), 'inproc://test', ctx)

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

    cl.shutdown()
    assert not t.is_alive()


def test_multithreaded():
    class SomeRPC(RPCInterface):
        @exposed
        def ping(self) -> bytes:
            sleep(0.1)
            return b'pong'

        @exposed
        def shutdown(self) -> None:
            raise Shutdown()

    ctx = zmq.Context()

    def _target():
        api = SomeRPC()
        srv = Server(api, f'inproc://test', ctx)
        srv.listen()

    t = Thread(target=_target)
    t.start()

    cl = Client(SomeRPC(), 'inproc://test', ctx)

    res = []
    def _client():
        res.append(cl.ping() == b'pong')

    clients = []
    for _ in range(5):
        t = Thread(target=_client)
        t.start()
        clients.append(t)

    for c in clients:
        c.join(timeout=1)
        assert not c.is_alive()

    cl.shutdown()

    assert len(res) == 5
    assert all(res)


def test_rpc_interface_metaclass():
    class IBar(RPCInterface):
        @exposed
        def bar(self) -> None:
            return

    class IFoo(RPCInterface):
        @exposed
        def foo(self) -> None:
            return

    class Foo(IFoo, IBar):
        def foo(self):
            return None

        def foobar(self):
            return None

    assert Foo.__exposedmethods__ == {'foo', 'bar'}
