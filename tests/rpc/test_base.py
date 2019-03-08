from time import sleep
from threading import Thread, enumerate as tenum
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

import pytest
import zmq

from tiktorch.rpc.base import (
    serialize_args, deserialize_args,
    deserialize_return, serialize_return,
    Server, Client, RPCFuture, isfutureret
)
from tiktorch.rpc.connections import InprocConnConf
from tiktorch.rpc.interface import exposed, RPCInterface, get_exposed_methods
from tiktorch.rpc.exceptions import Shutdown, Timeout, Canceled, CallException


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


import logging
import logging.config

logger = logging.getLogger(__name__)


@pytest.fixture
def log_debug():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'default': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    })


@pytest.fixture
def spawn(conn_conf):
    d = {
        'client': None,
        'thread': None,
    }

    def _spawn(iface, service):
        def _target():
            api = service()
            srv = Server(api, conn_conf)
            srv.listen()

        t = Thread(target=_target, name='TestServerThread')
        t.start()

        d['thread'] = t
        d['client'] = Client(iface(), conn_conf)

        return d['client']

    yield _spawn

    if d['client']:
        with pytest.raises(Shutdown):
            d['client'].shutdown()

        d['thread'].join()

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


def test_serialize_deserialize_decorated_method_return():
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
    def concat_async(self, a: bytes, b: bytes) -> RPCFuture[bytes]:
        raise NotImplementedError

    @exposed
    def concat_broken_async(self, a: bytes, b: bytes) -> RPCFuture[bytes]:
        raise NotImplementedError

    @exposed
    def raise_outside_future(self, a: bytes, b: bytes) -> RPCFuture[bytes]:
        raise NotImplementedError

    @exposed
    def none_return(self) -> None:
        return None

    @exposed
    def shutdown(self) -> None:
        raise Shutdown()


class ConcatRPCSrv(IConcatRPC):
    def __init__(self):
        self.executor = ThreadPoolExecutor()

    def concat(self, a: bytes, b: bytes) -> bytes:
        return a + b

    def concat_async(self, a: bytes, b: bytes) -> RPCFuture[bytes]:

        def _slow_concat(a, b):
            sleep(0.4)
            return a + b

        return self.executor.submit(_slow_concat, a, b)

    def concat_broken_async(self, a: bytes, b: bytes) -> RPCFuture[bytes]:
        def _broken(a, b):
            raise Exception('broken')

        return self.executor.submit(_broken, a, b)

    def raise_outside_future(self, a: bytes, b: bytes) -> RPCFuture[bytes]:
        raise Exception('broken')

    def not_exposed(self) -> None:
        pass

@pytest.fixture
def conn_conf():
    ctx = zmq.Context()
    return InprocConnConf('test', 'pubsub_test', ctx, timeout=2000)


def test_server(spawn):
    cl = spawn(IConcatRPC, ConcatRPCSrv)

    resp = cl.concat(b'foo', b'bar')
    assert resp == b'foobar'


def test_client_dir(conn_conf):
    cl = Client(IConcatRPC(), conn_conf)

    methods = dir(cl)

    assert 'concat' in methods
    assert 'shutdown' in methods

    assert 'not_exposed' not in methods


def test_method_returning_none(spawn):
    cl = spawn(IConcatRPC, ConcatRPCSrv)

    res = cl.none_return()


def test_error_doesnt_stop_server(spawn):
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


    cl = spawn(SomeRPC, SomeRPC)

    assert cl.ping() == b'pong'

    with pytest.raises(Exception):
        cl.raise_exc()

    assert cl.ping() == b'pong'

    with pytest.raises(Exception):
        cl.unknown_return_type()

    assert cl.ping() == b'pong'


def test_multithreaded(spawn):
    class SomeRPC(RPCInterface):
        @exposed
        def ping(self) -> bytes:
            sleep(0.1)
            return b'pong'

        @exposed
        def shutdown(self) -> None:
            raise Shutdown()


    cl = spawn(SomeRPC, SomeRPC)

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


def test_futures(spawn, log_debug):
    executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix='superexecutor')

    class SomeRPC(RPCInterface):
        def __init__(self):
            self.count = 2

        @exposed
        def compute(self) -> RPCFuture[bytes]:
            def _compute():
                logger.debug('Computing')
                sleep(0.5)
                logger.debug('Computed')
                res = b'4%d' % self.count
                self.count += 1
                return res

            return executor.submit(_compute)

        @exposed
        def shutdown(self):
            raise Shutdown()


    cl = spawn(SomeRPC, SomeRPC)

    try:
        f = cl.compute()
        f2 = cl.compute()
        assert f.result(timeout=5) == b'42'
        assert f2.result(timeout=5) == b'43'
    finally:
        executor.shutdown()


def test_futures_concat(spawn):
    cl = spawn(IConcatRPC, ConcatRPCSrv)

    f = cl.concat_async(b'4', b'2')
    f2 = cl.concat_async(b'hello, ', b'world')

    assert f.result(timeout=5) == b'42'
    assert f2.result(timeout=5) == b'hello, world'


def test_futures_with_exception(spawn):
    cl = spawn(IConcatRPC, ConcatRPCSrv)

    f = cl.concat_broken_async(b'4', b'2')

    with pytest.raises(CallException):
        assert f.result(timeout=5) == b'42'

    f = cl.raise_outside_future(b'4', b'2')
    with pytest.raises(CallException):
        assert f.result(timeout=5) == b'42'


def test_isfutureret():
    def foo() -> None:
        return

    isfutureret(foo)
