import pytest
import time
import multiprocessing as mp
from concurrent.futures import Future
from collections import namedtuple

from tiktorch.rpc.mp import MPClient, MPServer, create_client
from tiktorch.rpc import exposed, RPCInterface, RPCFuture, Shutdown
from tiktorch import log


class ITestApi(RPCInterface):
    @exposed
    def fast_compute(self, a: int, b: int):
        raise NotImplementedError

    @exposed
    def compute(self, a: int, b: int):
        raise NotImplementedError

    @exposed
    def compute_fut(self, a: int, b: int) -> RPCFuture[str]:
        raise NotImplementedError

    @exposed
    def broken(self, a, b):
        raise NotImplementedError

    @exposed
    def shutdown(self):
        raise Shutdown()


class ApiImpl(ITestApi):
    def fast_compute(self, a, b):
        return f"test {a + b}"

    def compute(self, a, b):
        time.sleep(0.3)
        return f"test {a + b}"

    def compute_fut(self, a, b) -> Future:
        time.sleep(0.3)
        f = Future()
        f.set_result(f"test {a + b}")
        return f

    def shutdown(self):
        print("shutdown")
        raise Shutdown()


def _srv(conn, log_queue):
    log.configure(log_queue)
    srv = MPServer(ApiImpl(), conn)
    srv.listen()


@pytest.fixture
def client(log_queue):
    child, parent = mp.Pipe()

    p = mp.Process(target=_srv, args=(parent, log_queue))
    p.start()

    client = create_client(ITestApi, child)

    yield client

    client.shutdown()
    p.join()


def test_async(client: ITestApi):
    with pytest.raises(NotImplementedError):
        client.broken(1, 2)


def test_sync(client: ITestApi):
    res = client.compute(1, b=2)
    assert res == "test 3"

    with pytest.raises(NotImplementedError):
        f = client.broken(1, 2)


def test_future(client: ITestApi):
    res = client.compute_fut(1, b=2)
    assert res.result() == "test 3"



def test_race_condition(log_queue):
    class SlowConn:
        def __init__(self, conn):
            self._conn = conn

        def send(self, *args):
            self._conn.send(*args)
            # Block so future will be resolved earlier than we return value
            time.sleep(0.5)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    child, parent = mp.Pipe()

    p = mp.Process(target=_srv, args=(parent, log_queue))
    p.start()

    client = create_client(ITestApi, SlowConn(child))

    res = client.fast_compute(2, 2)
