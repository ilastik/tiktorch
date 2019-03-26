import pytest
import time
import multiprocessing as mp
from concurrent.futures import Future

from tiktorch.rpc.mp import MPClient, MPServer
from tiktorch.rpc import exposed, RPCInterface, Shutdown


class ITestApi(RPCInterface):
    @exposed
    def compute(self, a, b):
        raise NotImplementedError

    @exposed
    def compute_fut(self, a, b):
        raise NotImplementedError

    @exposed
    def broken(self, a, b):
        raise NotImplementedError

    @exposed
    def shutdown(self):
        raise Shutdown()


class ApiImpl(ITestApi):
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


def _srv(conn):
    srv = MPServer(ApiImpl(), conn)
    srv.listen()


def test_async():
    parent_conn, child_conn = mp.Pipe()

    p = mp.Process(target=_srv, args=(child_conn,))
    p.start()

    client = MPClient(ITestApi(), parent_conn)
    f = client.compute(1, b=2)
    assert f.result() == "test 3"

    with pytest.raises(NotImplementedError):
        f = client.broken(1, 2)
        f.result()

    client.shutdown()


def test_sync():
    parent_conn, child_conn = mp.Pipe()

    p = mp.Process(target=_srv, args=(child_conn,))
    p.start()

    client = MPClient(ITestApi(), parent_conn)
    res = client.compute.sync(1, b=2)
    assert res == "test 3"

    with pytest.raises(NotImplementedError):
        f = client.broken.sync(1, 2)

    client.shutdown()


def test_future():
    parent_conn, child_conn = mp.Pipe()

    p = mp.Process(target=_srv, args=(child_conn,))
    p.start()

    client = MPClient(ITestApi(), parent_conn)
    res = client.compute_fut(1, b=2)
    assert res.result(timeout=5) == "test 3"
    client.shutdown()

def end_generator(start, end, batch_size):
    for idx in range(start + batch_size, end, batch_size):
        yield idx

    yield end


def test_end_gen():
    result = list(end_generator(0, 10, 7))
    assert result == [7, 10]

    result = list(end_generator(0, 4, 7))
    assert result == [4]

    result = list(end_generator(0, 4, 3))
    assert result == [4]

    result = list(end_generator(0, 4, 3))
    assert result == [3, 4]
