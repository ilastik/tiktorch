import pytest
import time
import multiprocessing as mp

from tiktorch.rpc.mp import MPClient, MPServer
from tiktorch.rpc import exposed, RPCInterface, Shutdown


class ITestApi(RPCInterface):
    @exposed
    def compute(self, a, b):
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
        return f'test {a + b}'

    def shutdown(self):
        print('shutdown')
        raise Shutdown()


def test_async():
    def _srv(conn):
        srv = MPServer(ApiImpl(), conn)
        srv.listen()

    parent_conn, child_conn = mp.Pipe()

    p = mp.Process(target=_srv, args=(child_conn,))
    p.start()

    client = MPClient(ITestApi(), parent_conn)
    f = client.compute(1, b=2)
    assert f.result() == 'test 3'

    with pytest.raises(NotImplementedError):
        f = client.broken(1, 2)
        f.result()

    client.shutdown()


def test_sync():
    def _srv(conn):
        srv = MPServer(ApiImpl(), conn)
        srv.listen()

    parent_conn, child_conn = mp.Pipe()

    p = mp.Process(target=_srv, args=(child_conn,))
    p.start()

    client = MPClient(ITestApi(), parent_conn)
    res = client.compute.sync(1, b=2)
    assert res == 'test 3'

    with pytest.raises(NotImplementedError):
        f = client.broken.sync(1, 2)

    client.shutdown()
