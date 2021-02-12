import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import CancelledError, Future, TimeoutError

import pytest

from tiktorch import log
from tiktorch.rpc import RPCFuture, RPCInterface, Shutdown, exposed
from tiktorch.rpc.mp import FutureStore, MPServer, create_client


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
    def shutdown(self) -> RPCFuture[Shutdown]:
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

    def shutdown(self) -> Future:
        return Shutdown()


def _srv(conn, log_queue):
    log.configure(log_queue)
    srv = MPServer(ApiImpl(), conn)
    srv.listen()


@pytest.fixture
def client(log_queue):
    child, parent = mp.Pipe()

    p = mp.Process(target=_srv, args=(parent, log_queue))
    p.start()

    client = create_client(ITestApi, child, timeout=10)

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
        client.broken(1, 2)


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

    client.fast_compute(2, 2)

    client.shutdown().result()


def test_future_timeout(client: ITestApi, log_queue):
    child, parent = mp.Pipe()

    p = mp.Process(target=_srv, args=(parent, log_queue))
    p.start()

    client = create_client(ITestApi, child, timeout=0.001)

    with pytest.raises(TimeoutError):
        client.compute(1, 2)

    with pytest.raises(TimeoutError):
        client.compute.async_(1, 2).result()

    with pytest.raises(TimeoutError):
        client.compute_fut(1, 2).result()

    client.compute.async_(1, 2).result(timeout=3)

    client.shutdown()
    p.join()


class ICancelable(RPCInterface):
    @exposed
    def compute(self) -> RPCFuture:
        raise NotImplementedError

    @exposed
    def process_queued(self) -> None:
        raise NotImplementedError

    @exposed
    def cancelled_count(self) -> int:
        return self._executor.canceled_count

    @exposed
    def shutdown(self) -> Shutdown:
        return Shutdown()


cancelled = object()


class CancelledExecutor:
    def __init__(self):
        self._q = queue.Queue()
        self._running = threading.Event()

        self._worker = threading.Thread(target=self._worker)
        self._worker.daemon = True
        self._worker.start()
        self.cancelled_count = 0

    def submit(self, func) -> RPCFuture:
        f = RPCFuture()
        self._q.put((func, f))
        return f

    def stop(self):
        self._q.put((None, None))
        self.process_queued()

    def process_queued(self):
        self._running.set()

    def _worker(self):
        while True:
            self._running.wait()

            task, fut = self._q.get()

            if task is None:
                break

            try:
                if fut.cancelled():
                    fut.set_result(cancelled)
                    self.cancelled_count += 1
                else:
                    res = task()
                    fut.set_result(res)
            except Exception as e:
                fut.set_exception(e)


class CancelableSrv(ICancelable):
    def __init__(self):
        self._executor = CancelledExecutor()

    def compute(self) -> RPCFuture:
        return self._executor.submit(lambda: 42)

    def process_queued(self) -> None:
        self._executor.process_queued()

    def cancelled_count(self) -> int:
        return self._executor.cancelled_count

    def shutdown(self) -> Shutdown:
        self._executor.stop()
        return Shutdown()


def test_canceled_executor():
    executor = CancelledExecutor()

    assert executor.cancelled_count == 0
    f = executor.submit(lambda: 42)

    f2 = executor.submit(lambda: 42)
    f2.cancel()

    executor.process_queued()

    assert f.result(timeout=1) == 42
    assert f2.result(timeout=1) is cancelled
    assert executor.cancelled_count == 1

    executor.stop()


def _cancel_srv(conn, log_queue):
    log.configure(log_queue)
    srv = MPServer(CancelableSrv(), conn)
    srv.listen()


def _run_srv(srv_cls, conn, log_queue):
    log.configure(log_queue)
    srv = MPServer(srv_cls(), conn)
    srv.listen()


@pytest.fixture
def spawn(log_queue):
    data = {}

    def _spawn(iface_cls, srv_cls):
        child, parent = mp.Pipe()

        p = mp.Process(target=_run_srv, args=(srv_cls, parent, log_queue))
        p.start()

        data["client"] = client = create_client(iface_cls, child)
        data["process"] = p
        return client

    yield _spawn

    data["client"].shutdown()
    data["process"].join()


def test_future_cancelation(spawn):
    client: ICancelable = spawn(ICancelable, CancelableSrv)

    f = client.compute()
    f.cancel()

    f2 = client.compute()

    client.process_queued()
    assert f2.result(timeout=1) == 42

    with pytest.raises(CancelledError):
        assert f.result(timeout=1)

    assert client.cancelled_count() == 1


@pytest.fixture
def fut_store():
    return FutureStore()


def test_future_store(fut_store):
    id1, f1 = "testid1", Future()
    id2, f2 = "testid2", Future()

    fut_store.put(id1, f1)
    fut_store.put(id2, f2)

    popped_id1 = fut_store.pop_future(f1)

    assert popped_id1 == id1
    assert fut_store.pop_id(id1) is None

    popped_fut2 = fut_store.pop_id(id2)

    assert popped_fut2 == f2
    assert fut_store.pop_future(f2) is None
