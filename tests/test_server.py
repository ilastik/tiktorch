import os
import time

from unittest import mock

import pytest
import numpy as np

from tiktorch.launcher import LocalServerLauncher
from tiktorch.rpc import Shutdown, TCPConnConf, Shutdown, Client
from tiktorch.rpc_interface import IFlightControl
from tiktorch.server import TikTorchServer, Watchdog
from tiktorch.types import NDArray, NDArrayBatch, Model, ModelState


@pytest.fixture
def srv():
    tik_srv = TikTorchServer()
    yield tik_srv
    shutdown_raised = False
    try:
        tik_srv.shutdown()
    except Shutdown:
        shutdown_raised = True

    assert shutdown_raised


def test_tiktorch_server_ping(srv):
    assert srv.ping() == b"pong"


def test_load_model(srv, nn_sample):
    srv.load_model(nn_sample.model, nn_sample.state, [])
    assert "Handler" in srv.active_children()


def test_forward(datadir, srv, nn_sample):
    input_arr = np.load(os.path.join(datadir, "fwd_input.npy"))
    out_arr = np.load(os.path.join(datadir, "fwd_out.npy"))
    out_arr.shape = (1, 320, 320)

    srv.load_model(nn_sample.model, nn_sample.state, [])

    res = srv.forward(NDArray(input_arr)).result(timeout=30)
    res_numpy = res.as_numpy()
    np.testing.assert_array_almost_equal(res_numpy[0], out_arr)


class TestWatchdog:
    class SrvStub:
        def __init__(self):
            self._last_ping = None

        def last_ping(self):
            return self._last_ping

        def set_lastping_time(self, ts: float):
            self._last_ping = ts

        def shutdown(self):
            pass

    @pytest.fixture
    def srv_stub(self):
        return mock.Mock(wraps=self.SrvStub())

    @pytest.fixture
    def watchdog(self, srv_stub):
        s = Watchdog(srv_stub, kill_timeout=0.1)
        yield s
        s.stop()

    def test_watchdog_queries_for_last_ping(self, srv_stub, watchdog):
        watchdog.start()
        time.sleep(0.2)
        srv_stub.last_ping.assert_called()

    def test_watchdog_shutdowns_srv_if_ping_timestamp_is_stale(self, srv_stub, watchdog):
        watchdog.start()
        srv_stub.set_lastping_time(time.time() - 100)
        time.sleep(0.2)
        srv_stub.shutdown.assert_called()

    def test_watchdog_does_nothing_if_lastping_is_unknown(self, srv_stub, watchdog):
        watchdog.start()
        srv_stub.set_lastping_time(None)
        time.sleep(0.2)
        srv_stub.shutdown.assert_not_called()

    def test_shutdown(self, srv_port, pub_port):
        conn_conf = TCPConnConf("127.0.0.1", srv_port, pub_port, timeout=2)
        launcher = LocalServerLauncher(conn_conf)
        launcher.start(kill_timeout=1, ping_interval=9999)

        assert launcher.is_server_running()

        time.sleep(3)

        assert not launcher.is_server_running()
