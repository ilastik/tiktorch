import os
import time
from unittest import mock

import numpy as np
import pytest

from tiktorch.launcher import LocalServerLauncher, ConnConf
from tiktorch.rpc import Client, Shutdown, TCPConnConf
from tiktorch.rpc_interface import IFlightControl
from tiktorch.server.base import TikTorchServer, Watchdog
from tiktorch.types import Model, ModelState, NDArray, NDArrayBatch


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
