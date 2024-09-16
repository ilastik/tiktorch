import threading
import time

from tiktorch.server.grpc import flight_control_servicer


def test_shutdown_event_is_set_after_timeout():
    evt = threading.Event()

    _ = flight_control_servicer.FlightControlServicer(kill_timeout=0.01, done_evt=evt)
    assert "WatchdogThread" in [t.name for t in threading.enumerate()]
    assert not evt.is_set()
    assert evt.wait(timeout=0.1)


def test_shutdown_event_is_not_set_while_pings_keep_coming():
    evt = threading.Event()
    stop_pinger = threading.Event()

    servicer = flight_control_servicer.FlightControlServicer(kill_timeout=0.1, done_evt=evt)
    assert "WatchdogThread" in [t.name for t in threading.enumerate()]

    def _pinger():
        while not stop_pinger.is_set():
            servicer.Ping(None, None)
            time.sleep(0.01)

    pinger_thread = threading.Thread(target=_pinger, name="Pinger", daemon=True)
    pinger_thread.start()

    assert not evt.is_set()
    assert not evt.wait(timeout=0.2)

    stop_pinger.set()
    assert evt.wait(timeout=0.2)


def test_shutdown_timeout_0_means_no_watchdog():
    evt = threading.Event()
    _ = flight_control_servicer.FlightControlServicer(kill_timeout=0.0, done_evt=evt)
    assert "WatchdogThread" not in [t.name for t in threading.enumerate()]
    assert not evt.is_set()
    assert not evt.wait(timeout=0.1)
