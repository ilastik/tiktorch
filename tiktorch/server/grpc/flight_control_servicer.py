import logging
import threading
import time
from typing import Optional

from tiktorch.proto import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)


class FlightControlServicer(inference_pb2_grpc.FlightControlServicer):
    def __init__(self, *, kill_timeout, done_evt) -> None:
        self.__done_evt: threading.Event = done_evt
        self.__kill_timeout = kill_timeout
        self.__last_ping = time.time()

        self.__watchdog_thread: Optional[threading.Thread] = None
        if self.__kill_timeout:
            self.__watchdog_thread = self._start_watchdog_thread()

    def _start_watchdog_thread(self):
        """
        Starts daemon thread that sets shutdown event if no pings were received in
        defined interval
        """

        def _run_watchdog():
            logger.info("Starting watchdog thread")
            while not self.__done_evt.is_set():
                time.sleep(self.__kill_timeout)
                since_last_ping = time.time() - self.__last_ping
                logger.debug(
                    "Watchdog thread woke up. Last ping was %.2f seconds ago. Kill timeout is %.2f.",
                    since_last_ping,
                    self.__kill_timeout,
                )

                if since_last_ping >= self.__kill_timeout:
                    logger.info("Setting shutdown event because ping %.2f timeout was exceeded", self.__kill_timeout)
                    self.__done_evt.set()

        watchdog_thread = threading.Thread(target=_run_watchdog, name="WatchdogThread", daemon=True)
        watchdog_thread.start()
        return watchdog_thread

    def Ping(self, request: inference_pb2.Empty, context) -> inference_pb2.Empty:
        self.__last_ping = time.time()
        return inference_pb2.Empty()

    def Shutdown(self, request: inference_pb2.Empty, context) -> inference_pb2.Empty:
        if self.__done_evt:
            self.__done_evt.set()
        return inference_pb2.Empty()
