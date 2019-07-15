import argparse
import logging
import os

from .base import KILL_TIMEOUT, ServerProcess

logger = logging.getLogger(__name__)

# Output pid for process tracking
print(os.getpid(), flush=True)

parsey = argparse.ArgumentParser()
parsey.add_argument("--addr", type=str, default="127.0.0.1")
parsey.add_argument("--port", type=str, default="29500")
parsey.add_argument("--notify-port", type=str, default="29501")
parsey.add_argument("--debug", action="store_true")
parsey.add_argument("--dummy", action="store_true")
parsey.add_argument("--kill-timeout", type=int, default=KILL_TIMEOUT)

args = parsey.parse_args()
logger.info("Starting server on %s:%s", args.addr, args.port)

srv = ServerProcess(address=args.addr, port=args.port, notify_port=args.notify_port, kill_timeout=args.kill_timeout)

if args.dummy:
    from tiktorch.dev.dummy_server import DummyServerForFrontendDev

    srv.listen(provider_cls=DummyServerForFrontendDev)

else:
    srv.listen()
