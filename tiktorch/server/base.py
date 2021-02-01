import argparse
import logging
import logging.handlers
import os

from torch import multiprocessing as mp

mp.set_start_method("spawn", force=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Output pid for process tracking
    print(os.getpid(), flush=True)

    parsey = argparse.ArgumentParser()
    parsey.add_argument("--addr", type=str, default="127.0.0.1")
    parsey.add_argument("--port", type=str, default="5567")
    parsey.add_argument("--debug", action="store_true")
    parsey.add_argument("--dummy", action="store_true")
    parsey.add_argument("--connection-file", help="where to write connection parameters file")
    parsey.add_argument(
        "--kill-timeout",
        type=float,
        default=0.0,
        help="how long to wait for pings before sever will automatically shutdown",
    )

    args = parsey.parse_args()
    from . import grpc

    grpc.serve(args.addr, args.port, connection_file_path=args.connection_file, kill_timeout=args.kill_timeout)
