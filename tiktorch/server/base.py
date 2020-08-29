import argparse
import logging
import logging.handlers
import os

from torch import multiprocessing as mp

mp.set_start_method("spawn", force=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KILL_TIMEOUT = 60  # seconds


def main():
    # Output pid for process tracking
    print(os.getpid(), flush=True)

    parsey = argparse.ArgumentParser()
    parsey.add_argument("--addr", type=str, default="127.0.0.1")
    parsey.add_argument("--port", type=str, default="5567")
    parsey.add_argument("--debug", action="store_true")
    parsey.add_argument("--dummy", action="store_true")
    parsey.add_argument("--kill-timeout", type=int, default=KILL_TIMEOUT)

    args = parsey.parse_args()
    print(f"Starting server on {args.addr}:{args.port}")

    from . import grpc

    grpc.serve(args.addr, args.port)
