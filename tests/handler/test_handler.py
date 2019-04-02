import logging
import numpy
import pytest
import threading
import time

from torch import multiprocessing as mp

from tiktorch.rpc.mp import MPClient, Shutdown
from tiktorch.types import NDArray, NDArrayBatch
from tiktorch.tiktypes import TikTensor, TikTensorBatch

from tiktorch.handler.handler import HandlerProcess, IHandler, run as run_handler


def test_initialization_in_main_proc(tiny_model_2d, log_queue):
    hp = HandlerProcess(**tiny_model_2d, log_queue=log_queue)
    active_children = hp.active_children()
    try:
        hp.shutdown()
    except Shutdown:
        pass

    assert "Training" in active_children
    assert "Inference" in active_children
    assert "DryRun" in active_children


def test_initialization(tiny_model_2d, log_queue):
    client_conn, handler_conn = mp.Pipe()
    client = MPClient(IHandler(), client_conn)
    try:
        p = mp.Process(
            target=run_handler, name="Handler", kwargs={"conn": handler_conn, **tiny_model_2d, "log_queue": log_queue}
        )
        p.start()
        children = client.active_children()
        print("children", children, flush=True)
        print("children.res", children.result(timeout=5))
    finally:
        client.shutdown().result(timeout=20)
        print(threading.enumerate())


def test_forward_2d(tiny_model_2d, log_queue):
    client_conn, handler_conn = mp.Pipe()
    client = MPClient(IHandler(), client_conn)
    try:
        p = mp.Process(
            target=run_handler, name="Handler", kwargs={"conn": handler_conn, **tiny_model_2d, "log_queue": log_queue}
        )
        p.start()
        C, Y, X = tiny_model_2d["config"]["input_channels"], 15, 15
        futs = []
        for data in [
            TikTensor(numpy.random.random((C, Y, X)).astype(numpy.float32)),
            TikTensor(numpy.random.random((C, Y, X)).astype(numpy.float32)),
            TikTensor(numpy.random.random((C, Y, X)).astype(numpy.float32)),
            TikTensor(numpy.random.random((C, Y, X)).astype(numpy.float32)),
        ]:
            futs.append(client.forward(data))

        for i, fut in enumerate(futs):
            fut.result(timeout=10)
            print(f"got fut {i + 1}/{len(futs)}", flush=True)
    finally:
        client.shutdown().result(timeout=20)


def test_forward_3d(tiny_model_3d, log_queue):
    client_conn, handler_conn = mp.Pipe()
    client = MPClient(IHandler(), client_conn)
    try:
        p = mp.Process(
            target=run_handler, name="Handler", kwargs={"conn": handler_conn, **tiny_model_3d, "log_queue": log_queue}
        )
        p.start()
        C, Z, Y, X = tiny_model_3d["config"]["input_channels"], 15, 15, 15
        futs = []
        for data in [
            TikTensor(numpy.random.random((C, Z, Y, X)).astype(numpy.float32)),
            TikTensor(numpy.random.random((C, Z, Y, X)).astype(numpy.float32)),
            TikTensor(numpy.random.random((C, Z, Y, X)).astype(numpy.float32)),
            TikTensor(numpy.random.random((C, Z, Y, X)).astype(numpy.float32)),
        ]:
            futs.append(client.forward(data))

        for i, fut in enumerate(futs):
            fut.result(timeout=10)
            print(f"got fut {i + 1}/{len(futs)}", flush=True)
    finally:
        client.shutdown().result(timeout=20)
