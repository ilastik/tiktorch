import logging
import numpy
import pytest

from torch import multiprocessing as mp

from tiktorch.rpc.mp import MPClient, Shutdown
from tiktorch.types import NDArray, NDArrayBatch
from tiktorch.tiktypes import TikTensor, TikTensorBatch

from tiktorch.handler.handler import HandlerProcess, IHandler, run as run_handler

logger = logging.getLogger(__name__)


def test_initialization(tiny_model_2d):
    hp = HandlerProcess(**tiny_model_2d)
    active_children = hp.active_children()
    hp.shutdown()
    assert len(active_children) in (2, 3)


def test_forward(tiny_model_2d, log_queue):
    client_conn, handler_conn = mp.Pipe()
    client = MPClient(IHandler(), client_conn)
    try:
        p = mp.Process(target=run_handler, kwargs={"conn": handler_conn, **tiny_model_2d, "log_queue": log_queue})
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
        for fut in futs:
            fut.result(timeout=5)
    finally:
        client.shutdown()


@pytest.mark.skip
def test_forward2(tiny_model):
    ts = DummyServer(**tiny_model)
    C, X = 3, 5
    numpy.random.seed(0)
    keys = [(0,), (1,), (2,), (3,)]
    x = NDArrayBatch(
        [
            NDArray(numpy.random.random((X, C)).astype(numpy.float32), keys[0]),
            NDArray(numpy.random.random((X, C)).astype(numpy.float32), keys[1]),
            NDArray(numpy.random.random((X, C)).astype(numpy.float32), keys[2]),
            NDArray(numpy.random.random((X, C)).astype(numpy.float32), keys[3]),
        ]
    )
    try:
        # fetch inital idle reports for training and inference processes
        # idle0 = ts.listen(timeout=15)
        # assert idle0 is not None, "Waiting for idle 0 timed out"
        # assert idle0[0] == "report_idle", f"idle0: {idle0}"
        # idle0 = ts.listen(timeout=15)
        # assert idle0 is not None, "Waiting for idle 0 timed out"
        # assert idle0[0] == "report_idle", f"idle0: {idle0}"

        ts.forward(x)
        answer1 = ts.listen(timeout=15)
        assert answer1 is not None, "Answer 1 timed out"
        forward_answer1, answer1_dict = answer1
        assert forward_answer1 == "forward_answer", f"answer1: {answer1}"
        assert keys == answer1_dict["keys"]

        # idle1 = ts.listen(timeout=15)
        # assert idle1 is not None, "Waiting for idle 1 timed out"
        # assert idle1[0] == "report_idle", f"idle1: {idle1}"

        ts.forward(x)
        answer2 = ts.listen(timeout=15)
        assert answer2 is not None, "Answer 2 timed out"
        forward_answer2, answer2_dict = answer2
        assert forward_answer2 == "forward_answer"
        assert keys == answer2_dict["keys"]

        # idle2 = ts.listen(timeout=15)
        # assert idle2 is not None, "Waiting for idle 2 timed out"
        # assert idle2[0] == "report_idle", f"idle2: {idle2}"
    finally:
        ts.shutdown()

    assert answer1_dict["data"].equal(answer2_dict["data"]), "unequal data"
