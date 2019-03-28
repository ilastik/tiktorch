import logging
import numpy

from tiktorch.types import NDArray, NDArrayBatch
from tiktorch.handler.handler import HandlerProcess

logger = logging.getLogger(__name__)


def test_initialization(tiny_model_2d):
    hp = HandlerProcess(**tiny_model_2d)
    active_children = hp.active_children()
    hp.shutdown()
    assert len(active_children) in (2, 3)


def test_forward(tiny_model):
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
    ts.forward(x)
    answer = ts.listen()
    ts.shutdown()
    assert answer is not None, "Answer timed out"
    forward_answer, answer_dict = answer
    assert forward_answer == "forward_answer"
    assert keys == answer_dict["keys"]


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
