import logging
import numpy
import pickle
import torch

from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl
from tiktorch.handler import HandlerProcess
from tiktorch.handler.constants import SHUTDOWN, SHUTDOWN_ANSWER
from tiktorch.types import NDArray, NDArrayBatch
from torch.multiprocessing import Pipe, set_start_method
from typing import Union, Tuple


logger = logging.getLogger(__name__)


class DummyServer(INeuralNetworkAPI, IFlightControl):
    def __init__(self, **kwargs):
        self.handler_conn, server_conn = Pipe()
        self.handler = HandlerProcess(server_conn=server_conn, **kwargs)
        self.handler.start()

    @property
    def devices(self):
        return self.handler.devices

    @devices.setter
    def devices(self, devices: list):
        self.handler.devices = devices

    def dry_run_on_device(self, device, upper_bound):
        return self.handler.dry_run_on_device(device, upper_bound)

    def forward(self, batch: NDArrayBatch) -> None:
        self.handler_conn.send(
            (
                "forward",
                {"keys": [a.id for a in batch], "data": torch.stack([torch.from_numpy(a.as_numpy()) for a in batch])},
            )
        )

    def active_children(self):
        self.handler_conn.send(("active_children", {}))

    def listen(self) -> Union[None, Tuple[str, dict]]:
        if self.handler_conn.poll(timeout=10):
            answer = self.handler_conn.recv()
            logger.debug("got answer: %s", answer)
            return answer
        else:
            return None

    def shutdown(self):
        self.handler_conn.send(SHUTDOWN)
        got_shutdown_answer = False
        while self.handler.is_alive():
            if self.handler_conn.poll(timeout=2):
                answer = self.handler_conn.recv()
                if answer == SHUTDOWN_ANSWER:
                    got_shutdown_answer = True

        assert got_shutdown_answer


def test_initialization():
    kwargs = {"model_file": b"", "model_state": b"", "optimizer_state": b""}
    with open("../tiny_models.py", "r") as f:
        kwargs["model_file"] = pickle.dumps(f.read())

    kwargs["config"] = {"model_class_name": "TestModel0", "optimizer_config": {"method": "Adam"}}

    ts = DummyServer(**kwargs)
    ts.active_children()
    active_children = ts.listen()
    ts.shutdown()
    assert active_children is not None
    assert len(active_children) == 2
    assert "TrainingProcess" in active_children
    assert "InferenceProcess" in active_children


def test_forward():
    kwargs = {"model_file": b"", "model_state": b"", "optimizer_state": b""}

    with open("../tiny_models.py", "r") as f:
        kwargs["model_file"] = pickle.dumps(f.read())

    kwargs["config"] = {"model_class_name": "TestModel0", "optimizer_config": {"method": "Adam"}}

    ts = DummyServer(**kwargs)
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

def test_devices():
    kwargs = {"model_file": b"", "model_state": b"", "optimizer_state": b""}

    with open("tiny_models.py", "r") as f:
        kwargs["model_file"] = pickle.dumps(f.read())

    kwargs["config"] = {"model_class_name": "TinyConvNet2d", "optimizer_config": {"method": "Adam"}}

    ts = DummyServer(**kwargs)
    ts.devices = ['cpu']

def test_dry_run():
    kwargs = {"model_file": b"", "model_state": b"", "optimizer_state": b""}

    with open("tiny_models.py", "r") as f:
        kwargs["model_file"] = pickle.dumps(f.read())

    kwargs["config"] = {"model_class_name": "TinyConvNet3d", "optimizer_config": {"method": "Adam"}}

    ts = DummyServer(**kwargs)
    ts.dry_run_on_device(torch.device('cpu'), (1, 1, 125, 1250, 1250))
