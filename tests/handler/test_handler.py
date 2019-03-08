import numpy
import pickle
import pytest
import torch

from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl
from tiktorch.handler import HandlerProcess
from tiktorch.types import NDArray, NDArrayBatch
from torch.multiprocessing import Pipe, set_start_method
from typing import Union, Tuple


class DummyServer(INeuralNetworkAPI, IFlightControl):
    def __init__(self, **kwargs):
        self.handler_conn, server_conn = Pipe()
        self.handler = HandlerProcess(server_conn=server_conn, **kwargs)
        self.handler.start()

    def forward(self, batch: NDArrayBatch) -> None:
        self.handler_conn.send(
            (
                "forward",
                {"keys": [a.id for a in batch], "data": torch.stack([torch.from_numpy(a.as_numpy()) for a in batch])},
            )
        )

    def listen(self) -> Union[None, Tuple[str, dict]]:
        if self.handler_conn.poll(timeout=3):
            print('got answer!')
            return self.handler_conn.recv()
        else:
            return None


def test_minimal_initialization():
    kwargs = {"config": {}, "model_file": b"", "model_state": b"", "optimizer_state": b""}
    with open("../tiny_models.py", "r") as f:
        kwargs["model_file"] = pickle.dumps(f.read())

    kwargs["config"] = {"model_class_name": "TestModel0", "optimizer_config": {"method": "Adam"}}

    ts = DummyServer(**kwargs)

def test_forward():
    kwargs = {"config": {}, "model_file": b"", "model_state": b"", "optimizer_state": b""}

    with open("../tiny_models.py", "r") as f:
        kwargs["model_file"] = pickle.dumps(f.read())

    kwargs["config"] = {"model_class_name": "TestModel0", "optimizer_config": {"method": "Adam"}}

    ts = DummyServer(**kwargs)
    C, Z, Y, X = 3, 1, 5, 6
    numpy.random.seed(0)
    keys = [(0, ), (1, ), (2, ), (3, )]
    x = NDArrayBatch(
        [
            NDArray(numpy.random.random((C, Z, Y, X)), keys[0]),
            NDArray(numpy.random.random((C, Z, Y, X)), keys[1]),
            NDArray(numpy.random.random((C, Z, Y, X)), keys[2]),
            NDArray(numpy.random.random((C, Z, Y, X)), keys[3]),
        ]
    )
    ts.forward(x)
    answer = ts.listen()
    assert answer is not None, 'Answer timed out'
    forward_answer, answer_dict = answer
    assert forward_answer == 'forward_answer'
    assert keys == answer_dict['keys']
    print(answer_dict['data'])