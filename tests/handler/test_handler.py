import pickle

from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl
from tiktorch.handler import HandlerProcess
from torch.multiprocessing import Pipe


class DummyServer(INeuralNetworkAPI, IFlightControl):
    def __init__(self, **kwargs):
        self.handler_conn, server_conn = Pipe()
        self.handler = HandlerProcess(server_conn=server_conn, **kwargs)


def test_minimal_initialization():
    kwargs = {"config": {}, "model_file": b"", "model_state": b"", "optimizer_state": b""}

    with open("../tiny_models.py", "r") as f:
        kwargs["model_file"] = pickle.dumps(f.read())

    kwargs["config"] = {"model_class_name": "TestModel0"}

    ts = DummyServer(**kwargs)
