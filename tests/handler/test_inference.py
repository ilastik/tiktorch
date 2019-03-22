import torch

from torch import multiprocessing as mp

from tiktorch.handler.inference import IInference, InferenceProcess, run
from tiktorch.tiktypes import TikTensor
from tiktorch.rpc.mp import MPServer, MPClient

from tests.data.tiny_models import TinyConvNet2d

def test_inference(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    inference = InferenceProcess(config=config, model=model)
    data = TikTensor(torch.zeros(in_channels, 15, 15), (0, ))
    pred = inference.forward(data)
    # TODO: Actual forward resul
    assert data == pred.result()


def test_inference_in_proc(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    handler_conn, inference_conn = mp.Pipe()
    p = mp.Process(target=run, kwargs={"conn": inference_conn, "model": model, "config": config})
    p.start()
    client =MPClient(IInference(), handler_conn)
    data = TikTensor(torch.zeros(in_channels, 15, 15), (0,))
    client.forward(data)

