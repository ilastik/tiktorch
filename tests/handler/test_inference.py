import torch

from torch import multiprocessing as mp

from tiktorch.handler.inference import IInference, InferenceProcess, run
from tiktorch.tiktypes import TikTensor
from tiktorch.rpc.mp import MPClient, Shutdown

from tests.data.tiny_models import TinyConvNet2d


def test_inference2d(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    inference = InferenceProcess(config=config, model=model)
    data = TikTensor(torch.zeros(in_channels, 15, 15), (0,))
    pred = inference.forward(data)
    assert isinstance(pred.result(), TikTensor)
    try:
        inference.shutdown()
    except Shutdown:
        pass


def test_inference2d_in_proc(tiny_model_2d, log_queue):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    handler_conn, inference_conn = mp.Pipe()
    p = mp.Process(target=run, kwargs={"conn": inference_conn, "model": model, "config": config, "log_queue": log_queue})
    p.start()
    client = MPClient(IInference(), handler_conn)
    data = TikTensor(torch.zeros(in_channels, 15, 15), (0,))
    f = client.forward(data)
    f.result()
    client.shutdown()


def test_inference3d_in_proc(tiny_model_3d, log_queue):
    config = tiny_model_3d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    handler_conn, inference_conn = mp.Pipe()
    p = mp.Process(target=run, kwargs={"conn": inference_conn, "model": model, "config": config, "log_queue": log_queue})
    p.start()
    client = MPClient(IInference(), handler_conn)
    data = TikTensor(torch.zeros(in_channels, 15, 15, 15), (0,))
    f = client.forward(data)
    f.result()
    client.shutdown()
