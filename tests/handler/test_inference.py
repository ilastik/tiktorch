import torch

from torch import multiprocessing as mp

from tiktorch.handler.inference import IInference, InferenceProcess, run
from tiktorch.tiktypes import TikTensor, TikTensorBatch
from tiktorch.rpc.mp import create_client, MPClient, Shutdown

from tests.data.tiny_models import TinyConvNet2d, TinyConvNet3d


def test_initialization(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    ip = InferenceProcess(config=config, model=model)
    ip.shutdown()


def test_inference2d(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    inference = InferenceProcess(config=config, model=model)
    inference.set_devices([torch.device('cpu')])
    data = TikTensor(torch.zeros(in_channels, 15, 15), (0,))
    pred = inference.forward(data)
    assert isinstance(pred.result(timeout=10), TikTensor)
    inference.shutdown()


def test_inference2d_in_proc(tiny_model_2d, log_queue):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    handler_conn, inference_conn = mp.Pipe()
    p = mp.Process(
        target=run, kwargs={"conn": inference_conn, "model": model, "config": config, "log_queue": log_queue}
    )
    p.start()
    client = create_client(IInference, handler_conn)
    try:
        client.set_devices([torch.device('cpu')])
        data = TikTensor(torch.zeros(in_channels, 15, 15), (0,))
        f = client.forward(data)
        f.result(timeout=10)
    finally:
        client.shutdown()


def test_inference3d(tiny_model_3d, log_queue):
    config = tiny_model_3d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet3d(in_channels=in_channels)
    inference = InferenceProcess(config=config, model=model)
    inference.set_devices([torch.device('cpu')])
    data = TikTensor(torch.zeros(in_channels, 15, 15, 15), (0,))
    pred = inference.forward(data)
    try:
        assert isinstance(pred.result(timeout=10), TikTensor)
    finally:
        inference.shutdown()


def test_inference3d_in_proc(tiny_model_3d, log_queue):
    config = tiny_model_3d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet3d(in_channels=in_channels)
    handler_conn, inference_conn = mp.Pipe()
    p = mp.Process(
        target=run, kwargs={"conn": inference_conn, "model": model, "config": config, "log_queue": log_queue}
    )
    p.start()
    client = create_client(IInference, handler_conn)
    try:
        client.set_devices([torch.device('cpu')])
        f = []
        n = 10
        for i in range(n):
            data = TikTensor(torch.rand(in_channels, 15, 15, 15))
            f.append(client.forward(data))

        for i in range(n):
            f[i].result(timeout=10)
            print("received ", i)
    finally:
        client.shutdown()
