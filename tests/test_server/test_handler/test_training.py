import torch
from torch import multiprocessing as mp

from tiktorch.server.handler.training import ITraining, TrainingProcess, run
from tiktorch.tiktypes import TikTensor, TikTensorBatch
from tiktorch.rpc.mp import MPClient, create_client, Shutdown

from tests.data.tiny_models import TinyConvNet2d


def test_initialization(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    tp = TrainingProcess(config=config, model=model)
    tp.shutdown()


def test_set_devices(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    tp = TrainingProcess(config=config, model=model)
    tp.set_devices([torch.device("cpu")])
    tp.shutdown()


def test_training(tiny_model_2d):
    config = tiny_model_2d["config"]
    config["num_iterations_per_update"] = 10
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    training = TrainingProcess(config=config, model=model)
    try:
        training.set_devices([torch.device("cpu")])
        data = TikTensorBatch(
            [
                TikTensor(torch.zeros(in_channels, 15, 15), ((1,), (1,))),
                TikTensor(torch.ones(in_channels, 9, 9), ((2,), (2,))),
            ]
        )
        labels = TikTensorBatch(
            [
                TikTensor(torch.ones(in_channels, 15, 15, dtype=torch.uint8), ((1,), (1,))),
                TikTensor(torch.full((in_channels, 9, 9), 2, dtype=torch.uint8), ((2,), (2,))),
            ]
        )
        training.update_dataset("training", data, labels)
        training.resume_training()
        import time

        time.sleep(10)
    finally:
        training.shutdown()


def test_training_in_proc(tiny_model_2d, log_queue):
    config = tiny_model_2d["config"]
    config["num_iterations_per_update"] = 10
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    handler_conn, training_conn = mp.Pipe()
    p = mp.Process(target=run, kwargs={"conn": training_conn, "model": model, "config": config, "log_queue": log_queue})
    p.start()
    client = create_client(ITraining, handler_conn)
    try:
        client.set_devices([torch.device("cpu")])
        data = TikTensorBatch(
            [
                TikTensor(torch.zeros(in_channels, 15, 15), ((1,), (1,))),
                TikTensor(torch.ones(in_channels, 9, 9), ((2,), (2,))),
            ]
        )
        labels = TikTensorBatch(
            [
                TikTensor(torch.ones(in_channels, 15, 15, dtype=torch.uint8), ((1,), (1,))),
                TikTensor(torch.full((in_channels, 9, 9), 2, dtype=torch.uint8), ((2,), (2,))),
            ]
        )
        client.update_dataset("training", data, labels)
        client.resume_training()
    finally:
        client.shutdown()


# def test_validation(tiny_model_2d):
