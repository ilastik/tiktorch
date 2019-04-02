import torch
import threading
import time

from torch import multiprocessing as mp

from tiktorch.handler.training import ITraining, TrainingProcess, run
from tiktorch.tiktypes import TikTensor, TikTensorBatch
from tiktorch.rpc.mp import MPClient, Shutdown

from tests.data.tiny_models import TinyConvNet2d


def test_initialization(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    tp = TrainingProcess(config=config, model=model)
    shutdown_raised = False
    try:
        tp.shutdown()
    except Shutdown:
        shutdown_raised = True

    assert shutdown_raised


def test_set_devices(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    tp = TrainingProcess(config=config, model=model)
    shutdown_raised = False
    try:
        tp.set_devices([torch.device("cpu")])
        tp.shutdown()
    except Shutdown:
        shutdown_raised = True

    assert shutdown_raised


def test_training(tiny_model_2d):
    config = tiny_model_2d["config"]
    config["max_num_iterations_per_update"] = 10
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    training = TrainingProcess(config=config, model=model)
    try:
        training.set_devices([torch.device("cpu")])
        data = TikTensorBatch(
            [
                TikTensor(torch.zeros(in_channels, 15, 15), (0,), torch.ones(in_channels, 9, 9)),
                TikTensor(torch.ones(in_channels, 15, 15), (0,), torch.zeros(in_channels, 9, 9)),
            ]
        )
        training.update_dataset("train", data)
        training.resume_training()
    finally:
        shutdown_raised = False
        try:
            training.shutdown()
        except Shutdown:
            shutdown_raised = True

        assert shutdown_raised


def test_training_in_proc(tiny_model_2d, log_queue):
    config = tiny_model_2d["config"]
    config["max_num_iterations_per_update"] = 10
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    handler_conn, training_conn = mp.Pipe()
    p = mp.Process(target=run, kwargs={"conn": training_conn, "model": model, "config": config, "log_queue": log_queue})
    p.start()
    client = MPClient(ITraining(), handler_conn)
    try:
        client.set_devices([torch.device("cpu")]).result(timeout=5)
        data = TikTensorBatch(
            [
                TikTensor(torch.zeros(in_channels, 15, 15), (0,), torch.ones(in_channels, 9, 9)),
                TikTensor(torch.ones(in_channels, 15, 15), (0,), torch.zeros(in_channels, 9, 9)),
            ]
        )
        client.update_dataset("train", data)
        client.resume_training()
        print("resumed training")
    finally:
        client.shutdown().result(timeout=30)


# def test_validation(tiny_model_2d):
