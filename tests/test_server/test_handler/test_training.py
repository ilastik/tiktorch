import torch
import pytest
import queue
import threading
from torch import multiprocessing as mp

from tests.data.tiny_models import TinyConvNet2d
from tiktorch.rpc.mp import MPClient, Shutdown, create_client
from tiktorch.server.handler.training import ITraining, TrainingProcess, run
from tiktorch.server.handler import training
from tiktorch.tiktypes import TikTensor, TikTensorBatch


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


class TestTrainingWorker:
    class DummyCmd(training.ICommand):
        def execute(self):
            pass

    @pytest.fixture
    def worker(self):
        return training.TrainingWorker(None)

    @pytest.fixture
    def worker_thread(self, worker):
        t = threading.Thread(target=worker.run)
        t.start()
        yield t
        worker.send_command(training.StopCmd(worker))
        t.join()

    def test_not_running_worker_has_stopped_status(self, worker):
        assert training.State.Stopped == worker.status

    def test_started_worker_has_idle_status(self, worker, worker_thread):
        cmd = self.DummyCmd().awaitable
        worker.send_command(cmd)
        cmd.wait()

        assert training.State.Idle == worker.state

    def test_resuming_transitions_to_waiting_device_with_no_devices(self, worker, worker_thread):
        cmd = training.ResumeCmd(worker).awaitable
        worker.send_command(cmd)
        cmd.wait()

        assert training.State.WaitingDevice == worker.state

    def test_adding_removing_devices_on_running_worker_transitions(self, worker, worker_thread):
        cmd = training.ResumeCmd(worker)
        worker.send_command(cmd)

        add_device = training.AddDeviceCmd(worker, "cpu").awaitable
        worker.send_command(add_device)

        add_device.wait()

        assert training.State.Running == worker.state

        remove_device = training.RemoveDeviceCmd(worker, "cpu").awaitable
        worker.send_command(remove_device)
        remove_device.wait()

        assert training.State.WaitingDevice == worker.state
