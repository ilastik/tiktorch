import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest
import torch

from tiktorch.server.session import State, commands
from tiktorch.server.session.supervisor import Supervisor


class TestTrainingWorkerSupervisor:
    class DummyCmd(commands.ICommand):
        def execute(self, ctx):
            pass

    class DummyTrainer:
        def __init__(self):
            self.iteration_count = 0
            self.max_num_iterations = 0
            self._break_cb = None
            self._devs = []

        def set_break_callback(self, cb):
            self._break_cb = cb

        def forward(self, input_tensor):
            return torch.Tensor([42])

        def move_to(self, devices):
            self._devs = devices

        def set_max_num_iterations(self, val):
            self.max_num_iterations = val

        def stop_fitting(self, max_num_iterations=None, max_num_epochs=None):
            return self._break_cb and self._break_cb() or self.iteration_count >= self.max_num_iterations

        def fit(self):
            while not self.stop_fitting():
                self.iteration_count += 1
                time.sleep(0.01)

    @pytest.fixture
    def trainer(self):
        return self.DummyTrainer()

    @pytest.fixture
    def worker(self, trainer):
        return Supervisor(trainer)

    @pytest.fixture
    def worker_thread(self, worker):
        t = threading.Thread(target=worker.run)
        t.start()
        yield t
        worker.send_command(commands.StopCmd())
        t.join()

    def test_not_running_worker_has_stopped_status(self, worker):
        assert State.Stopped == worker.state

    def test_started_worker_has_idle_status(self, worker, worker_thread):
        cmd = self.DummyCmd().awaitable
        worker.send_command(cmd)
        cmd.wait()

        assert State.Paused == worker.state

    def test_resuming_transitions_to_idle_with_no_devices(self, worker, worker_thread):
        cmd = commands.ResumeCmd().awaitable
        worker.send_command(cmd)
        cmd.wait()

        assert State.Idle == worker.state

    def test_transition_to_running(self, worker, worker_thread):
        cmd = commands.ResumeCmd()
        worker.send_command(cmd)

        add_work = commands.SetMaxNumberOfIterations(1000).awaitable
        worker.send_command(add_work)
        add_work.wait()

        assert State.Idle == worker.state

        add_device = commands.SetDevicesCmd([torch.device("cpu")]).awaitable
        worker.send_command(add_device)

        add_device.wait()

        assert State.Running == worker.state

        remove_device = commands.SetDevicesCmd([])
        awaitable_remove = remove_device.awaitable
        worker.send_command(awaitable_remove)
        awaitable_remove.wait()
        assert [torch.device("cpu")] == remove_device.result

        assert State.Idle == worker.state

    def test_exception_during_train_should_transition_to_paused(self, worker, worker_thread, trainer):
        fit_called = threading.Event()

        def _exc():
            fit_called.set()
            raise Exception()

        trainer.fit = _exc

        cmd = commands.ResumeCmd()
        worker.send_command(cmd)

        add_work = commands.SetMaxNumberOfIterations(1000).awaitable
        worker.send_command(add_work)
        add_work.wait()

        assert State.Idle == worker.state

        assert not fit_called.is_set()
        add_device = commands.SetDevicesCmd([torch.device("cpu")]).awaitable
        worker.send_command(add_device)
        add_device.wait()

        fit_called.wait()
        time.sleep(0.2)  # FIXME: Find a better way to wait for pause event with timeout
        assert State.Paused == worker.state

    def test_forward(self, worker, worker_thread, trainer):
        fut = Future()
        forward_cmd = commands.ForwardPass(fut, np.array([1]))
        worker.send_command(forward_cmd)
        assert 42 == fut.result(timeout=0.5)
