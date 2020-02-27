import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest
import torch

from tiktorch.server.session import State
from tiktorch.server.session.backend import commands
from tiktorch.server.session.backend.supervisor import Supervisor


class TestExemplumSupervisor:
    class DummyCmd(commands.ICommand):
        def execute(self, ctx):
            pass

    class DummyExemplum:
        def __init__(self):
            self.iteration_count = 0
            self.max_num_iterations = 0
            self._break_cb = None
            self._devs = []

        def set_break_callback(self, cb):
            self._break_cb = cb

        def forward(self, input_tensor):
            return torch.Tensor([42])

        def set_max_num_iterations(self, val):
            self.max_num_iterations = val

        def stop_training(self, max_num_iterations=None, max_num_epochs=None):
            return self._break_cb and self._break_cb() or self.iteration_count >= self.max_num_iterations

        def train(self):
            while not self.stop_training():
                self.iteration_count += 1
                time.sleep(0.01)

    @pytest.fixture
    def exemplum(self):
        return self.DummyExemplum()

    @pytest.fixture
    def supervisor(self, exemplum):
        return Supervisor(exemplum)

    @pytest.fixture
    def worker_thread(self, supervisor):
        t = threading.Thread(target=supervisor.run)
        t.start()
        yield t
        supervisor.send_command(commands.StopCmd())
        t.join()

    def test_not_running_worker_has_stopped_status(self, supervisor):
        assert State.Stopped == supervisor.state

    def test_started_worker_has_idle_status(self, supervisor, worker_thread):
        cmd = self.DummyCmd().awaitable
        supervisor.send_command(cmd)
        cmd.wait()

        assert State.Paused == supervisor.state

    def test_resuming_transitions_to_idle_with_no_devices(self, supervisor, worker_thread):
        cmd = commands.ResumeCmd().awaitable
        supervisor.send_command(cmd)
        cmd.wait()

        assert State.Idle == supervisor.state

    def test_transition_to_running(self, supervisor, worker_thread):
        cmd = commands.ResumeCmd()
        supervisor.send_command(cmd)

        add_work = commands.SetMaxNumIterations(2).awaitable
        supervisor.send_command(add_work)
        add_work.wait()

        assert supervisor.state == State.Running

    def test_exception_during_train_should_transition_to_paused(self, supervisor, worker_thread, exemplum):
        train_called = threading.Event()

        def _exc():
            train_called.set()
            raise Exception()

        exemplum.train = _exc

        cmd = commands.ResumeCmd()
        supervisor.send_command(cmd)

        add_work = commands.SetMaxNumIterations(2).awaitable
        supervisor.send_command(add_work)
        add_work.wait()

        train_called.wait()
        assert supervisor.state == State.Running
        time.sleep(0.2)  # FIXME: Find a better way to wait for pause event with timeout
        assert supervisor.state == State.Paused

    def test_finished_training_should_transition_to_paused(self, supervisor, worker_thread, exemplum):
        cmd = commands.ResumeCmd()
        supervisor.send_command(cmd)

        add_work = commands.SetMaxNumIterations(2).awaitable
        supervisor.send_command(add_work)
        add_work.wait()
        assert supervisor.state == State.Running
        time.sleep(0.1)  # FIXME: Find a better way to wait for pause event with timeout
        assert supervisor.state == State.Idle

    def test_forward(self, supervisor, worker_thread, exemplum):
        fut = Future()
        forward_cmd = commands.ForwardPass(fut, np.array([1]))
        supervisor.send_command(forward_cmd)
        assert 42 == fut.result(timeout=0.5)
