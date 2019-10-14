from __future__ import annotations

import typing
import threading

from torch.utils.data import DataLoader

from tiktorch.server.datasets import DynamicDataLoaderWrapper

from .types import State

if typing.TYPE_CHECKING:
    from .worker import TrainingWorker


__all__ = [
    "ICommand",
    "AwaitableCommand",
    "PauseCmd",
    "ResumeCmd",
    "StopCmd",
    "SetDevicesCmd",
    "UpdateDatasetCmd",
    "SetMaxNumberOfIterations",
]


class ICommand:
    __awaitable = None

    @property
    def awaitable(self):
        if not self.__awaitable:
            self.__awaitable = AwaitableCommand(self)

        return self.__awaitable

    def execute(self) -> None:
        raise NotImplementedError()


class AwaitableCommand(ICommand):
    def __init__(self, cmd: ICommand):
        self._cmd = cmd
        self._done_evt = threading.Event()

    def wait(self):
        self._done_evt.wait()

    def execute(self):
        try:
            self._cmd.execute()
        finally:
            self._done_evt.set()

    def __repr__(self):
        return f"Awaitable {self._cmd!r}"


class _WorkerCmd(ICommand):
    def __init__(self, worker: TrainingWorker):
        self._worker = worker


class PauseCmd(_WorkerCmd):
    def execute(self):
        self._worker.transition_to(State.Paused)


class ResumeCmd(_WorkerCmd):
    def execute(self):
        self._worker.transition_to(State.Running)


class StopCmd(_WorkerCmd):
    def execute(self):
        self._worker.transition_to(State.Stopped)


class SetDevicesCmd(ICommand):
    def __init__(self, worker, devices):
        self._worker = worker
        self._devices = devices

        self.result = None

    def execute(self):
        self.result = self._worker.set_devices(self._devices)


class UpdateDatasetCmd(ICommand):
    def __init__(self, trainer, dataset, loader_kwargs, *, raw_data, labels):
        self._trainer = trainer
        self._dataset = dataset
        self._raw_data = raw_data
        self._labels = labels
        self._loader_kwargs = loader_kwargs  # FIXME

    def execute(self):
        self._dataset.update(self._raw_data, self._labels)
        loader = DataLoader(**self._loader_kwargs)
        self._trainer.bind_loader("train", DynamicDataLoaderWrapper(loader))


class SetMaxNumberOfIterations(ICommand):
    def __init__(self, worker: TrainingWorker, num_iterations: int) -> None:
        self._worker = worker
        self._num_iterations = num_iterations

    def execute(self) -> None:
        self._worker.set_max_num_iterations(self._num_iterations)
