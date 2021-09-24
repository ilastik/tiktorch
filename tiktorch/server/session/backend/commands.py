from __future__ import annotations

import itertools
import logging
import queue
import threading
import typing
from dataclasses import dataclass, field

from tiktorch.server.session import types

if typing.TYPE_CHECKING:
    from tiktorch.server.session.backend.supervisor import Supervisor

    # from tiktorch.server.datasets import DynamicDataset


logger = logging.getLogger(__name__)

__all__ = [
    "ICommand",
    "AwaitableCommand",
    "PauseCmd",
    "ResumeCmd",
    "StopCmd",
    "UpdateDatasetCmd",
    "SetMaxNumIterations",
]


class Context:
    """
    Command execution context
    Contains modifiable entities as attributes
    """

    def __init__(self, *, supervisor: Supervisor) -> None:
        self.session = supervisor


class ICommand:
    __awaitable = None

    @property
    def awaitable(self):
        if not self.__awaitable:
            self.__awaitable = AwaitableCommand(self)

        return self.__awaitable

    def execute(self, ctx: Context) -> None:
        raise NotImplementedError()


class AwaitableCommand(ICommand):
    def __init__(self, cmd: ICommand):
        self._cmd = cmd
        self._done_evt = threading.Event()

    def wait(self):
        self._done_evt.wait()

    def execute(self, ctx: Context) -> None:
        try:
            self._cmd.execute(ctx)
        finally:
            self._done_evt.set()

    def __repr__(self):
        return f"Awaitable {self._cmd!r}"


class PauseCmd(ICommand):
    def execute(self, ctx: Context) -> None:
        ctx.session.transition_to(types.State.Paused)


class ResumeCmd(ICommand):
    def execute(self, ctx: Context) -> None:
        ctx.session.transition_to(types.State.Running)


class StopCmd(ICommand):
    def execute(self, ctx: Context) -> None:
        ctx.session.transition_to(types.State.Stopped)


class UpdateDatasetCmd(ICommand):
    def __init__(self, name, *, raw_data, labels):
        self._name = name
        self._raw_data = raw_data
        self._labels = labels

    def execute(self, ctx: Context) -> None:
        logger.warning("Not Implemented")
        # dataset = ctx.exemplum.get_dataset(self._name)
        # dataset.update(self._raw_data, self._labels)


class SetMaxNumIterations(ICommand):
    def __init__(self, num_iterations: int) -> None:
        self._num_iterations = num_iterations

    def execute(self, ctx: Context) -> None:
        ctx.session.set_max_num_iterations(self._num_iterations)


class ForwardPass(ICommand):
    def __init__(self, future, input_tensors):
        self._input_tensors = input_tensors
        self._future = future

    def execute(self, ctx: Context) -> None:
        try:
            self._future.set_result(ctx.session.forward(self._input_tensors))
        except Exception as e:
            self._future.set_exception(e)


class CommandPriorityQueue(queue.PriorityQueue):
    COMMAND_PRIORITIES = {StopCmd: 0}

    @dataclass(order=True)
    class _PrioritizedItem:
        priority: typing.Tuple[int, int]
        item: ICommand = field(compare=False)

    __counter = itertools.count()

    @classmethod
    def _make_queue_item(cls, cmd: ICommand):
        priority = cls.COMMAND_PRIORITIES.get(type(cmd), 999)
        return cls._PrioritizedItem((priority, next(cls.__counter)), cmd)

    def put(self, cmd: ICommand, block=True, timeout=None) -> None:
        return super().put(self._make_queue_item(cmd), block, timeout)

    def get(self, block=True, timeout=None):
        queue_item = super().get(block, timeout)
        return queue_item.item
