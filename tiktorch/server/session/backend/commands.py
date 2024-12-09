from __future__ import annotations

import itertools
import logging
import queue
import threading
import typing
from dataclasses import dataclass, field
from typing import Generic, Type, TypeVar

from tiktorch.trainer import TrainerState

if typing.TYPE_CHECKING:
    from tiktorch.server.session.backend.supervisor import BioModelSupervisor, Supervisors, TrainerSupervisor

    # from tiktorch.server.datasets import DynamicDataset


logger = logging.getLogger(__name__)

__all__ = [
    "ICommand",
    "AwaitableCommand",
    "StartTrainingCmd",
    "PauseTrainingCmd",
    "ResumeTrainingCmd",
    "ShutdownWithTeardownCmd",
    "SetResumeStateTrainingCmd",
    "SetPauseStateTrainingCmd",
    "SetStartStateTrainingCmd",
    "UpdateDatasetCmd",
    "SetMaxNumIterations",
]

SupervisorType = TypeVar("SupervisorType")


class Context(Generic[SupervisorType]):
    """
    Command execution context
    Contains modifiable entities as attributes
    """

    def __init__(self, *, supervisor: SupervisorType) -> None:
        self.session = supervisor


class ICommand:
    __awaitable = None

    def __init__(self, is_termination_signal: bool = False):
        self._is_termination_signal = is_termination_signal

    def is_stop(self):
        return self._is_termination_signal

    @property
    def awaitable(self):
        if not self.__awaitable:
            self.__awaitable = AwaitableCommand(self)

        return self.__awaitable

    def execute(self, ctx: Context) -> None:
        raise NotImplementedError()

    def is_command(self, command_to_check: Type[ICommand]):
        """
        Identify the command even if it is wrapped as an awaitable one
        """
        if isinstance(self, AwaitableCommand):
            return isinstance(self._cmd, command_to_check)
        else:
            return isinstance(self, command_to_check)


class AwaitableCommand(ICommand):
    def __init__(self, cmd: ICommand):
        self._cmd = cmd
        self._done_evt = threading.Event()
        self._exception: Exception | None = None  # Store the exception
        super().__init__(is_termination_signal=self._cmd.is_stop())

    def wait(self):
        self._done_evt.wait()
        if self._exception is not None:
            raise self._exception

    def execute(self, ctx: Context) -> None:
        try:
            self._cmd.execute(ctx)
        except Exception as e:
            self._exception = e
        finally:
            self._done_evt.set()

    def __repr__(self):
        return f"Awaitable {self._cmd!r}"


class PauseTrainingCmd(ICommand):
    def execute(self, ctx: Context[TrainerSupervisor]) -> None:
        ctx.session.pause()


class ResumeTrainingCmd(ICommand):
    def execute(self, ctx: Context[TrainerSupervisor]) -> None:
        ctx.session.resume()


class SetStartStateTrainingCmd(ICommand):
    def execute(self, ctx: Context[TrainerSupervisor]) -> None:
        ctx.session.transition_to_state(new_state=TrainerState.RUNNING, valid_states={TrainerState.IDLE})


class SetPauseStateTrainingCmd(ICommand):
    def execute(self, ctx: Context[TrainerSupervisor]) -> None:
        ctx.session.transition_to_state(new_state=TrainerState.PAUSED, valid_states={TrainerState.RUNNING})


class SetResumeStateTrainingCmd(ICommand):
    def execute(self, ctx: Context[TrainerSupervisor]) -> None:
        ctx.session.transition_to_state(new_state=TrainerState.RUNNING, valid_states={TrainerState.PAUSED})


class ShutdownCmd(ICommand):
    def __init__(self):
        super().__init__(is_termination_signal=True)

    def execute(self, ctx: Context) -> None:
        pass


class ShutdownWithTeardownCmd(ShutdownCmd):
    def execute(self, ctx: Context[Supervisors]) -> None:
        ctx.session.shutdown()


class StartTrainingCmd(ICommand):
    def execute(self, ctx: Context[TrainerSupervisor]) -> None:
        ctx.session.start()


class UpdateDatasetCmd(ICommand):
    def __init__(self, name, *, raw_data, labels):
        super().__init__()
        self._name = name
        self._raw_data = raw_data
        self._labels = labels

    def execute(self, ctx: Context[BioModelSupervisor]) -> None:
        logger.warning("Not Implemented")
        ctx.session.update_dataset()
        # dataset = ctx.exemplum.get_dataset(self._name)
        # dataset.update(self._raw_data, self._labels)


class SetMaxNumIterations(ICommand):
    def __init__(self, num_iterations: int) -> None:
        super().__init__()
        self._num_iterations = num_iterations

    def execute(self, ctx: Context[BioModelSupervisor]) -> None:
        ctx.session.set_max_num_iterations(self._num_iterations)


class ForwardPass(ICommand):
    def __init__(self, future, input_tensors):
        super().__init__()
        self._input_tensors = input_tensors
        self._future = future

    def execute(self, ctx: Context[Supervisors]) -> None:
        try:
            self._future.set_result(ctx.session.forward(self._input_tensors))
        except Exception as e:
            self._future.set_exception(e)


class CommandPriorityQueueUtils:
    """
    Utility for managing and processing commands in a priority queue.
    """

    def __init__(self) -> None:
        self.queue = CommandPriorityQueue()

    def send_command(self, cmd: ICommand) -> None:
        if not isinstance(cmd, ICommand):
            raise ValueError(f"Expected instance of ICommand got {cmd}")

        logger.debug("Sending command %s", cmd)
        self.queue.put(cmd)

    def process_commands(self, session):
        cmd: ICommand = self.queue.get()
        ctx = Context(supervisor=session)
        logger.debug("Executing %s", cmd)

        try:
            cmd.execute(ctx)
        except Exception as e:
            logger.exception(f"Failed to execute %s with exception {e}", cmd)
        finally:
            self.queue.task_done()
        logger.debug(f"Finished executing {cmd}")

        return cmd.is_stop()


class CommandPriorityQueue(queue.PriorityQueue):
    COMMAND_PRIORITIES = {ShutdownWithTeardownCmd: 0}

    @dataclass(order=True)
    class _PrioritizedItem:
        priority: typing.Tuple[int, int]
        item: ICommand = field(compare=False)

    __counter = itertools.count()

    @classmethod
    def _make_queue_item(cls, cmd: ICommand):
        if cmd.is_stop():
            priority = 0
        else:
            priority = cls.COMMAND_PRIORITIES.get(type(cmd), 999)
        return cls._PrioritizedItem((priority, next(cls.__counter)), cmd)

    def put(self, cmd: ICommand, block=True, timeout=None) -> None:
        return super().put(self._make_queue_item(cmd), block, timeout)

    def get(self, block=True, timeout=None):
        queue_item = super().get(block, timeout)
        return queue_item.item
