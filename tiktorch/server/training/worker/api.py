from __future__ import annotations

import typing
import threading
import logging
from concurrent.futures import Future

from . import base, commands, types
from tiktorch.configkeys import TRAINING, VALIDATION

if typing.TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


class TrainingWorker:
    def __init__(self, trainer):
        self._supervisor = base.Supervisor(trainer)
        self._supervisor_thread = threading.Thread(target=self._supervisor.run, name="ModelThread")
        self._supervisor_thread.start()

    def update_dataset(self, name: str, *, data: TikTensorBatch, labels: TikTensorBatch) -> None:
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        update_cmd = commands.UpdateDatasetCmd(name, raw_data=data, labels=labels)
        self._supervisor.send_command(update_cmd)

    def set_max_number_of_iterations(self, num: int) -> None:
        self._supervisor.send_command(commands.SetMaxNumberOfIterations(num))

    def forward(self, input_tensor):
        res = Future()
        self._supervisor.send_command(commands.ForwardPass(res, input_tensor))
        return res

    def shutdown(self) -> None:
        logger.debug("Shutting down...")

        stop_cmd = commands.StopCmd()
        self._supervisor.send_command(stop_cmd.awaitable)
        stop_cmd.awaitable.wait()

        self._supervisor_thread.join()

        logger.debug("Shutdown complete")

    def resume_training(self) -> None:
        resume_cmd = commands.ResumeCmd()
        self._supervisor.send_command(resume_cmd.awaitable)
        resume_cmd.awaitable.wait()

    def pause_training(self) -> None:
        self._supervisor.send_command(commands.PauseCmd())

    def set_devices(self, devices: List[torch.device]) -> List[torch.device]:
        """
        set devices to train on. This request blocks until previous devices are free.
        :param devices: devices to use for training
        """
        set_dev_cmd = commands.SetDevicesCmd(devices)
        self._supervisor.send_command(set_dev_cmd.awaitable)
        set_dev_cmd.awaitable.wait()

        if set_dev_cmd.result is None:
            logger.error("Failed to set devices")
            return []

        return set_dev_cmd.result

    def get_idle(self) -> bool:
        return self._supervisor.state == types.State.Paused

    def on_idle(self, callback: typing.Callable[[], []]) -> None:
        self._supervisor.on_idle(callback)
