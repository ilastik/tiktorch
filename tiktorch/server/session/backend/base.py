from __future__ import annotations

import logging
import threading
import typing
from concurrent.futures import Future

from bioimageio.core.prediction_pipeline import PredictionPipeline

from tiktorch.configkeys import TRAINING, VALIDATION
from tiktorch.server.session import types
from tiktorch.server.session.backend import commands, supervisor
from tiktorch.tiktypes import TikTensorBatch

logger = logging.getLogger(__name__)


class SessionBackend:
    def __init__(self, pipeline: PredictionPipeline):
        self._supervisor = supervisor.Supervisor(pipeline)
        self._supervisor_thread = threading.Thread(target=self._supervisor.run, name="ModelThread")
        self._supervisor_thread.start()

    def update_dataset(self, name: str, *, data: TikTensorBatch, labels: TikTensorBatch) -> None:
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        update_cmd = commands.UpdateDatasetCmd(name, raw_data=data, labels=labels)
        self._supervisor.send_command(update_cmd)

    def set_max_num_iterations(self, num: int) -> None:
        self._supervisor.send_command(commands.SetMaxNumIterations(num))

    def forward(self, input_tensors):
        res = Future()
        self._supervisor.send_command(commands.ForwardPass(res, input_tensors))
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

    def get_idle(self) -> bool:
        return self._supervisor.state == types.State.Paused

    def on_idle(self, callback: typing.Callable[[], None]) -> None:
        self._supervisor.on_idle(callback)
