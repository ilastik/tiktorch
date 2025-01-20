from __future__ import annotations

import logging
from abc import ABC
from concurrent.futures import Future
from pathlib import Path

from bioimageio.core import PredictionPipeline, Sample

from tiktorch.configkeys import TRAINING, VALIDATION
from tiktorch.server.session.backend import commands
from tiktorch.server.session.backend.supervisor import BioModelSupervisor, QueueTasks, TrainerState, TrainerSupervisor
from tiktorch.tiktypes import TikTensorBatch
from tiktorch.trainer import Trainer

logger = logging.getLogger(__name__)


class SessionBackend(ABC):
    def __init__(self, supervisor):
        self._supervisor = supervisor
        self._queue_tasks = QueueTasks(supervisor)
        self._queue_tasks.start()

    def shutdown(self):
        self._queue_tasks.shutdown()
        logger.debug("Shutdown complete")


class BioModelSessionBackend(SessionBackend):
    """Session backend for bioimageio models

    Currently used only for inference.
    """

    def __init__(self, pipeline: PredictionPipeline):
        supervisor = BioModelSupervisor(pipeline)
        super().__init__(supervisor)

    def update_dataset(self, name: str, *, data: TikTensorBatch, labels: TikTensorBatch) -> None:
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        update_cmd = commands.UpdateDatasetCmd(name, raw_data=data, labels=labels)
        self._queue_tasks.send_command(update_cmd)

    def set_max_num_iterations(self, num: int) -> None:
        self._queue_tasks.send_command(commands.SetMaxNumIterations(num))

    def forward(self, input_tensors):
        res = Future()
        self._queue_tasks.send_command(commands.ForwardPass(res, input_tensors))
        return res


class TrainerSessionBackend(SessionBackend):
    """Session backend for training

    Currently, supports only custom unet models decoupled from bioimageio models
    """

    def __init__(self, trainer: Trainer):
        self._trainer = trainer
        supervisor = TrainerSupervisor(trainer)
        super().__init__(supervisor)

    def forward(self, input_tensors: Sample):
        res = Future()
        self._queue_tasks.send_command(commands.ForwardPass(res, input_tensors))
        return res

    def resume_training(self) -> None:
        resume_cmd = commands.ResumeTrainingCmd()
        self._queue_tasks.send_command(resume_cmd.awaitable)
        resume_cmd.awaitable.wait()

    def pause_training(self) -> None:
        pause_cmd = commands.PauseTrainingCmd()
        self._queue_tasks.send_command(pause_cmd.awaitable)
        pause_cmd.awaitable.wait()

    def start_training(self) -> None:
        start_cmd = commands.StartTrainingCmd()
        self._queue_tasks.send_command(start_cmd.awaitable)
        start_cmd.awaitable.wait()

    def save(self, file_path: Path) -> None:
        save_cmd = commands.SaveTrainingCmd(file_path)
        self._queue_tasks.send_command(save_cmd.awaitable)
        save_cmd.awaitable.wait()

    def export(self, file_path: Path) -> None:
        export_cmd = commands.ExportTrainingCmd(file_path)
        self._queue_tasks.send_command(export_cmd.awaitable)
        export_cmd.awaitable.wait()

    def get_state(self) -> TrainerState:
        return self._supervisor.get_state()

    def get_best_model_idx(self) -> int:
        return self._supervisor.get_best_model_idx()
