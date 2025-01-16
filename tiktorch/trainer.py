from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, List, TypeVar

import torch
import yaml
from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.trainer import UNetTrainer
from pytorch3dunet.unet3d.utils import create_lr_scheduler, create_optimizer, get_tensorboard_formatter
from torch import nn

T = TypeVar("T", bound=Callable)

logger = logging.getLogger(__name__)


class Callbacks(ABC, Generic[T]):
    def __init__(self):
        self._callbacks: List[T] = []

    def register(self, callback: T):
        self._callbacks.append(callback)

    def unregister(self, callback: T):
        self._callbacks.remove(callback)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class BaseCallbacks(Callbacks[T]):
    def __call__(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)


class ShouldStopCallbacks(Callbacks[Callable[[], bool]]):
    def __call__(self, *args, **kwargs):
        for callback in self._callbacks:
            if callback():
                return True


ErrorCallbacks = BaseCallbacks[Callable[[Exception], None]]


class ModelPhase(Enum):
    Train = "train"
    Eval = "val"


@dataclass(frozen=True)
class Logs:
    mode: ModelPhase
    loss: float
    eval_score: float
    iteration: int
    epoch: int
    max_epochs: int
    iteration: int
    max_iterations: int

    def __str__(self):
        iterations = f"Iteration[{self.iteration}/{self.max_iterations}]"
        epochs = f"Epochs[{self.epoch}/{self.max_epochs}]"
        return f"{epochs}, {iterations}: mode={self.mode}, loss={self.loss}, eval_score={self.eval_score}"


LogsCallbacks = Callbacks[Callable[[Logs], None]]


class TrainerAction(Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    SHUTDOWN = "shutdown"


class TrainerState(Enum):
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    FAILED = 3
    FINISHED = 4


class Trainer(UNetTrainer):
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        loss_criterion,
        eval_criterion,
        loaders,
        checkpoint_dir,
        max_num_epochs,
        max_num_iterations,
        validate_after_iters=200,
        log_after_iters=100,
        validate_iters=None,
        num_iterations=1,
        num_epoch=0,
        eval_score_higher_is_better=True,
        tensorboard_formatter=None,
        skip_train_validation=False,
        resume=None,
        pre_trained=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_criterion=loss_criterion,
            eval_criterion=eval_criterion,
            loaders=loaders,
            checkpoint_dir=checkpoint_dir,
            max_num_epochs=max_num_epochs,
            max_num_iterations=max_num_iterations,
            validate_after_iters=validate_after_iters,
            log_after_iters=log_after_iters,
            validate_iters=validate_iters,
            num_iterations=num_iterations,
            num_epoch=num_epoch,
            eval_score_higher_is_better=eval_score_higher_is_better,
            tensorboard_formatter=tensorboard_formatter,
            skip_train_validation=skip_train_validation,
            resume=resume,
            pre_trained=pre_trained,
            **kwargs,
        )
        self.logs_callbacks: LogsCallbacks = BaseCallbacks()
        self.should_stop_callbacks: Callbacks = ShouldStopCallbacks()

    def fit(self):
        return super().fit()

    def train(self):
        return super().train()

    def validate(self):
        return super().validate()

    def forward(self, input_tensors):
        self.model.eval()
        with torch.no_grad():
            self.model(input_tensors)

    def should_stop(self) -> bool:
        """
        Intervene on how to stop the training.
        """
        return self.should_stop_callbacks() or self.should_stop_model_criteria()

    def should_stop_model_criteria(self) -> bool:
        """
        Retain the logic designed by a custom model on how to stop the training
        e.g. learning rate lower than a threshold.
        """
        return super().should_stop()

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        logs = Logs(
            mode=ModelPhase(phase),
            loss=loss_avg,
            eval_score=eval_score_avg,
            iteration=self.num_iterations,
            epoch=self.num_epochs,
            max_epochs=self.max_num_epochs,
            max_iterations=self.max_num_iterations,
        )
        self.logs_callbacks(logs)
        # todo: why the internal training logging isn't printed on the stdout, although it is set
        logger.info(str(logs))
        return super()._log_stats(phase, loss_avg, eval_score_avg)


class TrainerYamlParser:
    def __init__(self, yaml_string: str):
        self._yaml_string = yaml_string
        self._yaml_config = yaml.safe_load(self._yaml_string)

    def get_device(self):
        return self._yaml_config["device"]

    def parse(self) -> Trainer:
        """
        Source: pytorch 3d unet
        """

        config = self._yaml_config

        model = get_model(config["model"])

        if torch.cuda.device_count() > 1 and not config["device"] == "cpu":
            model = nn.DataParallel(model)
        if torch.cuda.is_available() and not config["device"] == "cpu":
            model = model.cuda()

        # Create loss criterion
        loss_criterion = get_loss_criterion(config)
        # Create evaluation metric
        eval_criterion = get_evaluation_metric(config)

        # Create data loaders
        loaders = get_train_loaders(config)

        # Create the optimizer
        optimizer = create_optimizer(config["optimizer"], model)

        # Create learning rate adjustment strategy
        lr_scheduler = create_lr_scheduler(config.get("lr_scheduler", None), optimizer)

        trainer_config = config["trainer"]
        # Create tensorboard formatter
        tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop("tensorboard_formatter", None))
        # Create trainer
        resume = trainer_config.pop("resume", None)
        pre_trained = trainer_config.pop("pre_trained", None)

        return Trainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_criterion=loss_criterion,
            eval_criterion=eval_criterion,
            loaders=loaders,
            tensorboard_formatter=tensorboard_formatter,
            resume=resume,
            pre_trained=pre_trained,
            **trainer_config,
        )
