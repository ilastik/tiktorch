import io
import logging
import torch.nn, torch.optim
import threading

from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import closing
from copy import deepcopy
from multiprocessing.connection import Connection
from torch.utils.data import DataLoader
from typing import Any, List, Generic, Iterator, Iterable, Sequence, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from inferno.trainers import Trainer as InfernoTrainer

from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION, TRAINING, VALIDATION, REQUEST_FOR_DEVICES
from .datasets import DynamicDataset

from tiktorch.rpc import RPCInterface, exposed, Shutdown
from tiktorch.rpc.mp import MPServer
from tiktorch.tiktypes import TikTensor, TikTensorBatch

# logging.basicConfig(level=logging.INFO)
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {"default": {"level": "DEBUG", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"}},
        "loggers": {"": {"handlers": ["default"], "level": "DEBUG", "propagate": True}},
    }
)


class TikTrainer(InfernoTrainer):
    def __init__(self, *args, break_event: Optional[threading.Event] = None, **kwargs):
        self.break_event = break_event
        super().__init__(*args, **kwargs)

    def stop_fitting(self, max_num_iterations=None, max_num_epochs=None):
        if self.break_event is not None and self.break_event.is_set():
            return True
        else:
            return super().stop_fitting(max_num_iterations=max_num_iterations, max_num_epochs=max_num_epochs)

    @classmethod
    def build(cls, *args, break_event: threading.Event = None, **kwargs):
        trainer = super().build(*args, **kwargs)
        trainer.break_event = break_event
        return trainer


class ITraining(RPCInterface):
    @exposed
    def set_devices(self, device_names: Sequence[str]):
        raise NotImplementedError()

    @exposed
    def shutdown(self):
        raise Shutdown

    @exposed
    def resume_training(self):
        raise NotImplementedError

    @exposed
    def pause_training(self):
        raise NotImplementedError()

    @exposed
    def update_dataset(self, name: str, data: TikTensorBatch):
        raise NotImplementedError()


def run(conn: Connection, config: dict, model: torch.nn.Module, optimizer_state: bytes = b""):
    training_proc = TrainingProcess(config, model, optimizer_state)
    srv = MPServer(training_proc, conn)
    srv.listen()


training_settings_lock = threading.Lock()


class TrainingProcess(ITraining):
    """
    Process to run an inferno trainer instance to train a neural network. This instance is used for validation as well.
    """

    name = "TrainingProcess"

    def __init__(self, config: dict, model: torch.nn.Module, optimizer_state: bytes = b""):
        self.logger = logging.getLogger(self.name)
        self.logger.info("Starting")
        assert hasattr(self, SHUTDOWN[0]), "make sure the 'shutdown' method has the correct name"
        self._shutdown_event = threading.Event()

        self.model = model
        self.optimizer_state = optimizer_state
        self.max_num_iterations = 0

        self.datasets = {TRAINING: DynamicDataset(), VALIDATION: DynamicDataset()}
        self.update_loader = {TRAINING: True, VALIDATION: True}
        self.loader_kwargs = {
            TRAINING: {"dataset": self.datasets[TRAINING]},
            VALIDATION: {"dataset": self.datasets[VALIDATION]},
        }

        # setup inferno trainer configuration
        trainer_config = {
            "logger_config": {"name": "InfernoTrainer"},
            "max_num_iterations": self.max_num_iterations,
            "max_num_epochs": "inf",
        }
        # Some keys may not be set by the user
        if "max_num_iterations" in config:
            raise ValueError(
                "max_num_iterations is reserved for internal use."
                "The user should set max_num_iterations_per_update instead"
            )

        # Catch all forbidden keys we might have forgotten to implement an Exception for
        for key in trainer_config.keys():
            if key in config:
                raise ValueError("%s reserved for internal use. Do not specify in config!")

        # Some keys are required from the user
        if "optimizer_config" not in config:
            raise ValueError("Missing optimizer configuration!")

        trainer_config.update(deepcopy(config))
        if "max_num_iterations_per_update" not in config:
            config["max_num_iterations_per_update"] = 100

        self.config = config
        self.trainer_config = trainer_config

        self._train_event = threading.Event()
        self._update_trainer_event = threading.Event()
        self._update_trainer_event.set()
        self.trainer = TikTrainer.build(model=self.model, break_event=self._shutdown_event, **self.trainer_config)

        self.training_thread = threading.Thread(target=self._training_worker)
        self.training_thread.start()

    def end_of_training_iteration(self, iteration_num, trigger):
        if not self._train_event.is_set() or self._shutdown_event.is_set():
            raise StopIteration

    def end_of_validation_iteration(self, trigger):
        if not self._train_event.is_set() or self._shutdown_event.is_set():
            raise StopIteration

    def _training_worker(self):
        self.logger.info("Training thread started")
        # todo: configure (create/load) inferno trainer fully
        trainer = self.trainer

        if self.optimizer_state:
            optimizer = self.create_optimizer(self.optimizer_state)
            if optimizer is not None:
                trainer.optimizer = optimizer

        while not self._shutdown_event.is_set():
            try:
                self._train_event.wait(timeout=1)
            except TimeoutError:
                pass

            if self._train_event.is_set():
                if self._update_trainer_event.is_set():
                    self._update_trainer_event.clear()
                    self.logger.info("Update trainer")
                    trainer.set_max_num_iterations(self.max_num_iterations)
                    for name in (TRAINING, VALIDATION):
                        if self.update_loader[name]:
                            with training_settings_lock:
                                self.update_loader[name] = False
                                trainer.bind_loader(name, DataLoader(**self.loader_kwargs[name]))

                # trainer.register_callback(self.end_of_training_iteration, trigger="end_of_training_iteration")
                # trainer.register_callback(self.end_of_validation_iteration, trigger="end_of_validation_iteration")
                with training_settings_lock:
                    trainer.set_max_num_iterations(self.max_num_iterations)

                self.logger.info("Start training for %d iterations", self.max_num_iterations - trainer._iteration_count)
                trainer.fit()
                # num_iterations = self.max_num_iterations - trainer._iteration_count
                # assert num_iterations > 0
                # trainer.train_for(
                #     num_iterations=num_iterations,
                #     break_callback=lambda *args: not self._train_event.is_set() or self._shutdown_event.is_set(),
                # )
            else:
                pass

    def create_optimizer(self, optimizer_state: bytes) -> Optional[torch.optim.Optimizer]:
        try:
            optimizer: torch.optim.Optimizer = getattr(torch.optim, self.trainer_config["optimizer_config"]["method"])
            with closing(io.BytesIO(optimizer_state)) as f:
                optimizer.load_state_dict(torch.load(f))
        except Exception as e:
            self.logger.warning(
                "Could not load optimizer state due to %s.\nCreating new optimizer from %s",
                e,
                self.config["optimizer_config"],
            )
        else:
            return optimizer

    def shutdown(self) -> None:
        self._shutdown_event.set()
        self.training_thread.join()
        self.logger.debug("Shutdown complete")

    def resume_training(self) -> None:
        self._train_event.set()

    def pause_training(self) -> None:
        self._train_event.clear()
        # with training_settings_lock:
        #     self.trainer.set_max_num_iterations(0)

    def update_dataset(self, name: str, data: TikTensorBatch) -> None:
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        self.datasets[name].update(data)
        if name == TRAINING:
            self.max_num_iterations += self.config["max_num_iterations_per_update"] * len(data)

        self._update_trainer_event.set()

    def update_hparams(self, name: str, hparams: dict):
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        for key, value in hparams.items():
            if key in ("batch_size",):
                with training_settings_lock:
                    self.update_loader[name] = True
                    self.loader_kwargs[name][key] = value
            else:
                raise NotImplementedError(f"How to set {key} as a hyper parameter?")

        self._update_trainer_event.set()

    # def update_devices(self, devices: List[torch.device]):
    #     if devices != self.devices:
    #         # todo: switch devices for inferno trainer
    #         if "cpu" in devices:
    #             assert all(["cpu" in d for d in devices]), "cannot mix cpu and gpu atm"
    #             trainer.cpu()
    #         else:
    #             trainer.cuda(devices=devices)
    #
    #     now_idle = [d for d in self.devices if d not in devices]
    #     if now_idle:
    #         self.handler_conn.send(("report_idle", {"name": self.name, "devices": now_idle}))
    #
    #     self.devices = devices
    #     if self.devices:
    #         while self.waiting_for_devices:
    #             self.waiting_for_devices.pop()()
