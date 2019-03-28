import io
import logging
import torch.nn, torch.optim
import threading
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import closing
from copy import deepcopy
from multiprocessing.connection import Connection
from torch.utils.data import DataLoader
from typing import Any, List, Generic, Iterator, Iterable, Sequence, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from inferno.trainers import Trainer as InfernoTrainer

from .constants import TRAINING, VALIDATION
from .datasets import DynamicDataset

from tiktorch.rpc import RPCInterface, exposed, Shutdown
from tiktorch.rpc.mp import MPServer
from tiktorch.tiktypes import TikTensor, TikTensorBatch
from tiktorch import log


class TikTrainer(InfernoTrainer):
    def __init__(self, *args, break_events: Optional[List[threading.Event]] = None, **kwargs):
        self.break_events = break_events
        super().__init__(*args, **kwargs)

    def stop_fitting(self, max_num_iterations=None, max_num_epochs=None):
        if self.break_events and any([e.is_set() for e in self.break_events]):
            return True
        else:
            return super().stop_fitting(max_num_iterations=max_num_iterations, max_num_epochs=max_num_epochs)

    @classmethod
    def build(cls, *args, break_events: List[threading.Event] = None, **kwargs):
        trainer = super().build(*args, **kwargs)
        trainer.break_events = break_events
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

    @exposed
    def update_hparams(self, name: str, hparams: Dict):
        raise NotImplementedError


def run(
    conn: Connection,
    config: dict,
    model: torch.nn.Module,
    optimizer_state: bytes = b"",
    log_queue: Optional[mp.Queue] = None,
):
    log.configure(log_queue)
    training_proc = TrainingProcess(config, model, optimizer_state)
    srv = MPServer(training_proc, conn)
    srv.listen()


training_settings_lock = threading.Lock()


class TrainingProcess(ITraining):
    """
    Process to run an inferno trainer instance to train a neural network. This instance is used for validation as well.
    """

    trainer_defaults = {
        "criterion_config": {"method": "MSELoss"},
        "logger_config": {"name": "InfernoTrainer"},
        "max_num_iterations": 0,
        "max_num_iterations_per_update": 100,
        "max_num_epochs": "inf",
        "optimizer_config": {"method": "Adam"},
    }

    def __init__(self, config: dict, model: torch.nn.Module, optimizer_state: bytes = b""):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting")
        self._shutdown_event = threading.Event()

        self.model = model
        self.optimizer_state = optimizer_state

        self.datasets = {TRAINING: DynamicDataset(), VALIDATION: DynamicDataset()}
        self.update_loader = {TRAINING: True, VALIDATION: True}
        self.loader_kwargs = {
            TRAINING: {"dataset": self.datasets[TRAINING]},
            VALIDATION: {"dataset": self.datasets[VALIDATION]},
        }

        for key, default in self.trainer_defaults.items():
            if key not in config:
                config[key] = default

        self.config = config

        self._pause_event = threading.Event()
        self._pause_event.set()
        self._update_trainer_event = threading.Event()
        self._update_trainer_event.set()
        self.trainer = TikTrainer.build(
            model=self.model, break_event=self._shutdown_event, **self.create_trainer_config()
        )

        self.training_thread = threading.Thread(target=self._training_worker)
        self.training_thread.start()

    # def end_of_training_iteration(self, iteration_num, trigger):
    #     if not self._pause_event.is_set() or self._shutdown_event.is_set():
    #         raise StopIteration
    #
    def end_of_validation_iteration(self, trigger):
        pass  # todo: return validation

    def create_trainer_config(self) -> Dict:
        trainer_config = {}
        for key, default in self.trainer_defaults.items():
            if key in self.config:
                trainer_config[key] = self.config[key]
            else:
                trainer_config[key] = default

        return trainer_config

    def _training_worker(self):
        self.logger.info("Training thread started")
        # todo: configure (create/load) inferno trainer fully
        trainer = TikTrainer.build(
            model=self.model,
            break_events=[self._shutdown_event, self._pause_event, self._update_trainer_event],
            **self.create_trainer_config(),
        )
        # trainer.register_callback(self.end_of_training_iteration, trigger="end_of_training_iteration")
        trainer.register_callback(self.end_of_validation_iteration, trigger="end_of_validation_iteration")

        if self.optimizer_state:
            optimizer = self.create_optimizer(self.optimizer_state)
            if optimizer is not None:
                trainer.optimizer = optimizer

        while not self._shutdown_event.is_set():
            try:
                self._pause_event.wait(timeout=1)
            except TimeoutError:
                pass

            if not self._pause_event.is_set():
                if self._update_trainer_event.is_set():
                    self.logger.info("Update trainer")
                    with training_settings_lock:
                        self._update_trainer_event.clear()
                        trainer.set_max_num_iterations(self.config["max_num_iterations"])
                        for name in (TRAINING, VALIDATION):
                            if self.update_loader[name]:
                                self.update_loader[name] = False
                                trainer.bind_loader(name, DataLoader(**self.loader_kwargs[name]))

                self.logger.info(
                    "Start training for %d iterations", self.config["max_num_iterations"] - trainer._iteration_count
                )
                trainer.fit()

    def create_optimizer(self, optimizer_state: bytes) -> Optional[torch.optim.Optimizer]:
        try:
            optimizer: torch.optim.Optimizer = getattr(torch.optim, self.config["optimizer_config"]["method"])
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
        raise Shutdown

    def resume_training(self) -> None:
        self._pause_event.clear()

    def pause_training(self) -> None:
        self._pause_event.set()
        # with training_settings_lock:
        #     self.trainer.set_max_num_iterations(0)

    def update_dataset(self, name: str, data: TikTensorBatch) -> None:
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        self.datasets[name].update(data)
        if name == TRAINING:
            self.config["max_num_iterations"] += self.config["max_num_iterations_per_update"] * len(data)

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
