import copy
import io
from datetime import datetime

import os

import logging
import torch.nn, torch.optim
import multiprocessing as mp
import time
import threading

from multiprocessing.connection import Connection
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import (
    Any,
    List,
    Generic,
    Iterator,
    Iterable,
    Sequence,
    TypeVar,
    Mapping,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
)

from inferno.trainers import Trainer as InfernoTrainer
from inferno.trainers.callbacks.logging import TensorboardLogger
from inferno.io.transform import Compose, Transform
from inferno.utils.exceptions import NotSetError
from inferno.extensions import criteria as inferno_criteria

from tiktorch.utils import add_logger, get_error_msg_for_invalid_config, get_transform
from tiktorch.rpc import RPCInterface, exposed, Shutdown, RPCFuture
from tiktorch.rpc.mp import MPServer
from tiktorch.tiktypes import TikTensor, LabeledTikTensorBatch, TikTensorBatch
from tiktorch.types import ModelState
from tiktorch import log
from tiktorch.configkeys import (
    NAME,
    TORCH_VERSION,
    TRAINING,
    VALIDATION,
    BATCH_SIZE,
    TRANSFORMS,
    TRAINING_SHAPE,
    TRAINING_SHAPE_LOWER_BOUND,
    TRAINING_SHAPE_UPPER_BOUND,
    NUM_ITERATIONS_DONE,
    NUM_ITERATIONS_MAX,
    NUM_ITERATIONS_PER_UPDATE,
    LOSS_CRITERION_CONFIG,
    OPTIMIZER_CONFIG,
    TRAINING_LOSS,
    LOGGING,
    DIRECTORY,
)

from tiktorch.handler.datasets import DynamicDataset


# inferno names
INFERNO_LOGGER_CONFIG = "logger_config"
INFERNO_MAX_NUM_EPOCHS = "max_num_epochs"

INFERNO_NAMES = {  # inferno names that we have an analogue to in the tiktorch config
    TRAINING: "train",
    VALIDATION: "validate",
    LOSS_CRITERION_CONFIG: "criterion_config",
    BATCH_SIZE: "batch_size",
    TRAINING_LOSS: "training_loss",
}


class TikTrainer(InfernoTrainer):
    def __init__(self, *args, break_events: Optional[List[threading.Event]] = None, **kwargs):
        self.break_events = break_events
        super().__init__(*args, **kwargs)

    @property
    def max_num_iterations(self):
        return self._max_num_iterations

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
    def set_devices(self, devices: Sequence[torch.device]) -> List[torch.device]:
        raise NotImplementedError

    @exposed
    def shutdown(self) -> Shutdown:
        raise NotImplementedError

    @exposed
    def resume_training(self) -> None:
        raise NotImplementedError

    @exposed
    def pause_training(self) -> None:
        raise NotImplementedError

    @exposed
    def get_idle(self) -> bool:
        raise NotImplementedError

    @exposed
    def update_dataset(self, name: str, data: TikTensorBatch, labels: TikTensorBatch):
        raise NotImplementedError

    @exposed
    def update_config(self, partial_config: dict) -> None:
        raise NotImplementedError

    @exposed
    def get_state(self) -> ModelState:
        raise NotImplementedError

    @exposed
    def get_model_state_dict(self) -> dict:
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


class TrainingProcess(ITraining):
    """
    Process to run an inferno trainer instance to train a neural network. This instance is used for validation as well.
    """

    trainer_defaults = {
        LOSS_CRITERION_CONFIG: {"method": "MSELoss"},
        NUM_ITERATIONS_MAX: 0,
        NUM_ITERATIONS_PER_UPDATE: 10,
        OPTIMIZER_CONFIG: {"method": "Adam"},
    }

    def __init__(self, config: dict, model: torch.nn.Module, optimizer_state: bytes = b""):
        self.logger = logging.getLogger(__name__)
        self.logger.info("started")
        self.shutdown_event = threading.Event()
        self.idle = False

        self.model = model
        self.logger.debug("here training init %s", self.model._modules["final_conv"]._parameters["weight"].data.mean())

        self.optimizer_state = optimizer_state
        self.training_settings_lock = threading.Lock()
        # self.devices = [torch.device("cpu")]
        self.devices = []
        self.base_device = "cpu"

        training_transform = Compose(
            *[
                get_transform(name, **kwargs)
                for name, kwargs in config[TRAINING].get(TRANSFORMS, {"Normalize": {"apply_to": [0]}}).items()
            ]
        )
        validation_transform = Compose(
            *[
                get_transform(name, **kwargs)
                for name, kwargs in config[VALIDATION].get(TRANSFORMS, {"Normalize": {"apply_to": [0]}}).items()
            ]
        )

        self.datasets = {
            TRAINING: DynamicDataset(transform=training_transform),
            VALIDATION: DynamicDataset(transform=validation_transform),
        }
        self.update_loader = {TRAINING: True, VALIDATION: True}
        self.loader_kwargs = {
            TRAINING: {"dataset": self.datasets[TRAINING]},
            VALIDATION: {"dataset": self.datasets[VALIDATION]},
        }

        for key, default in self.trainer_defaults.items():
            if key not in config[TRAINING]:
                config[TRAINING][key] = default

        self.config = config

        self._pause_event = threading.Event()
        self._pause_event.set()
        self.update_trainer_event = threading.Event()
        self.update_trainer_event.set()

        self.common_model = self.model
        self.trainer = TikTrainer.build(
            model=self.model,
            break_events=[self.shutdown_event, self._pause_event, self.update_trainer_event],
            **self.create_trainer_config(),
        )
        log_dir = self.config.get(LOGGING, {}).get(DIRECTORY, "")
        if os.path.exists(log_dir):
            log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            os.makedirs(log_dir, exist_ok=True)
            self.trainer.build_logger(
                TensorboardLogger,
                log_directory=log_dir,
                log_scalars_every=(1, "iteration"),
                log_images_every=(1, "epoch"),
            )
        self.trainer.register_callback(self.end_of_training_iteration, trigger="end_of_training_iteration")
        self.trainer.register_callback(self.end_of_validation_iteration, trigger="end_of_validation_iteration")
        self.trainer._iteration_count = self.config[TRAINING].get(NUM_ITERATIONS_DONE, 0)

        if self.optimizer_state:
            optimizer = self.create_optimizer(self.optimizer_state)
            if optimizer is not None:
                self.trainer.build_optimizer(optimizer)

        self.training_thread = threading.Thread(target=add_logger(self.logger)(self._training_worker), name="Training")
        self.training_thread.start()

    # def end_of_training_iteration(self, iteration_num, trigger):
    #     if not self._pause_event.is_set() or self.shutdown_event.is_set():
    #         raise StopIteration
    #
    def end_of_validation_iteration(self, trigger):
        pass  # todo: return validation

    def end_of_training_iteration(self, iteration_num, trigger):
        pass

    def create_trainer_config(self) -> Dict:
        trainer_config = {}
        for key, default in self.trainer_defaults.items():
            value = self.config[TRAINING].get(key, default)
            if key == LOSS_CRITERION_CONFIG:
                kwargs = dict(value)
                method = kwargs.pop("method")
                assert isinstance(method, str)
                # Look for criteria in torch
                criterion_class = getattr(torch.nn, method, None)
                if criterion_class is None:
                    # Look for it in extensions
                    criterion_class = getattr(inferno_criteria, method, None)
                assert criterion_class is not None, "Criterion {} not found.".format(method)
                criterion_config = {"method": LossWrapper(criterion_class(**kwargs), SparseOneHot())}
                trainer_config[INFERNO_NAMES.get(key, key)] = criterion_config
            else:
                trainer_config[INFERNO_NAMES.get(key, key)] = value

        trainer_config[INFERNO_LOGGER_CONFIG] = {"name": "InfernoTrainer"}
        trainer_config[INFERNO_MAX_NUM_EPOCHS] = "inf"

        return trainer_config

    def _training_worker(self):
        while True:
            if self.shutdown_event.is_set():
                break

            if self._pause_event.is_set():
                self.idle = True
                time.sleep(1)
            else:
                if self.update_trainer_event.is_set():
                    self.logger.info("Update trainer settings")
                    with self.training_settings_lock:
                        self.update_trainer_event.clear()
                        self.trainer.set_max_num_iterations(self.config[TRAINING][NUM_ITERATIONS_MAX])
                        self.logger.info(
                            "trainer iterations: %d/%d", self.trainer.iteration_count, self.trainer.max_num_iterations
                        )
                        for name in [TRAINING, VALIDATION]:
                            if self.update_loader[name]:
                                self.update_loader[name] = False
                                self.trainer.bind_loader(INFERNO_NAMES[name], DataLoader(**self.loader_kwargs[name]))

                if self.trainer.max_num_iterations > self.trainer.iteration_count:
                    self.idle = False
                    if self.devices:
                        self.logger.info(
                            "Start training for %d iterations",
                            self.trainer.max_num_iterations - self.trainer.iteration_count,
                        )

                        if self.base_device == "cpu":
                            self.trainer.cpu()
                        elif self.base_device == "cuda":
                            self.trainer.cuda(devices=[int(str(d).split(":")[1]) for d in self.devices])
                        else:
                            raise ValueError(self.base_device)

                        # make sure optimizer states are on correct device
                        for k in self.trainer.optimizer.state.keys():
                            param_state = self.trainer.optimizer.state[k]
                            for p in param_state.keys():
                                try:
                                    if not isinstance(param_state[p], int):
                                        param_state[p] = param_state[p].to(self.base_device)
                                except Exception as e:
                                    self.logger.debug(e)

                        self.logger.debug(
                            "here training before fit %s",
                            self.model._modules["final_conv"]._parameters["weight"].data.mean(),
                        )
                        self.trainer.fit()
                        # update common cpu model, as the trainer's model might be a gpu copy
                        self.logger.debug(
                            "here training after fit %s",
                            self.model._modules["final_conv"]._parameters["weight"].data.mean(),
                        )
                        self.logger.debug(
                            "here training after fit common before update %s",
                            self.common_model._modules["final_conv"]._parameters["weight"].data.mean(),
                        )
                        self.common_model.load_state_dict(self.trainer.model.state_dict())
                        self.logger.debug(
                            "here training after fit common %s",
                            self.common_model._modules["final_conv"]._parameters["weight"].data.mean(),
                        )
                        self.config[TRAINING][NUM_ITERATIONS_DONE] = self.trainer._iteration_count
                    else:
                        self.logger.info("Waiting for device")
                        time.sleep(1)
                else:
                    self.idle = True
                    time.sleep(1)

    def set_devices(self, devices: Sequence[torch.device]) -> List[torch.device]:
        """
        set devices to train on. This request blocks until previous devices are free.
        :param devices: devices to use for training
        """
        self.logger.debug("set devices %s", devices)
        if self.devices == devices:
            return []

        free_devices = [d for d in self.devices if d not in devices]

        with self.training_settings_lock:
            self.update_trainer_event.set()
            device_types = [d.type for d in devices]
            if "cpu" in device_types or len(devices) == 0:
                assert len(devices) <= 1, "Cannot train on cpu and gpu at the same time"
                # train on cpu
                self.base_device = "cpu"
                self.devices = [torch.device("cpu")]
            else:
                self.base_device = "cuda"
                self.devices = devices

        # wait for training worker to update training settings
        while not self.idle and self.update_trainer_event.is_set() and not self._pause_event.is_set():
            self.logger.debug("wait for old deviced to be free")
            time.sleep(2)

        self.logger.debug("new devices %s set", devices)
        return free_devices

    def get_idle(self):
        return self.idle

    def create_optimizer(self, optimizer_state: bytes) -> Optional[torch.optim.Optimizer]:
        try:
            kwargs = dict(self.config[TRAINING][OPTIMIZER_CONFIG])
            optimizer_class: Type[torch.optim.Optimizer] = getattr(torch.optim, kwargs.pop("method"))
            optimizer = optimizer_class(self.model.parameters(), **kwargs)
            try:
                optimizer.load_state_dict(torch.load(io.BytesIO(optimizer_state), map_location=self.base_device))
            except Exception as e:
                self.logger.warning(
                    "Could not load optimizer state due to %s.\nCreating new optimizer from %s",
                    e,
                    self.config[TRAINING][OPTIMIZER_CONFIG],
                )
            else:
                self.logger.info("restored optimizer state")
        except Exception as e:
            self.logger.exception(e)
            return None
        else:
            return optimizer

    def shutdown(self) -> Shutdown:
        self.logger.debug("Shutting down...")
        self.shutdown_event.set()
        try:
            self.training_thread.join(timeout=30)
        except TimeoutError as e:
            self.logger.error(e)

        self.logger.debug("Shutdown complete")
        return Shutdown()

    def resume_training(self) -> None:
        self.logger.warning("RESUME")
        self._pause_event.clear()

    def pause_training(self) -> None:
        self._pause_event.set()

    def update_dataset(self, name: str, data: TikTensorBatch, labels: TikTensorBatch) -> None:
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        self.datasets[name].update(data, labels)
        self.datasets[name].reset_indices()
        if name == TRAINING:
            old = self.config[TRAINING][NUM_ITERATIONS_MAX]
            with self.training_settings_lock:
                self.config[TRAINING][NUM_ITERATIONS_MAX] += self.config[TRAINING][NUM_ITERATIONS_PER_UPDATE] * len(
                    data
                )
                self.logger.info(
                    "increased %s from %d to %d", NUM_ITERATIONS_MAX, old, self.config[TRAINING][NUM_ITERATIONS_MAX]
                )
                ds = self.datasets[TRAINING]
                if ds:
                    self.loader_kwargs[TRAINING]["sampler"] = WeightedRandomSampler(
                        ds.get_weights(), len(ds), replacement=True
                    )
                else:
                    self.loader_kwargs[TRAINING].pop("sampler", None)

            # note: This sampler leads to an epoch, which might not see some of the samples in the training dataset
            #       (and others more than once)
            # todo: add more samplers (e.g. WeighedRandomBachSampler without replacement per batch)

        self.update_trainer_event.set()

    def update_config(self, partial_config: dict) -> None:
        assert not get_error_msg_for_invalid_config(partial_config)

        with self.training_settings_lock:
            self.update_trainer_event.set()
            for key, value in partial_config.items():
                if key in [TRAINING, VALIDATION]:
                    for subkey, subvalue in partial_config[key].items():
                        if subvalue is None:
                            # remove key from config
                            if subkey == TRAINING_SHAPE:
                                if len(self.datasets[TRAINING]) or len(self.datasets[VALIDATION]):
                                    raise NotImplementedError(
                                        "Cannot change training_shape after adding training or validation data"
                                    )
                            else:
                                raise NotImplementedError(f"How to remove {subkey} form config[{key}]?")

                            if subkey in self.config[key]:
                                del self.config[key][subkey]
                        elif subkey == BATCH_SIZE:
                            self.update_loader[key] = True
                            self.loader_kwargs[key][INFERNO_NAMES[BATCH_SIZE]] = subvalue
                    pass
                elif key in [NAME, TORCH_VERSION]:
                    self.config[key] = value
                else:
                    raise NotImplementedError(f"How to set {key} as a hyper parameter?")

    def get_state(self) -> ModelState:
        training_loss = self.trainer.get_state(INFERNO_NAMES[TRAINING_LOSS], default=float("Inf"))
        epoch = self.trainer.epoch_count
        weights_io = io.BytesIO()
        torch.save(self.model.state_dict(), weights_io)
        optim_state_io = io.BytesIO()

        if not isinstance(training_loss, float):
            training_loss = training_loss.item()

        try:
            torch.save(self.trainer.optimizer.state_dict(), optim_state_io)
        except NotSetError:
            optim_state = b""
        else:
            optim_state = optim_state_io.getvalue()

        return ModelState(
            loss=training_loss,
            epoch=epoch,
            model_state=weights_io.getvalue(),
            optimizer_state=optim_state,
            num_iterations_done=self.config[TRAINING].get(NUM_ITERATIONS_DONE, 0),
            num_iterations_max=self.config[TRAINING][NUM_ITERATIONS_MAX],
        )

    def get_model_state_dict(self) -> dict:
        state = self.model.state_dict()
        for k in state.keys():
            state[k] = state[k].cpu().detach()

        return state


class SparseOneHot(Transform):
    """Mask out the zero label """

    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def batch_function(self, tensors):
        prediction, target = tensors
        mask = torch.zeros_like(prediction)
        mask[target > 0] = 1
        mask.requires_grad = False
        one_hot_target = torch.zeros_like(prediction)
        for c in range(one_hot_target.shape[1]):
            label = c + 1
            one_hot_target[:, c] = target == label

        # mask prediction with mask
        masked_prediction = prediction * mask
        return masked_prediction, one_hot_target


# LossWrapper from neurofire.
# todo: add to inferno
class LossWrapper(torch.nn.Module):
    """
    Wrapper around a torch criterion.
    Enables transforms before applying the criterion.
    Should be subclassed for implementation.
    """

    def __init__(self, criterion, transforms=None, weight_function=None):
        super().__init__()
        # validate: the criterion needs to inherit from nn.Module
        # assert isinstance(criterion, nn.Module)
        self.criterion = criterion
        # validate: transforms need to be callable
        if transforms is not None:
            assert callable(transforms)
        self.transforms = transforms
        if weight_function is not None:
            assert callable(weight_function)
        self.weight_function = weight_function

    def apply_transforms(self, prediction, target):
        # check if the tensors (prediction and target are lists)
        # if so , we need to loop and apply the transforms to each element inidvidually
        is_listlike = isinstance(prediction, (list, tuple))
        if is_listlike:
            assert isinstance(target, (list, tuple))
        # list-like input
        if is_listlike:
            transformed_prediction, transformed_target = [], []
            for pred, targ in zip(prediction, target):
                tr_pred, tr_targ = self.transforms(pred, targ)
                transformed_prediction.append(tr_pred)
                transformed_target.append(tr_targ)
        # tensor input
        else:
            transformed_prediction, transformed_target = self.transforms(prediction, target)
        return transformed_prediction, transformed_target

    def forward(self, prediction, target):
        # calculate the weight based on prediction and target
        if self.weight_function is not None:
            weight = self.weight_function(prediction, target)
            self.criterion.weight = weight

        # apply the transforms to prediction and target or a list of predictions and targets
        if self.transforms is None:
            transformed_prediction, transformed_target = prediction, target
        else:
            transformed_prediction, transformed_target = self.apply_transforms(prediction, target)

        loss = self.criterion(transformed_prediction, transformed_target)
        return loss
