import copy
import io
import logging
import zipfile
import multiprocessing as mp
import os
import threading
import time
import queue
from datetime import datetime
from multiprocessing.connection import Connection
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import torch
import torch.nn
import torch.optim
from inferno.extensions import criteria as inferno_criteria
from inferno.io.transform import Compose, Transform
from inferno.trainers.callbacks.logging import TensorboardLogger
from inferno.utils.exceptions import NotSetError


from tiktorch import log
from tiktorch.configkeys import (
    BATCH_SIZE,
    DIRECTORY,
    LOGGING,
    LOSS_CRITERION_CONFIG,
    NAME,
    NUM_ITERATIONS_DONE,
    NUM_ITERATIONS_MAX,
    NUM_ITERATIONS_PER_UPDATE,
    OPTIMIZER_CONFIG,
    TORCH_VERSION,
    TRAINING,
    TRAINING_LOSS,
    TRAINING_SHAPE,
    TRAINING_SHAPE_LOWER_BOUND,
    TRAINING_SHAPE_UPPER_BOUND,
    TRANSFORMS,
    VALIDATION,
)
from tiktorch.rpc import RPCFuture, RPCInterface, Shutdown, exposed
from tiktorch.rpc.mp import MPServer
from tiktorch.server.utils import get_transform
from tiktorch.tiktypes import LabeledTikTensorBatch, TikTensor, TikTensorBatch
from tiktorch.types import ModelState
from tiktorch.utils import add_logger, get_error_msg_for_invalid_config
from tiktorch.server.reader import eval_model

from tiktorch.server.datasets import DynamicDataset
from .interface import ITraining
from . import worker


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


def run(
    conn: Connection,
    config: dict,
    model: torch.nn.Module,
    optimizer_state: bytes = b"",
    log_queue: Optional[mp.Queue] = None,
):
    try:
        # from: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    except ModuleNotFoundError:
        pass  # probably running on windows

    log.configure(log_queue)
    training_proc = TrainingProcess(config, model, optimizer_state)
    srv = MPServer(training_proc, conn)
    srv.listen()


def make_datasets(config):
    DEFAULT_TRANSFORM = {"Normalize": {"apply_to": [0]}}

    def composed_transforms(transforms):
        transforms = transforms or DEFAULT_TRANSFORM
        return Compose(*[get_transform(name, **kwargs) for name, kwargs in transforms.items()])

    training_transform = composed_transforms(config[TRAINING].get(TRANSFORMS))
    validation_transform = composed_transforms(config[VALIDATION].get(TRANSFORMS))

    return {
        TRAINING: DynamicDataset(transform=training_transform),
        VALIDATION: DynamicDataset(transform=validation_transform),
    }


class ConfigBuilder:
    DEFAULTS = {
        LOSS_CRITERION_CONFIG: {"method": "MSELoss"},
        NUM_ITERATIONS_MAX: 0,
        NUM_ITERATIONS_PER_UPDATE: 1,
        OPTIMIZER_CONFIG: {"method": "Adam"},
    }

    @classmethod
    def build(cls, config):
        result = {}

        for key, default in cls.DEFAULTS.items():
            if key not in config[TRAINING]:
                config[TRAINING][key] = default

        for key, default in cls.DEFAULTS.items():
            value = config[TRAINING].get(key, default)

            if key == LOSS_CRITERION_CONFIG:
                kwargs = dict(value)
                method = kwargs.pop("method")
                criterion_class = cls._resove_loss(method)
                criterion_config = {"method": LossWrapper(criterion_class(**kwargs), SparseOneHot())}
                result[INFERNO_NAMES.get(key, key)] = criterion_config
            else:
                result[INFERNO_NAMES.get(key, key)] = value

        result[INFERNO_LOGGER_CONFIG] = {"name": "InfernoTrainer"}
        result[INFERNO_MAX_NUM_EPOCHS] = "inf"

        return result

    @classmethod
    def _resove_loss(cls, loss_name):
        if not isinstance(loss_name, str):
            raise ValueError(f"Expected string as loss_name, got {loss_name}")

        criterion_class = getattr(torch.nn, loss_name, None)
        if criterion_class is None:
            # Look for it in extensions
            criterion_class = getattr(inferno_criteria, loss_name, None)

        if criterion_class is None:
            raise Exception(f"Criterion {loss_name} not found.")

        return criterion_class


class TrainingProcess(ITraining):
    """
    Process to run an inferno trainer instance to train a neural network. This instance is used for validation as well.
    """

    trainer_defaults = {
        LOSS_CRITERION_CONFIG: {"method": "MSELoss"},
        NUM_ITERATIONS_MAX: 0,
        NUM_ITERATIONS_PER_UPDATE: 1,
        OPTIMIZER_CONFIG: {"method": "Adam"},
    }

    def __init__(self, config: dict, model: torch.nn.Module, optimizer_state: bytes = b""):
        self._worker = None

        self.logger = logging.getLogger(__name__)

        self.common_model = model
        self.model = copy.deepcopy(model)
        self.optimizer_state = optimizer_state

        self.config = config

        self.datasets = make_datasets(config)

        self.trainer = worker.TikTrainer.build(
            dataset_by_name=self.datasets, model=self.model, **ConfigBuilder.build(config)
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
        self.trainer._iteration_count = self.config[TRAINING].get(NUM_ITERATIONS_DONE, 0)

        if self.optimizer_state:
            optimizer = self.create_optimizer(self.optimizer_state)
            if optimizer is not None:
                self.trainer.build_optimizer(optimizer)

        self._worker = worker.TrainingWorker(self.trainer)

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

    def remove_data(self, name: str, ids: List[str]) -> None:
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        for id_ in ids:
            self.datasets[name].remove(id_)

    def update_config(self, partial_config: dict) -> None:
        return
        assert not get_error_msg_for_invalid_config(partial_config)

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
                        raise NotImplementedError("Batch size update")
            elif key in [NAME, TORCH_VERSION]:
                self.config[key] = value
            else:
                raise NotImplementedError(f"How to set {key} as a hyper parameter?")

    def wait_for_idle(self) -> RPCFuture:
        f = RPCFuture()

        def _call():
            f.set_result(None)

        self._worker.on_idle(_call)
        return f

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

    def set_devices(self, devices: List[torch.device]) -> List[torch.device]:
        """
        set devices to train on. This request blocks until previous devices are free.
        :param devices: devices to use for training
        """
        return self._worker.set_devices(devices)

    def get_idle(self) -> bool:
        return self._worker.get_idle()

    def shutdown(self) -> Shutdown:
        self._worker.shutdown()
        return Shutdown()

    def resume_training(self) -> None:
        self._worker.resume_training()

    def pause_training(self) -> None:
        self._worker.pause_training()

    def update_dataset(self, name: str, data: TikTensorBatch, labels: TikTensorBatch) -> None:
        self._worker.update_dataset(name, data=data, labels=labels)

        self.config[TRAINING][NUM_ITERATIONS_MAX] += self.config[TRAINING][NUM_ITERATIONS_PER_UPDATE] * len(data)
        self._worker.set_max_number_of_iterations(self.config[TRAINING][NUM_ITERATIONS_MAX])


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


class ModelProcess(ITraining):
    def __init__(self, model_zip: bytes, devices: List[str]) -> None:
        with zipfile.ZipFile(io.BytesIO(model_zip)) as model_file:
            model = eval_model(model_file, devices)
        self._worker = worker.TrainingWorker(model)

    def forward(self, input_tensor):
        torch_tensor = torch.from_numpy(input_tensor)
        torch_result = self._worker.forward(torch_tensor)
        return torch_result.numpy()

    def shutdown(self) -> Shutdown:
        self._worker.shutdown()
        return Shutdown()


def run_model_process(conn: Connection, model_zip: bytes, devices: List[str], log_queue: Optional[mp.Queue] = None):
    try:
        # from: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    except ModuleNotFoundError:
        pass  # probably running on windows

    if log_queue:
        log.configure(log_queue)
    model_proc = ModelProcess(model_zip, devices)
    srv = MPServer(model_proc, conn)
    srv.listen()
