import copy
import io
import logging
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
from inferno.trainers import Trainer as InfernoTrainer
from inferno.trainers.callbacks.logging import TensorboardLogger
from inferno.utils.exceptions import NotSetError
from torch.utils.data import DataLoader, WeightedRandomSampler

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

from .datasets import DynamicDataLoaderWrapper, DynamicDataset, DynamicWeightedRandomSampler

try:
    # from: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
except ModuleNotFoundError:
    pass  # probably running on windows

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


class Devices:
    def __init__(self):
        self.devices = []
        self.base_device = "cpu"

    def update(self, devices: List[torch.device]) -> List[torch.device]:
        free_devices = [d for d in self.devices if d not in devices]

        if not devices:
            self.base_device = "cpu"
            self.devices = []

        else:
            self.base_device = devices[0].type
            if not all(d.type == self.base_device for d in devices):
                raise ValueError("Can't train on cpu and gpu at the same time")

            self.devices = devices

        return free_devices

    def __len__(self):
        return len(self.devices)


class TikTrainer(InfernoTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._break_cb = None

    def set_break_callback(self, callback):
        self._break_cb = callback

    @property
    def max_num_iterations(self):
        return self._max_num_iterations

    def stop_fitting(self, max_num_iterations=None, max_num_epochs=None):
        if self._break_cb and self._break_cb():
            return True
        else:
            return super().stop_fitting(max_num_iterations=max_num_iterations, max_num_epochs=max_num_epochs)

    @classmethod
    def build(cls, *args, **kwargs):
        return super().build(*args, **kwargs)

    def move_to(self, devices: Devices):
        if devices.base_device == "cpu":
            self.cpu()
        elif devices.base_device == "cuda":
            self.cuda(devices=[d.index for d in devices])
        else:
            raise ValueError(f"Unknown device type {devices.base_device}")

        # make sure optimizer states are on correct device
        for k in self.optimizer.state.keys():
            param_state = self.optimizer.state[k]
            for p in param_state.keys():
                try:
                    if not isinstance(param_state[p], int):
                        param_state[p] = param_state[p].to(devices.base_device)
                except Exception as e:
                    self.logger.exception("Failed to move optimizer to %s", devices)


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

    @exposed
    def remove_data(self, name: str, ids: List[str]) -> None:
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


class ICommand:
    __awaitable = None

    @property
    def awaitable(self):
        if not self.__awaitable:
            self.__awaitable = AwaitableCommand(self)

        return self.__awaitable

    def execute(self) -> None:
        raise NotImplementedError()


class AwaitableCommand(ICommand):
    def __init__(self, cmd: ICommand):
        self._cmd = cmd
        self._done_evt = threading.Event()

    def wait(self):
        self._done_evt.wait()

    def execute(self):
        try:
            self._cmd.execute()
        finally:
            self._done_evt.set()

    def __repr__(self):
        return f"Awaitable {self._cmd!r}"


class State:
    Idle = "idle"
    Paused = "paused"
    Running = "running"
    Stopped = "stopped"


class TrainingWorker:
    def __init__(self, trainer: TikTrainer) -> None:
        self._state = State.Stopped

        self._command_queue = queue.Queue()
        self._trainer = trainer
        self._trainer.set_break_callback(self.has_commands)
        self._devices = Devices()

    def send_command(self, cmd: ICommand) -> None:
        if not isinstance(cmd, ICommand):
            raise ValueError(f"Expected instance of ICommand got {cmd}")

        logger.debug("Sending command %s", cmd)
        self._command_queue.put(cmd)

    @property
    def state(self):
        return self._state

    def has_commands(self):
        return not self._command_queue.empty()

    def has_work(self):
        return self._trainer.max_num_iterations > self._trainer.iteration_count

    def set_devices(self, devices: List[torch.device]) -> List[torch.device]:
        free_devs = self._devices.update(devices)
        self._trainer.move_to(self._devices)
        self._update_state()
        return free_devs

    def transition_to(self, new_state: State) -> None:
        logger.debug("Attempting transition to state %s", new_state)
        self._state = new_state
        self._update_state()

    def set_max_num_iterations(self, num: int):
        self._trainer.set_max_num_iterations(num)
        self._update_state()

    def run(self):
        logger.info("Starting training worker")
        try:
            self._run()
        except Exception:
            logger.exception("Uncaught exception in training worker")
        finally:
            logger.info("Stopped training worker")

    def _run(self):
        self._set_state(State.Paused)

        while True:
            self._process_commands()

            if self.state == State.Stopped:
                break

            elif self._state == State.Idle or self._state == State.Paused:
                with self._command_queue.not_empty:
                    self._command_queue.not_empty.wait()

            elif self._state == State.Running:
                self._train()
                self._update_state()

    def _process_commands(self):
        while not self._command_queue.empty():
            try:
                cmd = self._command_queue.get_nowait()
                logger.debug("Executing %s", cmd)

                try:
                    cmd.execute()
                except Exception:
                    logger.exception("Failed to execute %s", cmd)
                finally:
                    self._command_queue.task_done()

            except queue.Empty:
                pass

    def _train(self):
        logger.info(
            "Start training for %d iterations", self._trainer.max_num_iterations - self._trainer.iteration_count
        )
        try:
            self._trainer.fit()
        except Exception as e:
            print("**************************EXC", e)
            logger.error("Exception training fit", exc_info=True)
            self.send_command(PauseCmd(self))

    def _update_state(self):
        if self._state == State.Running:
            should_idle = not (self._devices and self.has_work())
            if should_idle:
                self._set_state(State.Idle)

        elif self._state == State.Idle:
            should_run = self._devices and self.has_work()
            if should_run:
                self._set_state(State.Running)

    def _set_state(self, new_state: State) -> None:
        self._state = new_state
        logger.debug("Set new state %s", self._state)


class WorkerCmd(ICommand):
    def __init__(self, worker: TrainingWorker):
        self._worker = worker


class PauseCmd(WorkerCmd):
    def execute(self):
        self._worker.transition_to(State.Paused)


class ResumeCmd(WorkerCmd):
    def execute(self):
        self._worker.transition_to(State.Running)


class StopCmd(WorkerCmd):
    def execute(self):
        self._worker.transition_to(State.Stopped)


class SetDevicesCmd(ICommand):
    def __init__(self, worker, devices):
        self._worker = worker
        self._devices = devices

        self.result = None

    def execute(self):
        self.result = self._worker.set_devices(self._devices)


class UpdateDatasetCmd(ICommand):
    def __init__(self, trainer, dataset, loader_kwargs, *, raw_data, labels):
        self._trainer = trainer
        self._dataset = dataset
        self._raw_data = raw_data
        self._labels = labels
        self._loader_kwargs = loader_kwargs  # FIXME

    def execute(self):
        self._dataset.update(self._raw_data, self._labels)
        loader = DataLoader(**self._loader_kwargs)
        self._trainer.bind_loader("train", DynamicDataLoaderWrapper(loader))


class SetMaxNumberOfIterations(ICommand):
    def __init__(self, worker: TrainingWorker, num_iterations: int) -> None:
        self._worker = worker
        self._num_iterations = num_iterations

    def execute(self) -> None:
        self._worker.set_max_num_iterations(self._num_iterations)


logger = logging.getLogger(__name__)


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
        self.logger.info("started")
        self.shutdown_event = threading.Event()
        self.idle = False
        self._command_queue = queue.Queue()

        self.common_model = model
        self.model = copy.deepcopy(model)
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
        self.trainer._iteration_count = self.config[TRAINING].get(NUM_ITERATIONS_DONE, 0)

        if self.optimizer_state:
            optimizer = self.create_optimizer(self.optimizer_state)
            if optimizer is not None:
                self.trainer.build_optimizer(optimizer)

        self._worker = TrainingWorker(self.trainer)
        self._worker_thread = threading.Thread(target=self._worker.run, name="TrainingWorker")
        self._worker_thread.start()

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

    def set_devices(self, devices: List[torch.device]) -> List[torch.device]:
        """
        set devices to train on. This request blocks until previous devices are free.
        :param devices: devices to use for training
        """
        set_dev_cmd = SetDevicesCmd(self._worker, devices)
        set_dev_cmd_awaitable = set_dev_cmd.awaitable
        self._worker.send_command(set_dev_cmd_awaitable)
        set_dev_cmd_awaitable.wait()
        if set_dev_cmd.result is None:
            self.logger.error("Failed to set devices")
            return []

        return set_dev_cmd.result

    def get_idle(self):
        return self._worker.state in (State.Idle, State.Paused)

    def shutdown(self) -> Shutdown:
        self.logger.debug("Shutting down...")

        stop_cmd = StopCmd(self._worker)
        self._worker.send_command(stop_cmd.awaitable)
        stop_cmd.awaitable.wait()

        self.logger.debug("Shutdown complete")
        return Shutdown()

    def resume_training(self) -> None:
        self.logger.warning("RESUME")
        self._worker.send_command(ResumeCmd(self._worker))

    def pause_training(self) -> None:
        self.logger.warning("PAUSE")
        self._worker.send_command(PauseCmd(self._worker))

    def remove_data(self, name: str, ids: List[str]) -> None:
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        for id_ in ids:
            self.datasets[name].remove(id_)

    def update_dataset(self, name: str, data: TikTensorBatch, labels: TikTensorBatch) -> None:
        assert name in (TRAINING, VALIDATION), f"{name} not in ({TRAINING}, {VALIDATION})"
        update_cmd = UpdateDatasetCmd(
            self.trainer, self.datasets[name], self.loader_kwargs[name], raw_data=data, labels=labels
        )
        self._worker.send_command(update_cmd)
        self.config[TRAINING][NUM_ITERATIONS_MAX] += self.config[TRAINING][NUM_ITERATIONS_PER_UPDATE] * len(data)
        self._worker.send_command(SetMaxNumberOfIterations(self._worker, self.config[TRAINING][NUM_ITERATIONS_MAX]))

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
        print("APPLY ONE HOT", prediction.shape, target.shape)
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
