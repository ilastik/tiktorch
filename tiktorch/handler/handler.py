import importlib
import logging
import logging.config
import os.path
import pickle
import queue
import shutil
import sys
import tempfile
import threading
import torch
import time
import numpy as np
import bisect
import queue

from multiprocessing.connection import Connection, wait
from torch import multiprocessing as mp

from typing import Any, List, Generic, Iterator, Iterable, Sequence, Callable, Dict, Optional, Tuple, Set, Union

from tiktorch.rpc import RPCInterface, exposed, RPCFuture
from tiktorch.rpc.mp import MPServer, MPClient
from tiktorch.tiktypes import (
    TikTensor,
    TikTensorBatch,
    PointBase,
    Point2D,
    Point3D,
    Point4D,
    BatchPointBase,
    PointAndBatchPointBase,
)
from tiktorch.handler.constants import TRAINING, VALIDATION
from tiktorch.handler.training import run as run_training, ITraining
from tiktorch.handler.inference import run as run_inference, IInference
from tiktorch.handler.dryrun import run as run_dryrun, IDryRun
from tiktorch import log

from tiktorch.configkeys import (
    TRAINING_SHAPE,
    TRAINING_SHAPE_LOWER_BOUND,
    TRAINING_SHAPE_UPPER_BOUND,
    BATCH_SIZE,
    INPUT_CHANNELS,
)


class IHandler(RPCInterface):
    @exposed
    def set_devices(self, device_names: Sequence[str]):
        raise NotImplementedError()

    @exposed
    def active_children(self):
        raise NotImplementedError()

    @exposed
    def shutdown(self):
        raise NotImplementedError()

    @exposed
    def forward(self, data: TikTensor) -> RPCFuture[TikTensor]:
        raise NotImplementedError


def run(
    conn: Connection,
    config: dict,
    model_file: bytes,
    model_state: bytes,
    optimizer_state: bytes,
    log_queue: Optional[mp.Queue] = None,
):
    log.configure(log_queue)
    handler = HandlerProcess(config, model_file, model_state, optimizer_state, log_queue)
    srv = MPServer(handler, conn)
    srv.listen()


#                           - InferenceProcess
# server - HandlerProcess*-|
#                           - TrainingProcess
class HandlerProcess(IHandler):
    """
    Process to orchestrate the interplay of training/validation and inference
    """

    def __init__(
        self,
        config: dict,
        model_file: bytes,
        model_state: bytes,
        optimizer_state: bytes,
        log_queue: Optional[mp.Queue] = None,
    ) -> None:
        """
        :param config: configuration dict
        :param model_file: bytes of file describing the neural network model
        :param model_state: binarized model state dict
        :param optimizer_state: binarized optimizer state dict
        """
        assert model_file
        for required in ["model_class_name"]:
            if required not in config:
                raise ValueError(f"{required} missing in config")

        self.config = config

        self.shutdown_event = threading.Event()

        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting")
        self.valid_shapes: Optional[Union[List[Point2D], List[Point3D], List[Point4D]]] = None
        self.shrinkage: Optional[Union[Point2D, Point3D, Point4D]] = None
        self.idle_devices: List[torch.device] = []
        self.training_devices: List[torch.device] = []
        self.inference_devices: List[torch.device] = []

        self.tempdir = tempfile.mkdtemp()
        user_module_name = "usermodel"
        with open(os.path.join(self.tempdir, user_module_name + ".py"), "w") as f:
            f.write(pickle.loads(model_file))

        sys.path.insert(0, self.tempdir)
        user_module = importlib.import_module(user_module_name)

        self.model: torch.nn.Module = getattr(user_module, self.config["model_class_name"])(
            **self.config.get("model_init_kwargs", {})
        )
        self.logger.debug("created user model")

        if model_state:
            self.model.load_state_dict(torch.load(pickle.loads(model_state)))
            self.logger.info("restored model state")

        # start training process
        handler2training_conn, training2handler_conn = mp.Pipe()
        mp.Process(
            target=run_training,
            kwargs={
                "conn": training2handler_conn,
                "config": config,
                "model": self.model,
                "optimizer_state": optimizer_state,
                "log_queue": log_queue,
            },
        ).start()
        self._training: MPClient = MPClient(ITraining, handler2training_conn)

        # start inference process
        handler2inference_conn, inference2handler_conn = mp.Pipe()
        mp.Process(
            target=run_inference,
            kwargs={"conn": inference2handler_conn, "config": config, "model": self.model, "log_queue": log_queue},
        ).start()
        self._inference: MPClient = MPClient(IInference, handler2inference_conn)

        # start dryrun process
        handler2dryrun_conn, dryrun2handler_conn = mp.Pipe()
        mp.Process(
            target=run_dryrun,
            kwargs={"conn": dryrun2handler_conn, "config": config, "model": self.model, "log_queue": log_queue},
        ).start()
        self._dry_run = MPClient(IDryRun, handler2dryrun_conn)

        # start device setter thread that will wait for dry run processes to finish
        self.new_device_names: queue.Queue = queue.Queue()
        self.device_setter_thread = threading.Thread(target=self._device_setter_worker)
        self.device_setter_thread.start()

        self.set_devices(device_names=self.config.get("devices", ["cpu"]))

    def _device_setter_worker(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                new_device_names, fut = self.new_device_names.get(timeout=5)
            except TimeoutError:
                pass
            else:
                while not self.new_device_names.empty():
                    fut.cancel()
                    new_device_names, fut = self.new_device_names.get()

                new_devices = []
                for dn in new_device_names:
                    try:
                        torch_device = torch.device(dn)
                    except TypeError as e:
                        self.logger.error(e)
                    else:
                        new_devices.append(torch_device)

                # remove old devices that are not in the new list of devices
                self.idle_devices = [d for d in self.idle_devices if d in new_devices]

                if self.training.get_idle():
                    previous_training_devices = self.training.set_devices([])
                    assert previous_training_devices == self.training_devices, (previous_training_devices, self.training_devices)
                    self.idle_devices = self.training_devices + self.idle_devices
                else:
                    self.training_devices = [d for d in self.training_devices if d in new_devices]

                self.training.set_devices(self.training_devices)

                self.inference_devices = [d for d in self.inference_devices if d in new_devices]
                self.inference.set_devices(self.inference_devices)

                # do dry run for truly new devices
                new_devices = [d for d in new_devices if d not in self.devices]
                try:
                    approved_devices, training_shape, valid_shapes, shrinkage = self.dry_run.dry_run(
                        new_devices,
                        training_shape=self.config.get(TRAINING_SHAPE, None),
                        valid_shapes=self.valid_shapes,
                        shrinkage=self.shrinkage,
                    ).result()
                    if TRAINING_SHAPE in self.config:
                        assert training_shape == self.config[TRAINING_SHAPE]
                    else:
                        self.config[TRAINING_SHAPE] = training_shape

                    if self.valid_shapes is None:
                        self.valid_shapes = valid_shapes
                    else:
                        self.valid_shapes = [v for v in self.valid_shapes if v in valid_shapes]
                        assert self.valid_shapes, "No overlapping valid shapes found!"

                    if self.shrinkage is None:
                        self.shrinkage = shrinkage
                    else:
                        assert self.shrinkage == shrinkage

                except Exception as e:
                    fut.set_exeption(e)
                    self.logger.error(e)
                else:
                    self.idle_devices += approved_devices
                    fut.set_result((self.config[TRAINING_SHAPE], self.valid_shapes, self.shrinkage))

    # device handling and dry run
    @property
    def devices(self):
        return self.training_devices + self.idle_devices + self.inference_devices

    @property
    def inference(self):
        return self._inference

    @property
    def training(self):
        return self._training

    @property
    def dry_run(self):
        return self._dry_run

    def set_devices(
        self, device_names: Sequence[str]
    ) -> RPCFuture[
        Union[
            Tuple[Point2D, List[Point2D], Point2D],
            Tuple[Point3D, List[Point3D], Point3D],
            Tuple[Point4D, List[Point4D], Point4D],
        ]
    ]:
        fut = RPCFuture()
        self.new_device_names.put((device_names, fut))
        return fut

    # general
    def active_children(self) -> List[mp.Process]:
        return mp.active_children()

    def shutdown(self) -> None:
        self.logger.debug("Shutting down...")
        self.shutdown_event.set()
        self.dry_run.shutdown()
        self.inference.shutdown()
        self.training.shutdown()
        try:
            if self.tempdir:
                shutil.rmtree(self.tempdir)
        except Exception as e:
            self.logger.error(e)

        self.logger.debug("Shutdown complete")

    def report_idle(self, proc_name: str, devices: Sequence = tuple()) -> None:
        """
        report idle devices. Note that the process might not be fully idle.
        :param proc_name: child process name
        :param devices: devices that are idle (given back to the handler)
        """
        self.logger.debug("%s reported being idle", proc_name)
        if proc_name == TrainingProcess.name:
            assert all([d in self.training_devices for d in devices])
            self.training_devices = [d for d in self.training_devices if d not in devices]
            self.idle_devices += devices
            if self.inference_devices == REQUEST_FOR_DEVICES and not self.waiting_for_dry_run:
                self.assign_inference_devices()
        elif proc_name == InferenceProcess.name:
            assert all([d in self.inference_devices for d in devices])
            self.inference_devices = [d for d in self.inference_devices if d not in devices]
            self.idle_devices += devices
            if self.training_devices == REQUEST_FOR_DEVICES and not self.waiting_for_dry_run:
                self.assign_training_devices()
        elif proc_name == DryRunProcess.name:
            self.idle_devices += devices
            if self.inference_devices == REQUEST_FOR_DEVICES:
                self.assign_inference_devices()

            if self.training_devices == REQUEST_FOR_DEVICES:
                self.assign_training_devices()
        else:
            raise NotImplementedError(proc_name)

        # # todo: remove this idle report to server, only for debugging
        # self.server_conn.send(("report_idle", {"proc_name": proc_name, "devices": devices}))

    # inference
    def forward(self, data: TikTensor) -> RPCFuture[TikTensor]:
        # todo: update inference devices
        return self.inference.forward(data)

    # training
    def update_hparams(self, hparams: dict) -> None:
        # todo: check valid shapes if mini batch size changes
        pass

    def resume_training(self) -> None:
        self.training.resume_training()

    def pause_training(self) -> None:
        self.training.pause_training()

    def update_training_dataset(self, keys: Iterable, data: torch.Tensor) -> None:
        self.training_conn.send(("update_dataset", {"name": TRAINING, "keys": keys, "data": data}))

    def request_state(self) -> None:
        model_state = pickle.dumps(self.model.state_dict())
        optimizer_state = pickle.dumps(self.model.optimizer.state_dict())
        current_config = pickle.dumps(self.config)
        self.server_conn.send(
            (
                "request_state_answer",
                {"model_state": model_state, "optimizer_state": optimizer_state, "config": current_config},
            )
        )

    # validation
    def update_validation_dataset(self, keys: Iterable, data: torch.Tensor) -> None:
        self.training_conn.send(("update_dataset", {"name": VALIDATION, "keys": keys, "data": data}))

    # def validate(self):
    #     pass
