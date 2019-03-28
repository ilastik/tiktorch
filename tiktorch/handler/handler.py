import importlib
import logging
import logging.config
import os.path
import pickle
import shutil
import sys
import tempfile
import torch
import time
import numpy as np
import bisect
import queue

from functools import partial
from multiprocessing.connection import Connection, wait
from torch import multiprocessing as mp

from typing import Any, List, Generic, Iterator, Iterable, Sequence, Callable, Dict, Optional, Tuple, Set

from tiktorch.rpc import RPCInterface, exposed, RPCFuture
from tiktorch.rpc.mp import MPServer, MPClient
from tiktorch.tiktypes import TikTensor, TikTensorBatch, PointBase, Point2D, Point3D, Point4D, BatchPointBase, PointAndBatchPointBase
from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION, TRAINING, VALIDATION, REQUEST_FOR_DEVICES
from .dryrun import DryRunProcess
from tiktorch.handler.training import run as run_training, ITraining
from tiktorch.handler.inference import run as run_inference, IInference
from tiktorch.handler.dryrun import run as run_dryrun, IDryRun


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
    def forward(self, data: TikTensorBatch):
        raise NotImplementedError


def run(conn: Connection,  config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes):
    handler = HandlerProcess(config, model_file, model_state, optimizer_state)
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
        self, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes
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

        self.shutting_down: bool = False
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting")
        self.valid_shapes: list = []
        self.known_devices: dict = {}
        self.idle_devices: List[torch.device] = []
        self.training_devices: List[torch.device] = []
        self.inference_devices: List[torch.device] = []
        self.waiting_for_dry_run = True

        self.tempdir = tempfile.mkdtemp()
        user_module_name = "usermodel"
        with open(os.path.join(self.tempdir, user_module_name + ".py"), "w") as f:
            f.write(pickle.loads(model_file))

        sys.path.insert(0, self.tempdir)
        user_module = importlib.import_module(user_module_name)

        self.model: torch.nn.Module = getattr(user_module, self.config["model_class_name"])(**self.config.get("model_init_kwargs", {}))
        self.logger.debug("created user model")

        if model_state:
            self.model.load_state_dict(torch.load(pickle.loads(model_state)))
            self.logger.info("restored model state")

        # start training process
        handler2training_conn, training2handler_conn = mp.Pipe()
        p = mp.Process(
            target=run_training,
            kwargs={
                "conn": training2handler_conn,
                "config": config,
                "model": self.model,
                "optimizer_state": optimizer_state,
            },
        )
        p.start()
        self._training: MPClient = MPClient(ITraining, handler2training_conn)

        # start inference process
        handler2inference_conn, inference2handler_conn = mp.Pipe()
        p = mp.Process(
            target=run_inference,
            kwargs={
                "conn": inference2handler_conn,
                "config": config,
                "model": self.model,
                "optimizer_state": optimizer_state,
            },
        )
        p.start()
        self._inference: MPClient = MPClient(IInference, handler2inference_conn)

        # start dryun pocess
        handler2dryrun_conn, dryrun2handler_conn = mp.Pipe()
        p = mp.Process(
            target=run_dryrun,
            kwargs={
                "conn": dryrun2handler_conn,
                "config": config,
                "model": self.model,
            },
        )
        p.start()
        self._dry_run: MPClient = MPClient(IDryRun, handler2dryrun_conn)

        self.set_devices(device_names=self.config.get("devices", ["cpu"]))

    # device handling and dry run
    @property
    def devices(self):
        return self.idle_devices + self.training_devices + self.inference_devices

    @property
    def inference(self):
        return self._inference

    @property
    def training(self):
        return self._training

    @property
    def dry_run(self):
        return self._dry_run

    def set_devices(self, device_names: Sequence[str]) -> None:
        self.waiting_for_dry_run = True
        devices = []
        for device in device_names:
            try:
                torch_device = torch.device(device)
            except TypeError as e:
                self.logger.warning(e)
            else:
                devices.append(torch_device)

        # todo: compare to previous devices
        #   todo: free old devices, reset idle_devices, training_devices, inference_devices
        previous_devices = list(self.devices)
        if previous_devices:
            pass

        # for now hard reset all previous devices
        self.idle_devices: Sequence[torch.device] = []
        self.training_devices: Sequence[torch.device] = []
        self.inference_devices: Sequence[torch.device] = []

        for d in devices:
            self.dry_run.dry_run(d)

    def set_devices_answer(self, devices: Sequence[torch.device] = tuple(), valid_shapes: Sequence[PointBase] = tuple(), shrinkage: PointBase = None, failure_msg: str = "") -> None:
        if failure_msg:
            self.server_conn.send(("set_devices_answer", {"failure_msg": failure_msg}))
        else:
            self.waiting_for_dry_run = False
            self.server_conn.send(("set_devices_answer", {"device_names": [str(d) for d in devices]}))

    # general
    def active_children(self) -> List[mp.Process]:
        return mp.active_children()

    def shutdown(self) -> None:
        self.logger.debug("Shutting down...")
        self._shutting_down = True
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
    def assign_inference_devices(self) -> None:
        if not self.inference_devices and not self.idle_devices:
            # training is currently using all devices
            raise NotImplementedError(
                "cpu should always be available for forward pass! Freeing a training gpu to be implemented"
            )

        idle_gpus = [d for d in self.idle_devices if d.type != "cpu"]
        if idle_gpus:
            self.inference_devices += idle_gpus
            self.idle_devices = [d for d in self.idle_devices if d.type == "cpu"]
        else:
            self.inference_devices += self.idle_devices
            self.idle_devices = []

        self.inference.send(("update_devices", {"devices": self.inference_devices}))

    def forward(self, data: TikTensorBatch) -> None:
        keys: List = [d.id for d in data]
        data: List[torch.Tensor] = data.as_torch()
        self.logger.debug("forward")
        self.logger.debug("inference id %d", id(self.model))
        # todo: update inference devices
        self.inference_conn.send(("forward", {"keys": keys, "data": data}))

    # training
    def assign_training_devices(self) -> None:
        if len(self.devices) == 1 and self.devices[0] == "cpu":
            # todo: remove training on cpu (only useful for debugging)
            self.training_conn.send(("update_devices", {"devices": ["cpu"]}))
        else:
            idle_gpus = [d for d in self.idle_devices if d.type != "cpu"]
            if idle_gpus:
                self.training_devices = idle_gpus
                self.idle_devices = [d for d in self.idle_devices if d.type == "cpu"]
                self.training_conn.send(("update_devices", {"devices": idle_gpus}))
            else:
                self.training_devices = REQUEST_FOR_DEVICES

    def update_hparams(self, hparams: dict) -> None:
        # todo: check valid shapes if mini batch size changes
        pass

    def resume_training(self) -> None:
        self.training_conn.send((TrainingProcess.resume_training.__name__, {}))

    def pause_training(self) -> None:
        self.training_conn.send((TrainingProcess.pause_training.__name__, {}))

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
