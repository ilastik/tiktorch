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
from torch.multiprocessing import Process, Pipe, active_children, Queue

from typing import Any, List, Generic, Iterator, Iterable, Sequence, Callable, Dict, Optional, Tuple

from tiktorch.rpc import RPCInterface, exposed
from tiktorch.rpc.mp import MPServer
from tiktorch.tiktypes import TikTensor, TikTensorBatch, PointBase, Point2D, Point3D, Point4D, BatchPointBase, PointAndBatchPointBase
from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION, TRAINING, VALIDATION, REQUEST_FOR_DEVICES
from .dryrun import DryRunProcess
from .training import TrainingProcess
from .inference import InferenceProcess


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
        self.model_file = model_file  # to be deserialized in 'load_model'
        self.model_state = model_state  # to be deserialized in 'load_model'
        self.optimizer_state = optimizer_state  # to be deserialized in training process

        self._shutting_down: bool = False
        self.logger = logging.getLogger(self.name)
        self.logger.info("Starting")
        self.valid_shapes: list = []
        self.known_devices: dict = {}
        self.tempdir: str = ""
        self.idle_devices: List[torch.device] = []
        self.training_devices: List[torch.device] = []
        self.inference_devices: List[torch.device] = []
        self.waiting_for_dry_run = True
        self.dryrun_conn: Connection = None

        self.inference_conn, self.inference_proc = None, None
        self.training_conn, self.training_proc = None, None
        self.load_model(
            self.model_file,
            self.model_state,
            self.config["model_class_name"],
            self.config.get("model_init_kwargs", {}),
        )
        self.set_devices(device_names=self.config.get("devices", ["cpu"]))

    # device handling and dry run
    @property
    def devices(self):
        return self.idle_devices + self.training_devices + self.inference_devices

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

        if self.dryrun_conn is not None:
            self.dryrun_conn.send(SHUTDOWN)

        self.dryrun_conn, handler_conn = Pipe()
        self.dryrun = DryRunProcess(handler_conn=handler_conn, config=self.config, model=self.model, devices=devices)
        self.dryrun.start()

    def set_devices_answer(self, devices: Sequence[torch.device] = tuple(), valid_shapes: Sequence[PointBase] = tuple(), shrinkage: PointBase = None, failure_msg: str = "") -> None:
        if failure_msg:
            self.server_conn.send(("set_devices_answer", {"failure_msg": failure_msg}))
        else:
            self.waiting_for_dry_run = False
            self.server_conn.send(("set_devices_answer", {"device_names": [str(d) for d in devices]}))

    def load_model(self, model_file: bytes, model_state: bytes, model_class_name: str, model_init_kwargs: dict) -> None:
        if self.tempdir:
            # remove previous usermodule folder
            shutil.rmtree(self.tempdir)

        self.tempdir = tempfile.mkdtemp()
        user_module_name = "usermodel"
        with open(os.path.join(self.tempdir, user_module_name + ".py"), "w") as f:
            f.write(pickle.loads(model_file))

        sys.path.insert(0, self.tempdir)
        user_module = importlib.import_module(user_module_name)

        self.model: torch.nn.Module = getattr(user_module, model_class_name)(**model_init_kwargs)
        self.logger.debug("created user model")

        if model_state:
            self.model.load_state_dict(torch.load(pickle.loads(model_state)))
            # with closing(io.BytesIO(model_state)) as f:
            #     self.model.load_state_dict(torch.load(f))

            self.logger.info("restored model state")

        # (re-)start training and inference processes
        self.shutdown_children(
            conn_procs=[(self.training_conn, self.training_proc), (self.inference_conn, self.inference_proc)]
        )

        # set up training process
        self.training_conn, handler_conn_training = Pipe()
        self.config["training_devices"] = self.training_devices
        self.training_proc = TrainingProcess(
            handler_conn=handler_conn_training,
            config=self.config,
            model=self.model,
            optimizer_state=self.optimizer_state,
        )
        self.training_proc.start()
        # set up inference process
        self.config["inference_devices"] = self.inference_devices
        self.inference_conn, handler_conn_inference = Pipe()
        self.inference_proc = InferenceProcess(
            handler_conn=handler_conn_inference, config=self.config, model=self.model
        )
        self.inference_proc.start()      

    # general
    def active_children(self) -> None:
        self.server_conn.send([child_proc.name for child_proc in active_children()])

    def shutdown_children(self, conn_procs: Sequence[Tuple[Connection, Process]]) -> None:
        # initiate shutdown of children (to shut down in parallel)
        for conn, proc in conn_procs:
            if proc is None:
                continue

            try:
                conn.send(SHUTDOWN)
            except Exception as e:
                self.logger.error(e)

        # enforce shutdown of children
        shutdown_time = 20
        for conn, proc in conn_procs:
            if proc is None:
                continue

            while proc.is_alive():
                # look for shutting down answer
                if conn.poll(timeout=shutdown_time):
                    answer = conn.recv()
                    if answer == SHUTDOWN_ANSWER:
                        # give child process extra time to shutdown
                        proc.join(timeout=shutdown_time)
                else:
                    self.logger.error("Failed to shutdown %s gracefully. Sending kill...", proc.name)
                    proc.kill()
                    proc.join(timeout=shutdown_time)

            self.logger.debug("%s has shutdown", proc.name)

    def shutdown(self) -> None:
        logger = logging.getLogger(self.name + ".shutdown")
        logger.debug("Shutting down...")
        self._shutting_down = True
        self.server_conn.send(SHUTDOWN_ANSWER)
        try:
            self.shutdown_children(
                [(self.inference_conn, self.inference_proc), (self.training_conn, self.training_proc)]
            )
        except Exception as e:
            logger.error("Could not shut down children due to exception: %s", e)

        try:
            if self.tempdir:
                shutil.rmtree(self.tempdir)
        except Exception as e:
            logger.error(e)

        logger.debug("Shutdown complete")

    def shutting_down(self):
        self.logger.error("A child process is shutting down unscheduled.")

    def report_exception(self, proc_name: str, exception: Exception) -> None:
        self.logger.error("Received exception report from %s: %s", proc_name, exception)
        if proc_name == TrainingProcess.name:
            # todo: restart training proess
            pass
        elif proc_name == InferenceProcess.name:
            # todo: restart inference process
            pass
        else:
            raise NotImplementedError("Did not expect exception report form %s", proc_name)

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

    def generic_relay_to_server(self, method_name: str, **kwargs) -> None:
        if not self._shutting_down:
            self.server_conn.send((method_name, kwargs))

    def __getattr__(self, method_name) -> Callable:
        if method_name in ["forward_answer", "pause_training_answer"]:
            return partial(self.generic_relay_to_server, method_name=method_name)
        else:
            raise AttributeError(method_name)

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

        self.inference_conn.send(("update_devices", {"devices": self.inference_devices}))

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
