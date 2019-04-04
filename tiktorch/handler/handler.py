import importlib
import io
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
from tiktorch.rpc.mp import MPServer, MPClient, create_client, Shutdown
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
from tiktorch.configkeys import TRAINING, VALIDATION, MODEL_CLASS_NAME, MODEL_INIT_KWARGS
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
        raise NotImplementedError

    @exposed
    def active_children(self) -> List[str]:
        raise NotImplementedError

    @exposed
    def shutdown(self) -> RPCFuture[Shutdown]:
        raise NotImplementedError

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
        for required in [MODEL_CLASS_NAME]:
            if required not in config:
                raise ValueError(f"{required} missing in config")

        self.config = config

        self.shutdown_event = threading.Event()

        self.logger = logging.getLogger(__name__)
        self.logger.info("started")
        self.valid_shapes: Optional[Union[List[Point2D], List[Point3D], List[Point4D]]] = None
        self.shrinkage: Optional[Union[Point2D, Point3D, Point4D]] = None
        self.idle_devices: List[torch.device] = [torch.device("cpu")]
        self.training_devices: List[torch.device] = []
        self.inference_devices: List[torch.device] = []

        self.tempdir = tempfile.mkdtemp()
        user_module_name = "usermodel"
        with open(os.path.join(self.tempdir, user_module_name + ".py"), "wb") as f:
            f.write(model_file)

        sys.path.insert(0, self.tempdir)
        user_module = importlib.import_module(user_module_name)

        self.model: torch.nn.Module = getattr(user_module, self.config[MODEL_CLASS_NAME])(
            **self.config.get(MODEL_INIT_KWARGS, {})
        )
        self.logger.debug("created user model")

        if model_state:
            self.model.load_state_dict(torch.load(io.BytesIO(model_state)))
            self.logger.info("restored model state")

        # start training process
        handler2training_conn, training2handler_conn = mp.Pipe()
        mp.Process(
            target=run_training,
            name="Training",
            kwargs={
                "conn": training2handler_conn,
                "config": config,
                "model": self.model,
                "optimizer_state": optimizer_state,
                "log_queue": log_queue,
            },
        ).start()
        self._training: ITraining = create_client(ITraining, handler2training_conn)

        # start inference process
        handler2inference_conn, inference2handler_conn = mp.Pipe()
        mp.Process(
            target=run_inference,
            name="Inference",
            kwargs={"conn": inference2handler_conn, "config": config, "model": self.model, "log_queue": log_queue},
        ).start()
        self._inference: IInference = create_client(IInference, handler2inference_conn)

        # start dryrun process
        handler2dryrun_conn, dryrun2handler_conn = mp.Pipe()
        mp.Process(
            name="DryRun",
            target=run_dryrun,
            kwargs={"conn": dryrun2handler_conn, "config": config, "model": self.model, "log_queue": log_queue},
        ).start()
        self._dry_run: IDryRun = create_client(IDryRun, handler2dryrun_conn)

        # start device setter thread that will wait for dry run processes to finish
        self.new_device_names: queue.Queue = queue.Queue()
        self.device_setter_thread = threading.Thread(target=self._device_setter_worker, name="DeviceSetter")
        self.device_setter_thread.start()

        self.set_devices(device_names=self.config.get("devices", ["cpu"]))

    def _device_setter_worker(self) -> None:
        self.logger.info("started")
        while not self.shutdown_event.is_set():
            try:
                new_device_names, fut = self.new_device_names.get(timeout=3)
            except queue.Empty:
                # no new devices; reassign devices if necessary
                for fut in self._collect_idle_devices():
                    # todo: ret fut fut.result()  # wait for confirmation that idle devices are in fact free
                    pass  # wait for confirmation that idle devices are in fact free

                if not self.idle_devices and not self.inference_devices and not self.inference.get_idle():
                    self.logger.debug("reassigning a training device to inference")
                    self.inference_devices = [self.training_devices[-1]]
                    self.training_devices = self.training_devices[:-1]
                    self.training.set_devices(self.training_devices) # todo: change to futures
                    self.inference.set_devices(self.inference_devices)  # no need to wait here
                else:
                    self._assign_idle_devices()

                continue
            try:
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

                freed_training_devices_fut, freed_inference_devices_fut = self._collect_idle_devices(
                    new_devices=new_devices
                )

                # do dry run for truly new devices
                new_devices = [d for d in new_devices if d not in self.devices]
                if new_devices:
                    approved_devices, training_shape, valid_shapes, shrinkage = self.dry_run.dry_run(
                        new_devices,
                        training_shape=self.config.get(TRAINING_SHAPE, None),
                        valid_shapes=self.valid_shapes,
                        shrinkage=self.shrinkage,
                    ).result()
                    self.idle_devices += approved_devices

                    if TRAINING_SHAPE in self.config:
                        assert training_shape == self.config[TRAINING_SHAPE]
                    else:
                        self.config[TRAINING_SHAPE] = training_shape

                    if self.valid_shapes is None:
                        self.valid_shapes = valid_shapes
                    else:
                        self.valid_shapes = [v for v in self.valid_shapes if v in valid_shapes]
                        if not self.valid_shapes:
                            raise ValueError(f"No valid shapes found after {new_devices}")

                    if self.shrinkage is None:
                        self.shrinkage = shrinkage
                    else:
                        assert self.shrinkage == shrinkage

                # wait for old devices to be free
                # todo: wait for old devices to be free (when they are returned as futures)
                # freed_training_devices_fut.result()
                # freed_inference_devices_fut.result()

                # (re-)assign freed old and new devices
                self._assign_idle_devices()
            except Exception as e:
                fut.set_exception(e)
                self.logger.error(e)
            else:
                fut.set_result((self.config.get(TRAINING_SHAPE, None), self.valid_shapes, self.shrinkage))

        self.logger.info("stopped")

    def _collect_idle_devices(self, new_devices: Optional[Sequence[torch.device]] = None):
        if self.training.get_idle():
            self.idle_devices = self.training_devices + self.idle_devices
            self.training_devices = []
        elif new_devices is not None:
            self.training_devices = [d for d in self.training_devices if d in new_devices]

        if self.inference.get_idle():
            self.idle_devices += self.inference_devices
            self.inference_devices = []
        elif new_devices is not None:
            self.inference_devices = [d for d in self.inference_devices if d in new_devices]

        freed_training_devices_fut = self.training.set_devices(self.training_devices)
        freed_inference_devices_fut = self.inference.set_devices(self.inference_devices)
        return freed_training_devices_fut, freed_inference_devices_fut

    def _assign_idle_devices(self):
        if not self.idle_devices:
            return

        self.logger.debug("assigning idle devices")
        training_idle = self.training.get_idle()
        inference_idle = self.inference.get_idle()
        if training_idle and inference_idle:
            return
        elif training_idle and not inference_idle:  # all for inference
            self.inference_devices = self.idle_devices + self.inference_devices
        elif not training_idle and inference_idle:  # all for training
            self.training_devices += self.idle_devices
        elif not self.inference_devices:  # one device for inference, rest (if exists) for training
            self.inference_devices.insert(0, self.idle_devices[-1])
            self.training_devices += self.idle_devices[:-1]
        else:  # inference has at least one device already, assign the rest to training
            self.training_devices += self.idle_devices

        self.idle_devices = []
        self.training.set_devices(self.training_devices)
        self.inference.set_devices(self.inference_devices)

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
    def active_children(self) -> List[str]:
        return [c.name for c in mp.active_children()]

    def shutdown(self) -> None:
        self.logger.debug("Shutting down...")
        self.shutdown_event.set()
        # wait for threads to shutdown
        try:
            self.device_setter_thread.join(timeout=30)
        except TimeoutError as e:
            self.logger.error(e)

        # shutdown processes
        try:
            self.dry_run.shutdown()
        except TimeoutError as e:
            self.logger.error(e)
        try:
            self.inference.shutdown()
        except TimeoutError as e:
            self.logger.error(e)
        try:
            self.training.shutdown()
        except TimeoutError as e:
            self.logger.error(e)

        try:
            if self.tempdir:
                shutil.rmtree(self.tempdir)
        except Exception as e:
            self.logger.error(e)

        self.logger.debug("Shutdown complete")
        raise Shutdown

    def update_hparams(self, hparams: dict) -> None:
        # todo: check valid shapes if mini batch size changes
        pass

    # inference
    def forward(self, data: TikTensor) -> RPCFuture[TikTensor]:
        # todo: update inference devices
        return self.inference.forward(data)

    # training
    def resume_training(self) -> None:
        self.training.resume_training()

    def pause_training(self) -> None:
        self.training.pause_training()

    # def update_training_dataset(self, keys: Iterable, data: torch.Tensor) -> None:
    #     self.training_conn.send(("update_dataset", {"name": TRAINING, "keys": keys, "data": data}))
    #
    # def request_state(self) -> None:
    #     model_state = pickle.dumps(self.model.state_dict())
    #     optimizer_state = pickle.dumps(self.model.optimizer.state_dict())
    #     current_config = pickle.dumps(self.config)
    #     self.server_conn.send(
    #         (
    #             "request_state_answer",
    #             {"model_state": model_state, "optimizer_state": optimizer_state, "config": current_config},
    #         )
    #     )
    #
    # # validation
    # def update_validation_dataset(self, keys: Iterable, data: torch.Tensor) -> None:
    #     self.training_conn.send(("update_dataset", {"name": VALIDATION, "keys": keys, "data": data}))

    # def validate(self):
    #     pass
