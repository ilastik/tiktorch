import importlib
import io
import logging
import logging.config
import os.path

import shutil
import sys
import tempfile
import threading
import torch
import queue

from multiprocessing.connection import Connection, wait
from torch import multiprocessing as mp
from concurrent.futures import wait as wait_for_futures

from typing import Any, List, Generic, Iterator, Iterable, Sequence, Callable, Dict, Optional, Tuple, Set, Union

from tiktorch.rpc import RPCInterface, exposed, RPCFuture
from tiktorch.rpc.mp import MPServer, MPClient, create_client, Shutdown
from tiktorch.tiktypes import (
    TikTensor,
    LabeledTikTensor,
    TikTensorBatch,
    LabeledTikTensorBatch,
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
from tiktorch.utils import add_logger, get_error_msg_for_invalid_config, get_error_msg_for_incomplete_config

from tiktorch.configkeys import (
    NAME,
    TORCH_VERSION,
    TRAINING,
    VALIDATION,
    BATCH_SIZE,
    TRAINING_SHAPE,
    TRAINING_SHAPE_LOWER_BOUND,
    TRAINING_SHAPE_UPPER_BOUND,
    NUM_ITERATION_DONE,
    MAX_NUM_ITERATIONS,
    MAX_NUM_ITERATIONS_PER_UPDATE,
    LOSS_CRITERION_CONFIG,
    OPTIMIZER_CONFIG,
)


class IHandler(RPCInterface):
    @exposed
    def set_devices(
        self, device_names: Sequence[str]
    ) -> RPCFuture[
        Union[
            Tuple[Point2D, List[Point2D], Point2D],
            Tuple[Point3D, List[Point3D], Point3D],
            Tuple[Point4D, List[Point4D], Point4D],
        ]
    ]:
        raise NotImplementedError

    @exposed
    def active_children(self) -> List[str]:
        raise NotImplementedError

    @exposed
    def shutdown(self) -> Shutdown:
        raise NotImplementedError

    @exposed
    def update_config(self, partial_config: dict) -> RPCFuture[bool]:
        raise NotImplementedError

    # Inference
    @exposed
    def forward(self, data: TikTensor) -> RPCFuture[TikTensor]:
        raise NotImplementedError

    # Training
    @exposed
    def resume_training(self) -> None:
        raise NotImplementedError

    @exposed
    def pause_training(self) -> None:
        raise NotImplementedError

    @exposed
    def update_training_data(self, data: LabeledTikTensorBatch) -> None:
        raise NotImplementedError

    @exposed
    def update_validation_data(self, data: LabeledTikTensorBatch) -> None:
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
        self.idle_devices: List[torch.device] = []
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
            self.logger.debug("load model state")
            try:
                self.model.load_state_dict(torch.load(io.BytesIO(model_state)))
            except Exception as e:
                self.logger.exception(e)
            else:
                self.logger.info("restored model state")

        try:
            self.logger.debug("start training process")
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

            self.logger.debug("start inference process")
            handler2inference_conn, inference2handler_conn = mp.Pipe()
            mp.Process(
                target=run_inference,
                name="Inference",
                kwargs={"conn": inference2handler_conn, "config": config, "model": self.model, "log_queue": log_queue},
            ).start()
            self._inference: IInference = create_client(IInference, handler2inference_conn)

            self.logger.debug("start dryrun process")
            handler2dryrun_conn, dryrun2handler_conn = mp.Pipe()
            mp.Process(
                name="DryRun",
                target=run_dryrun,
                kwargs={"conn": dryrun2handler_conn, "config": config, "model": self.model, "log_queue": log_queue},
            ).start()
            self._dry_run: IDryRun = create_client(IDryRun, handler2dryrun_conn)

            # start device setter thread that will wait for dry run processes to finish
            self.new_device_names: queue.Queue = queue.Queue()
            self.device_setter_thread = threading.Thread(
                target=add_logger(self.logger)(self._device_setter_worker), name="DeviceSetter"
            )
            self.device_setter_thread.start()
        except Exception as e:
            self.logger.exception(e)
            self.shutdown()

    def _device_setter_worker(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                new_devices_entry = self.new_device_names.get(timeout=60)
            except queue.Empty:
                new_devices_entry = "time_to_check_if_someone_is_idle_dont_you_think?"

            self.training_idle = self.training.get_idle()
            self.inference_idle = self.inference.get_idle()
            self.logger.debug("got new devices entry: %s", new_devices_entry)
            if new_devices_entry is None:  # shutdown event
                return
            elif isinstance(new_devices_entry, tuple):
                new_device_names, fut = new_devices_entry
            else:  # idle changed signal
                # no new devices in a while; reassign devices if necessary
                for fut in self._collect_idle_devices():
                    # todo: ret fut fut.result()  # wait for confirmation that idle devices are in fact free
                    pass  # wait for confirmation that idle devices are in fact free

                if not self.idle_devices and not self.inference_devices and not self.inference.get_idle():
                    self.logger.debug("reassigning a training device to inference")
                    self.inference_devices = [self.training_devices[-1]]
                    self.training_devices = self.training_devices[:-1]
                    self.training.set_devices(self.training_devices)  # todo: change to futures
                    self.inference.set_devices(self.inference_devices).result(timeout=20)
                else:
                    self._assign_idle_devices()

                continue

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
                self.logger.debug("Requesting dry run for new devices: %s", new_devices)
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
                        # todo: make sure this happens inside the dry runa dn these new devcies aren't added at all
                        raise ValueError(f"No valid shapes found after adding devices: {new_devices}")

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
            fut.set_result((self.config.get(TRAINING_SHAPE, None), self.valid_shapes, self.shrinkage))

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
        freed_inference_devices_fut = self.inference.set_devices(self.inference_devices).result(timeout=20)
        return freed_training_devices_fut, freed_inference_devices_fut

    def _assign_idle_devices(self):
        if not self.idle_devices:
            return

        training_idle = self.training.get_idle()
        inference_idle = self.inference.get_idle()
        self.logger.debug(
            "assigning idle devices: %s (training idle: %r, inference idle: %r)",
            self.idle_devices,
            training_idle,
            inference_idle,
        )
        training_devices_changed = False
        inference_devices_changed = False
        if training_idle and inference_idle:
            return
        elif training_idle and not inference_idle:  # all for inference
            self.inference_devices = self.idle_devices + self.inference_devices
            inference_devices_changed = True
        elif not training_idle and inference_idle:  # all for training
            self.training_devices += self.idle_devices
            training_devices_changed = True
        elif not self.inference_devices:  # one device for inference, rest (if exists) for training
            self.inference_devices.insert(0, self.idle_devices[-1])
            inference_devices_changed = True
            if len(self.idle_devices) > 1:
                self.training_devices += self.idle_devices[:-1]
                training_devices_changed = True
        else:  # inference has at least one device already, assign the rest to training
            self.training_devices += self.idle_devices
            training_devices_changed = True

        self.idle_devices = []
        if training_devices_changed:
            self.logger.debug("assign new training devices: %s", self.training_devices)
            self.training.set_devices(self.training_devices)

        if inference_devices_changed:
            self.logger.debug("assign new inference devices: %s", self.inference_devices)
            self.inference.set_devices(self.inference_devices).result(timeout=20)

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

    def shutdown(self) -> Shutdown:
        self.logger.debug("Shutting down...")
        self.shutdown_event.set()
        # wait for threads to shutdown
        try:
            self.new_device_names.put(None)
            self.device_setter_thread.join(timeout=30)
        except TimeoutError as e:
            self.logger.error(e)

        timeout = 20
        # shutdown processes
        try:
            self.dry_run.shutdown.async_().result(timeout=timeout)
        except TimeoutError as e:
            self.logger.error(e)
        try:
            self.inference.shutdown.async_().result(timeout=timeout)
        except TimeoutError as e:
            self.logger.error(e)
        try:
            self.training.shutdown.async_().result(timeout=timeout)
        except TimeoutError as e:
            self.logger.error(e)

        try:
            if self.tempdir:
                shutil.rmtree(self.tempdir)
        except Exception as e:
            self.logger.error(e)

        self.logger.debug("Shutdown complete")
        return Shutdown()

    def update_config(self, partial_config: dict) -> None:
        # todo: check valid shapes if mini batch size changes
        error_msg = get_error_msg_for_invalid_config(partial_config)
        if error_msg:
            raise ValueError(error_msg)

        previous_config = {key: value for key, value in self.config.items() if key in partial_config}
        need_dry_run = False
        # todo: update inference, if needed
        for key, value in partial_config.items():
            if key in [TRAINING, VALIDATION]:
                for subkey, subvalue in partial_config[key].items():
                    if subvalue is None:
                        # remove key from config
                        if subkey == TRAINING_SHAPE:
                            need_dry_run = True
                        else:
                            raise NotImplementedError(f"How to delete {subkey} from {key}?")

                        if subkey in self.config[key]:
                            del self.config[key][subkey]
                    elif subkey in [BATCH_SIZE]:
                        self.config[key][subkey] = subvalue
            elif key in [NAME, TORCH_VERSION]:
                self.config[key] = value
            else:
                raise NotImplementedError(f"How to set {key} as a hyper parameter?")

        self.dry_run.update_config(partial_config)
        if need_dry_run:
            self.logger.info("Executing new dry run after config update...")
            current_devices = list(self.devices)
            self._idle_devices = []
            self.training.set_devices([])
            self.inference.set_devices([])
            self.set_devices(["cpu" if d.type == "cpu" else f"{d.type}:{d.index}" for d in current_devices])

        self.training.update_config(partial_config)
        # todo: reset config to previous on failure
        incomplete_msg = get_error_msg_for_incomplete_config(self.config)
        if incomplete_msg:
            raise ValueError(incomplete_msg)

    # inference
    def forward(self, data: TikTensor) -> RPCFuture[TikTensor]:
        self.new_device_names.put("whatever_just_update_idle_because_this_is_not_a_tuple_nor_None")
        self.logger.debug("forward")
        return self.inference.forward(data)

    # training
    def resume_training(self) -> None:
        self.training.resume_training()

    def pause_training(self) -> None:
        self.training.pause_training()

    def update_training_dataset(self, data: LabeledTikTensorBatch) -> None:
        self.training.update_dataset(TRAINING, data)

    def update_validation_dataset(self, data: LabeledTikTensorBatch) -> None:
        self.training.update_dataset(VALIDATION, data)

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
