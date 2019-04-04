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
import numpy
import bisect
import queue
import threading

from concurrent.futures import ThreadPoolExecutor, Future
from multiprocessing.connection import Connection
from torch import multiprocessing as mp

from torch.multiprocessing import Process, Pipe, Queue
from tiktorch import log

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
    Union,
    NamedTuple,
)

from tiktorch.rpc import RPCInterface, exposed, Shutdown, RPCFuture
from tiktorch.rpc.mp import MPServer
from tiktorch.tiktypes import (
    TikTensor,
    TikTensorBatch,
    PointAndBatchPointBase,
    PointBase,
    Point2D,
    Point3D,
    Point4D,
    BatchPointBase,
    BatchPoint2D,
    BatchPoint3D,
    BatchPoint4D,
)

from tiktorch.configkeys import (
    TRAINING_SHAPE,
    TRAINING_SHAPE_LOWER_BOUND,
    TRAINING_SHAPE_UPPER_BOUND,
    BATCH_SIZE,
    INPUT_CHANNELS,
)


def ret_through_conn(*args, fn: Callable, send_conn: Connection, **kwargs) -> None:
    send_conn.send(fn(*args, **kwargs))


def in_subproc(fn: Callable, *args, **kwargs) -> Connection:
    """
    Run 'fn' in a subprocess and return a connection that will hold the result
    :param fn: function to run in subprocess
    :return: Connection to result
    """
    recv_conn, send_conn = Pipe(duplex=False)
    subproc = Process(
        target=ret_through_conn, name=fn.__name__, args=args, kwargs={"fn": fn, "send_conn": send_conn, **kwargs}
    )
    subproc.start()
    return recv_conn


class IDryRun(RPCInterface):
    @exposed
    def dry_run(
        self,
        devices: Sequence[torch.device],
        training_shape: Optional[Union[Point2D, Point3D, Point4D]] = None,
        valid_shapes: Optional[List[Union[Point2D, Point3D, Point4D]]] = None,
        shrinkage: Optional[Union[Point2D, Point3D, Point4D]] = None,
    ) -> RPCFuture:
        raise NotImplementedError()

    @exposed
    def shutdown(self) -> Shutdown:
        raise NotImplementedError()


def run(conn: Connection, config: dict, model: torch.nn.Module, log_queue: Optional[mp.Queue] = None):
    log.configure(log_queue)
    # print('CUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])
    dryrun_proc = DryRunProcess(config, model)
    srv = MPServer(dryrun_proc, conn)
    srv.listen()


class DryRunProcess(IDryRun):
    """
    Process to execute a dry run to determine training and inference shape for 'model' on 'device'
    """

    def __init__(self, config: dict, model: torch.nn.Module) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.info("started")
        self.config = config
        self.model = model

        self.shutdown_event = threading.Event()

        self.training_shape = None
        self.valid_shapes = None
        self.shrinkage: Optional[Union[Point2D, Point3D, Point4D]] = None

        self.dry_run_queue = queue.Queue()
        self.dry_run_thread = threading.Thread(target=self._dry_run_worker, name="DryRun")
        self.dry_run_thread.start()

    def _dry_run_worker(self) -> None:
        self.logger.debug("started")
        while not self.shutdown_event.is_set():
            try:
                args = self.dry_run_queue.get(block=True, timeout=1)
            except queue.Empty:
                pass
            else:
                self._dry_run(*args)

        self.logger.debug("stopped")

    def dry_run(
        self,
        devices: Sequence[torch.device],
        training_shape: Optional[Union[Point2D, Point3D, Point4D]] = None,
        valid_shapes: Optional[List[Union[Point2D, Point3D, Point4D]]] = None,
        shrinkage: Optional[Union[Point2D, Point3D, Point4D]] = None,
    ) -> RPCFuture:
        fut = RPCFuture()
        self.dry_run_queue.put((devices, training_shape, valid_shapes, shrinkage, fut))
        return fut

    def _dry_run(
        self,
        devices: Sequence[torch.device],
        training_shape: Optional[Union[Point2D, Point3D, Point4D]],
        valid_shapes: Optional[List[Union[Point2D, Point3D, Point4D]]],
        shrinkage: Optional[Union[Point2D, Point3D, Point4D]],
        fut: Future,
    ) -> None:
        self.logger.info("Starting dry run for %s", devices)
        assert devices
        try:
            working_devices = self.minimal_device_test(devices)
            failed_devices = set(devices) - set(working_devices)
            if failed_devices:
                self.logger.error(f"Minimal device test failed for {failed_devices}")

            if self.training_shape is None:
                self.training_shape = self._determine_training_shape(training_shape=training_shape, devices=devices)
            else:
                assert self.training_shape == training_shape

            self._determine_valid_shapes(devices=devices, valid_shapes=valid_shapes)
            if shrinkage is not None:
                assert shrinkage == self.shrinkage

            fut.set_result((devices, self.training_shape, self.valid_shapes, self.shrinkage))
            self.logger.info("dry run done")
        except Exception as e:
            self.logger.error(e)
            fut.set_exception(e)

    def _determine_training_shape(
        self, devices: Sequence[torch.device], training_shape: Optional[Union[Point2D, Point3D, Point4D]] = None
    ):
        batch_size = self.config[BATCH_SIZE]
        input_channels = self.config[INPUT_CHANNELS]

        if TRAINING_SHAPE in self.config:
            # validate given training shape
            config_training_shape = PointBase.from_spacetime(input_channels, self.config[TRAINING_SHAPE])
            if training_shape is None:
                training_shape = config_training_shape
            else:
                assert training_shape == config_training_shape

            training_shape = training_shape.add_batch(batch_size)

            if TRAINING_SHAPE_UPPER_BOUND in self.config:
                training_shape_upper_bound = BatchPointBase.from_spacetime(
                    batch_size, input_channels, self.config[TRAINING_SHAPE_UPPER_BOUND]
                )
                if not (training_shape <= training_shape_upper_bound):
                    raise ValueError(
                        f"{TRAINING_SHAPE}: {training_shape} incompatible with {TRAINING_SHAPE_UPPER_BOUND}: {training_shape_upper_bound}"
                    )

            if TRAINING_SHAPE_LOWER_BOUND in self.config:
                training_shape_lower_bound = BatchPointBase.from_spacetime(
                    batch_size, input_channels, self.config[TRAINING_SHAPE_LOWER_BOUND]
                )
            else:
                training_shape_lower_bound = training_shape.__class__()

            if not (training_shape_lower_bound <= training_shape):
                raise ValueError(
                    f"{TRAINING_SHAPE_LOWER_BOUND}{training_shape_lower_bound} incompatible with {TRAINING_SHAPE}{training_shape}"
                )

            shrinkage = self.validate_shape(devices=devices, shape=training_shape, train_mode=True)
            self.logger.debug("shrinkage for training_shape %s", shrinkage)
            if shrinkage:
                if self.shrinkage is None:
                    self.shrinkage = shrinkage
                else:
                    assert self.shrinkage == shrinkage, f"self.shrinkage{self.shrinkage} == shrinkage{shrinkage}"
            else:
                raise ValueError(f"{TRAINING_SHAPE}: {training_shape} could not be processed on device: {devices}")
        else:
            # determine a valid training shape
            if TRAINING_SHAPE_UPPER_BOUND not in self.config:
                raise ValueError(f"config is missing {TRAINING_SHAPE} and/or {TRAINING_SHAPE_UPPER_BOUND}.")

            training_shape_upper_bound = BatchPointBase.from_spacetime(
                batch_size, input_channels, self.config[TRAINING_SHAPE_UPPER_BOUND]
            )

            if TRAINING_SHAPE_LOWER_BOUND in self.config:
                training_shape_lower_bound = BatchPointBase.from_spacetime(
                    batch_size, input_channels, self.config[TRAINING_SHAPE_LOWER_BOUND]
                )
            else:
                training_shape_lower_bound = training_shape_upper_bound.__class__()

            assert training_shape_lower_bound <= training_shape_upper_bound, (
                f"{TRAINING_SHAPE_LOWER_BOUND}: {training_shape_lower_bound} <= "
                f"{TRAINING_SHAPE_UPPER_BOUND}: {training_shape_upper_bound}"
            )

            # find optimal training shape
            training_shape = self.find_one_shape(
                training_shape_lower_bound, training_shape_upper_bound, devices=devices
            )

        return training_shape.drop_batch()

    def _determine_valid_shapes(
        self, devices: Sequence[torch.device], valid_shapes: Sequence[Union[Point2D, Point3D, Point4D]]
    ):
        # todo: find valid shapes
        if valid_shapes is None:
            self.valid_shapes = [self.training_shape]
        else:
            self.valid_shapes = [
                self.validate_shape(devices=devices, shape=s.add_batch(1), train_mode=False) for s in valid_shapes
            ]

    def minimal_device_test(self, devices: Sequence[torch.device]) -> Sequence[torch.device]:
        conns = [in_subproc(self._minimal_device_test, device=d) for d in devices]
        return [d for d, c in zip(devices, conns) if c.recv()]

    @staticmethod
    def _minimal_device_test(device: torch.device) -> bool:
        """
        Minimalistic test to check if a toy model can be loaded onto the device
        :return: True on success, False otherwise
        """
        try:
            with torch.no_grad():
                model = torch.nn.Conv2d(1, 1, 1).to(device)
                x = torch.zeros(1, 1, 1, 1).to(device)
                y = model(x)
                del model, x, y
        except Exception as e:
            # logger = logging.getLogger("DryRunProcess:minimal_device_test")
            # logger.error(e)
            return False

        return True

    def validate_shape(
        self, devices: Sequence[torch.device], shape: Union[BatchPoint2D, BatchPoint3D, BatchPoint4D], train_mode: bool
    ) -> Optional[PointBase]:
        assert devices
        shrinkage_conns = [
            in_subproc(self._validate_shape, model=self.model, device=d, shape=shape, train_mode=train_mode)
            for d in devices
        ]
        shrinkages = [conn.recv() for conn in shrinkage_conns]
        if None in shrinkages:
            return None

        shrink = shrinkages[0]
        if len(shrinkages) > 1:
            assert all([s == shrink for s in shrinkages[1:]])

        return shrink

    @staticmethod
    def _validate_shape(
        model: torch.nn.Module, device: torch.device, shape: PointAndBatchPointBase, train_mode: bool
    ) -> Optional[PointBase]:
        try:
            input = torch.ones(*shape).to(device=device)
            if train_mode:
                with torch.no_grad():
                    output = model.to(device)(input)
            else:
                output = model.to(device)(input)
                target = torch.randn_like(output)
                loss = torch.nn.MSELoss()
                loss(output, target).backward()
        except Exception:
            return None

        s_in = numpy.array(input.shape)
        s_out = numpy.array(output.shape)
        shrinkage = s_in - s_out
        assert all([s % 2 == 0 for s in shrinkage]), f"uneven shrinkage: {shrinkage}"
        shrinkage //= 2
        return shape.__class__(*shrinkage).drop_batch()

    def find_one_shape(
        self,
        lower_limit: Union[BatchPoint2D, BatchPoint3D, BatchPoint4D],
        upper_limit: Union[BatchPoint2D, BatchPoint3D, BatchPoint4D],
        devices: Sequence[torch.device],
        train_mode: bool = False,
        discard: float = 0,
    ) -> Optional[Union[BatchPoint2D, BatchPoint3D, BatchPoint4D]]:
        shape_class = type(lower_limit)
        assert (
            type(upper_limit) == shape_class
        ), f"type(upper_limit){type(upper_limit)} == type(lower_limit){type(lower_limit)}"
        lower_limit = numpy.array(lower_limit)
        upper_limit = numpy.array(upper_limit)
        diff = upper_limit - lower_limit
        assert all(diff >= 0), f"negative diff: {diff} = upper_limit({upper_limit}) - lower_limit({lower_limit}) "
        assert 0 <= discard < 1

        def update_nonzero(diff):
            nonzero_index = diff.nonzero()[0]
            nonzero = diff[nonzero_index]
            ndiff = len(nonzero)
            return nonzero_index, nonzero, ndiff

        nonzero_index, nonzero, ndiff = update_nonzero(diff)

        ncomb = numpy.prod(nonzero)
        if ncomb > 10000:
            self.logger.warning("Possibly testing too many combinations!!!")

        while ndiff:
            search_order = numpy.argsort(nonzero)[::-1]
            for diff_i in search_order:
                shape = shape_class(*(lower_limit + diff))
                shrinkage = self.validate_shape(devices=devices, shape=shape, train_mode=train_mode)
                if shrinkage:
                    if self.shrinkage is None:
                        self.shrinkage = shrinkage
                    else:
                        assert self.shrinkage == shrinkage

                    return shape

                reduced = int((1.0 - discard) * nonzero[diff_i] - 1)
                diff[nonzero_index[diff_i]] = reduced

            nonzero_index, nonzero, ndiff = update_nonzero(diff)

        return None

    def shutdown(self) -> Shutdown:
        self.logger.debug("Shutting down...")
        self.shutdown_event.set()
        try:
            self.dry_run_thread.join(timeout=20)
        except TimeoutError as e:
            self.logger.error(e)

        self.logger.debug("Shutdown complete")
        return Shutdown()
