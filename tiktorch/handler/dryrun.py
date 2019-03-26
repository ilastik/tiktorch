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
from torch.multiprocessing import Process, Pipe, Queue

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

from tiktorch.rpc import RPCInterface, exposed, Shutdown
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

from tiktorch.configkeys import *

# logging.basicConfig(level=logging.DEBUG)
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {"default": {"level": "DEBUG", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"}},
        "loggers": {"": {"handlers": ["default"], "level": "DEBUG", "propagate": True}},
    }
)


def ret_through_conn(*args, fn: Callable, conn: Connection, **kwargs) -> None:
    conn.send(fn(*args, **kwargs))


def in_subproc(fn: Callable, *args, **kwargs) -> Connection:
    """
    Run 'fn' in a subprocess and return a connection that will hold the result
    :param fn: function to run in subprocess
    :return: Connection to result
    """
    recv_conn, send_conn = Pipe(duplex=False)
    subproc = Process(target=ret_through_conn, args=args, kwargs={"fn": fn, "conn": send_conn, **kwargs})
    subproc.start()
    return recv_conn


class IDryRun(RPCInterface):
    @exposed
    def dry_run(self, devices: Sequence[torch.device]) -> Future:
        raise NotImplementedError()

    @exposed
    def shutdown(self):
        raise NotImplementedError()


def run(conn: Connection, config: dict, model: torch.nn.Module):
    # print('CUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])
    dryrun_proc = DryRunProcess(config, model)
    srv = MPServer(dryrun_proc, conn)
    srv.listen()


class DryRunProcess(IDryRun):
    """
    Process to execute a dry run to determine training and inference shape for 'model' on 'device'
    """

    name = "DryRunProcess"

    def __init__(self, config: dict, model: torch.nn.Module) -> None:
        self.logger = logging.getLogger(self.name)
        self.config = config
        self.model = model

        self._shutdown_event = threading.Event()

        self.dry_run_queue = queue.Queue()
        self.dry_run_thread = threading.Thread(target=self._dry_run_worker)
        self.dry_run_thread.start()

    def _dry_run_worker(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                devices, fut = self.dry_run_queue.get(block=True, timeout=1)
            except queue.Empty:
                pass
            else:
                self._dry_run(devices, fut)

    def dry_run(self, devices: Sequence[torch.device]) -> Future:
        fut = Future()
        self.dry_run_queue.put((devices, fut))
        return fut

    def _dry_run(self, devices: Sequence[torch.device], fut: Future) -> None:
        self.logger.info("Starting dry run")
        self.valid_shapes: List[Union[Point2D, Point3D, Point4D]] = []
        self.shrinkage: Union[Point2D, Point3D, Point4D] = None
        try:
            works = []
            for d in devices:
                if self.minimal_device_test(d):
                    works.append(d)

            devices = works
            # todo: sort by vram
            smallest_device = devices[0]
            # todo: determine all smallest devices
            smallest_devices = [smallest_device]

            batch_size = self.config[BATCH_SIZE]
            input_channels = self.config[INPUT_CHANNELS]
            if TRAINING_SHAPE in self.config:
                # validate given training shape
                training_shape = self.config[TRAINING_SHAPE]
                training_shape = BatchPointBase.from_spacetime(batch_size, input_channels, training_shape)

                if TRAINING_SHAPE_UPPER_BOUND in self.config:
                    training_shape_upper_bound = BatchPointBase.from_spacetime(
                        batch_size, input_channels, self.config[TRAINING_SHAPE_UPPER_BOUND]
                    )
                    assert (
                        training_shape <= training_shape_upper_bound
                    ), f"{TRAINING_SHAPE}{training_shape} <= {TRAINING_SHAPE_UPPER_BOUND}{training_shape_upper_bound}"

                    if TRAINING_SHAPE_LOWER_BOUND in self.config:
                        training_shape_lower_bound = self.config[TRAINING_SHAPE_LOWER_BOUND]
                    else:
                        training_shape_lower_bound = (1,) * len(training_shape_upper_bound)

                    assert (
                        training_shape_lower_bound <= training_shape
                    ), f"{TRAINING_SHAPE_LOWER_BOUND}{training_shape_lower_bound} <= {TRAINING_SHAPE}{training_shape}"

                shrinkage = self.validate_shape(
                    model=self.model, device=smallest_device, shape=training_shape, train_mode=True
                )
                self.logger.debug("shrinkage for training_shape %s", shrinkage)
                if shrinkage:
                    if self.shrinkage is None:
                        self.shrinkage = shrinkage
                    else:
                        assert self.shrinkage == shrinkage
                else:
                    raise ValueError(
                        f"{TRAINING_SHAPE}: {training_shape} could not be processed on smallest device: {smallest_device}"
                    )
            else:
                # determine a valid training shape
                if TRAINING_SHAPE_UPPER_BOUND in self.config:
                    training_shape_upper_bound = BatchPointBase.from_spacetime(
                        batch_size, input_channels, self.config[TRAINING_SHAPE_UPPER_BOUND]
                    )
                    if TRAINING_SHAPE_LOWER_BOUND in self.config:
                        training_shape_lower_bound = self.config[TRAINING_SHAPE_LOWER_BOUND]
                        assert all(
                            [mini <= maxi for mini, maxi in zip(training_shape_lower_bound, training_shape_upper_bound)]
                        )
                    else:
                        training_shape_lower_bound = (0,) * len(training_shape_upper_bound)

                    assert training_shape_lower_bound <= training_shape_upper_bound, (
                        f"{TRAINING_SHAPE_LOWER_BOUND}{training_shape_lower_bound} <= "
                        f"{TRAINING_SHAPE_UPPER_BOUND}{training_shape_upper_bound}"
                    )
                else:
                    raise ValueError(f"config is missing {TRAINING_SHAPE} and/or {TRAINING_SHAPE_UPPER_BOUND}.")

                # find optimal training shape
                training_shape = self.find_one_shape(
                    training_shape_lower_bound, training_shape_upper_bound, device=smallest_device
                )

            self.training_shape = training_shape.drop_batch()

            # find valid inference shapes (starting from the training shape without batch_size)
            # todo: really look for valid shapes
            # self.dry_run_on_device(smallest_device, self.config[TRAINING_SHAPE_UPPER_BOUND])
            self.valid_shapes.append(self.training_shape)

            fut.set_result((devices, self.training_shape, self.valid_shapes, self.shrinkage))
            self.logger.info("dry run done")
        except Exception as e:
            self.logger.error(e)
            fut.set_exception(e)

    def minimal_device_test(self, device: torch.device) -> bool:
        return in_subproc(self._minimal_device_test, device=device).recv()

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
            logger = logging.getLogger("DryRunProcess:minimal_device_test")
            logger.error(e)
            return False

        return True

    def validate_shape(
        self, model: torch.nn.Module, device: torch.device, shape: PointAndBatchPointBase, train_mode: bool
    ) -> Optional[PointBase]:
        return in_subproc(self._validate_shape, model=model, device=device, shape=shape, train_mode=train_mode).recv()

    @staticmethod
    def _validate_shape(
        model: torch.nn.Module, device: torch.device, shape: PointAndBatchPointBase, train_mode: bool
    ) -> Optional[PointBase]:
        try:
            input = torch.ones(*shape).to(device)
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
        lower_limit: PointAndBatchPointBase,
        upper_limit: PointAndBatchPointBase,
        device: torch.device,
        train_mode: bool = False,
        discard: float = 0,
    ) -> Optional[PointAndBatchPointBase]:
        shape_class = type(lower_limit)
        assert type(upper_limit) == shape_class
        lower_limit = numpy.array(lower_limit)
        upper_limit = numpy.array(upper_limit)
        diff = lower_limit - upper_limit
        assert all(diff >= 0), f"{upper_limit} - {lower_limit} = {diff}"
        nonzero_index = diff.nonzero()
        nonzero = diff[nonzero_index]
        ncomb = numpy.prod(nonzero)
        if ncomb > 10000:
            logger = logging.getLogger("DryRunProcess.find_one_shape")
            logger.error("Possibly testing too many combinations!!!")

        ndiff = len(nonzero)
        while ndiff:
            search_order = numpy.argsort(nonzero)
            for diff_i in search_order:
                for reduced in range(
                    int((1.0 - discard) * nonzero[diff_i]),
                    min(int(0.95 * nonzero[diff_i] - 1), int(0.90 * nonzero[diff_i])),
                    -1,
                ):
                    diff[nonzero_index[diff_i]] = reduced
                    shape = shape_class(lower_limit + diff)
                    shrinkage = self.validate_shape(device=device, shape=shape, train_mode=train_mode)
                    if shrinkage:
                        if self.shrinkage is None:
                            self.shrinkage = shrinkage
                        else:
                            assert self.shrinkage == shrinkage

                        return shape

        return None

    def shutdown(self):
        self._shutdown_event.set()
        self.dry_run_thread.join()
        raise Shutdown


