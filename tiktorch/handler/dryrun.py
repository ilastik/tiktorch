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

from functools import partial
from multiprocessing.connection import Connection
from torch.multiprocessing import Process, Pipe, Queue

from .handledchildprocess import HandledChildProcess
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

from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION, TRAINING, VALIDATION, REQUEST_FOR_DEVICES
from ..configkeys import *
from ..tiktypes import *

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


class DryRunProcess(HandledChildProcess):
    """
    Process to execute a dry run to determine training and inference shape for 'model' on 'device'
    """

    def __init__(
        self, handler_conn: Connection, config: dict, model: torch.nn.Module, devices: Sequence[torch.device]
    ) -> None:
        """
        :param handler_conn: Connection to communicate with server
        :raises: TypeError if
        """
        super().__init__(handler_conn=handler_conn, name="DryRunProcess")
        self.handler_conn = handler_conn
        self.config = config
        self.model = model
        self.devices = devices

    @staticmethod
    def minimal_device_test(device: torch.device) -> bool:
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

    @staticmethod
    def validate_shape(
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
                self.handle_incoming_msgs_callback()  # got a shutdown message?
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

    def run(self) -> None:
        self.logger = logging.getLogger(self.name)
        self.logger.info("Starting")
        self._shutting_down: bool = False
        self.valid_shapes: List[Union[Point2D, Point3D, Point4D]] = []
        self.training_shape: Union[Point2D, Point3D, Point4D] = None
        self.shrinkage: Union[Point2D, Point3D, Point4D] = None
        try:
            works = []
            for d in self.devices:
                if self.minimal_device_test(device=d):
                    works.append(d)

            self.devices = works
            # todo: sort by vram
            smallest_device = self.devices[0]
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
                        training_shape_lower_bound = (0,) * len(training_shape_upper_bound)

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
                    self.handler_conn.send(
                        (
                            "set_devices_answer",
                            {
                                "failure_msg": f"{TRAINING_SHAPE}: {training_shape} could not be processed on smallest device: {smallest_device}"
                            },
                        )
                    )
                    raise ValueError(f"{TRAINING_SHAPE} {training_shape}")
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
                    self.handler_conn.send(
                        (
                            "set_devices_answer",
                            {
                                "failure_msg": f"config is missing {TRAINING_SHAPE} and {TRAINING_SHAPE_UPPER_BOUND}. Specify either!"
                            },
                        )
                    )
                    raise ValueError(TRAINING_SHAPE)

                # find optimal training shape
                training_shape = self.find_one_shape(
                    training_shape_lower_bound, training_shape_upper_bound, device=smallest_device
                )

            self.training_shape = training_shape

            # find valid inference shapes (starting from the training shape without batch_size)
            # todo: really look for valid shapes
            # self.dry_run_on_device(smallest_device, self.config[TRAINING_SHAPE_UPPER_BOUND])
            self.valid_shapes.append(self.training_shape.drop_batch())

            self.handler_conn.send(
                (
                    "set_devices_answer",
                    {"devices": smallest_devices, "valid_shapes": self.valid_shapes, "shrinkage": self.shrinkage},
                )
            )
            self.handle_incoming_msgs(timeout=5)  # will report devices to be idle
            time.sleep(30)
            self.logger.info("done")
        except Exception as e:
            self.logger.error(e)
            self.handler_conn.send((REPORT_EXCEPTION, {"proc_name": self.name, "exception": e}))
            self.shutdown()

    def shutdown(self):
        # todo: shutdown gracefully
        self._shutting_down = True
        raise ValueError(SHUTDOWN)


def dry_run_on_device(self, device: torch.device, upper_bound: PointAndBatchPointBase) -> Dict:
    """
    Dry run on device to determine valid shapes and the optimal shape.
    Parameters
    ----------
    :param upper_bound: upper bound for the search space.
    """
    self.logger = logging.getLogger(self.name)
    t, c, z, y, x = upper_bound

    if not self.valid_shapes:
        self.logger.info("Creating search space....")
        self.create_search_space(c, z, y, x, device, max_processing_time=2.5)

    self.logger.info("Searching for optimal shape...")
    optimal_entry = self.find()[1]

    if len(optimal_entry) == 1:
        optimal_shape = {"shape": [1, c, 1] + optimal_entry + [optimal_entry[-1]]}
    else:
        optimal_shape = {"shape": [1, c] + optimal_entry + [optimal_entry[-1]]}

    self.logger.info("Optimal shape found: %s", optimal_shape["shape"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return optimal_shape


def create_search_space(self, c, z, y, x, device, max_processing_time: float) -> None:
    """
    Generates a sorted list of shapes which self.model can process.
    """
    assert self.model is not None

    # check if model is 3d
    parent_conn, child_conn = Pipe()
    process_3d = Process(target=self.is_model_3d, args=(self.model, c, device, child_conn))
    process_3d.start()
    process_3d.join()
    process_3d.terminate()
    is_3d = parent_conn.recv()
    self.logger.debug("Is model 3d? %s", is_3d)

    # create search space by iterating through shapes
    shape_queue = Queue()
    search_space_process = Process(
        target=self.iterate_through_shapes,
        args=(self.model, c, z, y, x, device, is_3d, max_processing_time, shape_queue),
    )
    search_space_process.start()
    search_space_process.join()
    search_space_process.terminate()

    while True:
        try:
            shape = shape_queue.get_nowait()
            key = shape[0] * shape[1] if is_3d else shape[0]
            bisect.insort(self.valid_shapes, [key, shape])
        except queue.Empty:
            break


@staticmethod
def is_model_3d(model, channels, device, child_conn) -> None:
    # checks if model can process 3d input
    for z in range(6, 19, 1):
        for s in range(32, 300, 1):
            x = torch.zeros(1, channels, z, s, s)
            try:
                with torch.no_grad():
                    y = model.to(device)(x.to(device))
            except RuntimeError:
                logging.debug("Model cannot process tensors of shape %s", (1, channels, z, s, s))
            else:
                child_conn.send(True)
                return
            del x
    child_conn.send(False)


@staticmethod
def iterate_through_shapes(model, c, z, y, x, device, is_3d, max_processing_time, shape_queue: Queue) -> bool:
    def _forward(*args):
        input = torch.zeros(1, c, *args).to(device)
        start = time.time()

        try:
            with torch.no_grad():
                output = model.to(device)(input)
        except RuntimeError:
            del input
            logging.debug("Model cannot process tensors of shape %s. Vary size!", [1, c, *args])

            for s in range(np.max(args[-1] - 15, 0), np.min([args[-1] + 15, x, y])):
                if is_3d:
                    input = torch.zeros(1, c, args[0], s, s).to(device)
                else:
                    input = torch.zeros(1, c, s, s).to(device)
                start = time.time()

                try:
                    with torch.no_grad():
                        output = model.to(device)(input)
                except RuntimeError:
                    del input
                    if is_3d:
                        msg = [1, c, args[0], s, s]
                    else:
                        msg = [1, c, s, s]
                    logging.debug("Model cannot process tensors of shape %s", msg)
                else:
                    del output, input

                    if time.time() - start > max_processing_time:
                        return True
                    else:
                        if is_3d:
                            logging.debug("Add shape %s to search space", (args[0], s, s))
                            shape_queue.put([args[0], s])
                        else:
                            logging.debug("Add shape %s to search space", (s, s))
                            shape_queue.put([s])
                        return False
        else:
            del output, input

            if time.time() - start > max_processing_time:
                return True
            else:
                logging.debug("Add shape %s to search space", args)
                if is_3d:
                    shape_queue.put([args[0], args[-1]])
                else:
                    shape_queue.put([args[-1]])
                return False

    if is_3d:
        for _z in range(np.min([10, z]), np.min([80, z]), 3):
            for _s in range(np.min([32, x, y]), np.min([512, x, y]), 85):
                if _forward(_z, _s, _s):
                    break
    else:
        for _s in range(np.min([32, x, y]), np.min([2000, x, y]), 80):
            if _forward(_s, _s):
                break


@staticmethod
def find_forward(model, valid_shapes, c, i, device, child_conn, train_mode) -> None:
    x = valid_shapes[i][1] + [valid_shapes[i][1][-1]]
    x = torch.zeros(1, c, *x).to(device)
    try:
        if train_mode:
            y = model.to(device)(x)
            target = torch.randn(*y.shape)
            loss = torch.nn.MSELoss()
            loss(y, target).backward()
        else:
            with torch.no_grad():
                y = model.to(device)(x)
    except RuntimeError:
        child_conn.send(False)
    else:
        del y
        child_conn.send(True)


def find(self, lo: int = 0, hi: int = None) -> None:
    """
    Recursive search for the largest valid shape that self.model can process.

    """
    assert self.model is not None, "model needs to be loaded!"

    if not self.valid_shapes:
        raise ValueError("No valid shapes found!")

    if hi is None:
        hi = len(self.valid_shapes)

    if lo > hi:
        raise ValueError()

    mid = lo + (hi - lo) // 2

    parent_conn, child_conn = Pipe()

    if hi - lo <= 1:
        p = Process(
            target=self.find_forward, args=(self.model, self.valid_shapes, lo, self.device, child_conn, self.train_mode)
        )
        p.start()
        p.join()
        p.terminate()

        shape_found = parent_conn.recv()
        if shape_found:
            return self.valid_shapes[lo]
        else:
            raise RuntimeError("No valid shape found.")

    p = Process(
        target=self.find_forward, args=(self.model, self.valid_shapes, mid, self.device, child_conn, self.train_mode)
    )
    p.start()
    p.join()
    p.terminate()

    processable_shape = parent_conn.recv()
    if processable_shape:
        return self.find(mid, hi)
    else:
        return self.find(lo, mid)
