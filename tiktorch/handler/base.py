import importlib
import io
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

from contextlib import closing
from functools import partial
from multiprocessing.connection import Connection, wait
from torch.multiprocessing import Process, Pipe, active_children, Queue

from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from ..types import NDArrayBatch
from .constants import SHUTDOWN, SHUTDOWN_ANSWER, REPORT_EXCEPTION, TRAINING, VALIDATION
from .training import TrainingProcess
from .inference import InferenceProcess

logging.basicConfig(level=logging.DEBUG)
# logging.config.dictConfig({
#     'version': 1,
#     'disable_existing_loggers': False,
#     'handlers': {
#         'default': {
#             'level': 'INFO',
#             'class': 'logging.StreamHandler',
#             'stream': 'ext://sys.stdout',
#         },
#     },
#     'loggers': {
#         '': {
#             'handlers': ['default'],
#             'level': 'DEBUG',
#             'propagate': True
#         },
#     }
# })

class HandlerProcess(Process):
    """
    Process to orchestrate the interplay of training/validation and inference
    """

    def __init__(
        self, server_conn: Connection, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes
    ):
        """
        :param server_conn: Connection to communicate with server
        :param config: configuration dict
        :param model_file: bytes of file describing the neural network model
        :param model_state: binarized model state dict
        :param optimizer_state: binarized optimizer state dict
        """
        assert hasattr(self, SHUTDOWN[0]), "make sure the 'shutdown' method has the correct name"
        assert hasattr(self, REPORT_EXCEPTION), "make sure the 'report_exception' method has the correct name"
        assert model_file
        for required in ["model_class_name"]:
            if required not in config:
                raise ValueError(f"{required} missing in config")

        super().__init__(name="HandlerProcess")
        self.server_conn = server_conn
        self.config = config
        self.model_file = model_file  # to be deserialized in 'load_model'
        self.model_state = model_state  # to be deserialized in 'load_model'
        self.optimizer_state = optimizer_state  # to be deserialized in training process
        self.valid_shapes: list = []

    @property
    def devices(self):
        return self._devices

    @devices.setter
    def devices(self, devices: list):
        self.logger = logging.getLogger(self.name)
        self._devices = []

        for device in devices:
            try:
                torch_device = torch.device(device)
            except TypeError as e:
                self.logger.debug(e)
                devices.remove(device)
                continue

            parent_conn, child_conn = Pipe()
            p = Process(target=self.device_test,
                        args=(torch_device, child_conn))
            p.start()
            p.join()
            p.terminate()
            if parent_conn.recv():
                self._devices.append(torch_device)

    @staticmethod
    def device_test(device, conn):
        try:
            with torch.no_grad():
                model = torch.nn.Conv2d(1, 1, 1).to(device)
                x = torch.zeros(1, 1, 1, 1).to(device)
                y = model(x)
                del model, x, y
                conn.send(True)
        except RuntimeError as e:
            conn.send(False)
            raise RuntimeError(e)

    def dry_run_on_device(self, device: torch.device, upper_bound, train_mode=False):
        """
        Dry run on device to determine blockshape.
        Parameters
        ----------
        :param upper_bound (list or tuple in order (t, c, z, y, x)) upper bound for the search space.
        :return: valid block shape (dict): {'shape': [1, c, z_opt, y_opt, x_opt]} (z_opt == 1 for 2d networks!)
        """
        self.logger = logging.getLogger(self.name)
        self.load_model()
        t, c, z, y, x = upper_bound
        if not self.valid_shapes:
            self.logger.info("Creating search space....")
            self.create_search_space(c, z, y, x, device, max_processing_time=2.5)

        self.logger.info("Searching for optimal shape...")

        optimal_entry = self.find(c, device=device, train_mode=train_mode)[1]
        if len(optimal_entry) == 1:
            optimal_shape = {"shape": [1, c, 1] + optimal_entry + [optimal_entry[-1]]}
        else:
            optimal_shape = {"shape": [1, c] + optimal_entry + [optimal_entry[-1]]}

        self.logger.info("Optimal shape found: %s", optimal_shape["shape"])

        return optimal_shape

    def create_search_space(self, c, z, y, x, device, max_processing_time=3):
        """
        Generates a sorted list of shapes which self.inference_model can process.
        """
        assert self.infernce_model is not None
        
        # check if model is 3d
        def _is_3d(is_3d_queue: Queue):
            # checks if model can process 3d input
            logger = logging.getLogger("is_3d")
            for _z in range(1, 19, 1):
                for _s in range(32, 300, 1):
                    _input = torch.zeros(1, c, _z, _s, _s)
                    try:
                        with torch.no_grad():
                            _out = self.inference_model.to(device)(_input.to(device))
                        is_3d_queue.put(True)
                        return
                    except RuntimeError:
                        logger.debug("Model cannot process tensors of shape %s", (1, c, _z, _s, _s))
            is_3d_queue.put(False)

        is_3d_queue = Queue()
        _3d_check_process = Process(target=_is_3d, args=(is_3d_queue,))
        _3d_check_process.start()
        _3d_check_process.join()
        is_3d = is_3d_queue.get_nowait()
        self.logger.debug("Is model 3d? %s", is_3d)

        # create a search space of valid shapes
        def _create_search_space(shape_queue: Queue):
            def _forward(*args):
                try:
                    start = time.time()
                    with torch.no_grad():
                        _out = self.inference_model.to(device)(torch.zeros(1, c, *args).to(device))
                    del _out
                    if time.time() - start > max_processing_time:
                        return True
                    else:
                        self.logger.debug("Add shape %s to search space", args)
                        if is_3d:
                            shape_queue.put([args[0], args[-1]])
                        else:
                            shape_queue.put([args[-1]])
                        return False
                except RuntimeError:
                    self.logger.debug("Model cannot process tensors of shape %s. Vary size!", [1, c, *args])
                    for __s in range(np.max(args[-1] - 15, 0), np.min([args[-1] + 15, x, y])):
                        try:
                            _input = torch.zeros(1, c, args[0], __s, __s) if is_3d else torch.zeros(1, c, __s, __s)
                            start = time.time()
                            with torch.no_grad():
                                _out = self.inference_model.to(device)(_input.to(device))
                            del _input, _out
                            if time.time() - start > max_processing_time:
                                return True
                            else:
                                if is_3d:
                                    self.logger.debug("Add shape %s to search space", (args[0], __s, __s))
                                    shape_queue.put([args[0], __s])
                                else:
                                    self.logger.debug("Add shape %s to search space", (__s, __s))
                                    shape_queue.put([__s])
                                return False
                        except RuntimeError:
                            del _input
                            _var_msg = [1, c, args[0], __s, __s] if is_3d else [1, c, __s, __s]
                            self.logger.debug("Model cannot process tensors of shape %s", _var_msg)

            if is_3d:
                for _z in range(np.min([1, z]), np.min([20, z]), 1):
                    for _s in range(np.min([32, x, y]), np.min([512, x, y]), 85):
                        if _forward(_z, _s, _s):
                            break
            else:
                for _s in range(np.min([32, x, y]), np.min([2000, x, y]), 80):
                    if _forward(_s, _s):
                        break

        shape_queue = Queue()
        search_space_process = Process(target=_create_search_space, args=(shape_queue,))
        search_space_process.start()
        search_space_process.join()

        while True:
            try:
                shape = shape_queue.get_nowait()
                key = shape[0] * shape[1] if is_3d else shape[0]
                bisect.insort(self.valid_shapes, [key, shape])
            except queue.Empty:
                break

    def find(self, c, lo=0, hi=None, device=torch.device('cpu'), train_mode=False):
        """
        Recursive search for the largest valid shape that self.infernce_model can process.

        """
        assert self.inference_model is not None
        if not self.valid_shapes:
            raise ValueError()

        if hi is None:
            hi = len(self.valid_shapes)

        if lo > hi:
            raise ValueError()

        mid = lo + (hi - lo) // 2

        def _forward(i: int, q: Queue):
            # data to torch.tensor
            x = self.valid_shapes[i][1] + [self.valid_shapes[i][1][-1]]
            x = torch.zeros(1, c, *x).to(device)
            try:
                if train_mode:
                    y = self.inference_model.to(self.device)(x)
                    target = torch.randn(*y.shape)
                    loss = torch.nn.MSELoss()
                    loss(y, target).backward()
                else:
                    with torch.no_grad():
                        y = self.inference_model.to(device)(x)
                del y
                q.put(True)
            except RuntimeError:
                q.put(False)

        q = Queue()

        if hi - lo <= 1:
            p = Process(target=_forward, args=(lo, q))
            p.start()
            p.join()
            try:
                shape_found = q.get_nowait()
            except queue.Empty:
                self.logger.debug("Queue is empty!")
                raise RuntimeError("No valid shapes found.")
            if shape_found:
                return self.valid_shapes[lo]
            else:
                raise RuntimeError("No valid shape found.")

        p = Process(target=_forward, args=(mid, q))
        p.start()
        p.join()

        try:
            processable_shape = q.get_nowait()
        except queue.Empty:
            self.logger.debug("Queue is empty!")
            return self.find(c, lo, mid, device, train_mode)

        if processable_shape:
            return self.find(c, mid, hi, device, train_mode)
        else:
            return self.find(c, lo, mid, device, train_mode)

    def load_model(self):
        self.tempdir = tempfile.mkdtemp()
        user_module_name = "usermodel"
        with open(os.path.join(self.tempdir, user_module_name + ".py"), "w") as f:
            f.write(pickle.loads(self.model_file))

        sys.path.insert(0, self.tempdir)
        user_module = importlib.import_module(user_module_name)

        self.training_model: torch.nn.Module = getattr(user_module, self.config["model_class_name"])(
            **self.config.get("model_init_kwargs", {})
        )
        self.inference_model: torch.nn.Module = getattr(user_module, self.config["model_class_name"])(
            **self.config.get("model_init_kwargs", {})
        )
        self.logger.debug("created user models")

        if self.model_state:
            with closing(io.BytesIO(self.model_state)) as f:
                self.training_model.load_state_dict(torch.load(f))

            self.logger.info("restored model state")

    def run(self):
        self._shutting_down = False
        self.logger = logging.getLogger(self.name)
        self.logger.info("Starting")
        try:
            self.load_model()

            # set up training process
            self.training_conn, handler_conn_training = Pipe()
            self.training_proc = TrainingProcess(
                handler_conn=handler_conn_training,
                config=self.config,
                model=self.training_model,
                optimizer_state=self.optimizer_state,
            )
            self.training_proc.start()
            self.inference_conn, handler_conn_inference = Pipe()
            self.inference_proc = InferenceProcess(
                handler_conn=handler_conn_inference, config=self.config, model=self.inference_model
            )
            self.inference_proc.start()

            while not self._shutting_down:
                self.logger.debug("loop")
                for ready in wait([self.server_conn, self.inference_conn, self.training_conn]):
                    call, kwargs = ready.recv()
                    self.logger.debug("got call %s", call)
                    meth = getattr(self, call, None)
                    if meth is None:
                        raise NotImplementedError(call)

                    meth(**kwargs)
        except Exception as e:
            self.logger.error(str(e))
            self.server_conn.send(("report_shutdown", {"name": self.name, "exception": e}))
            self.shutdown()

    # general
    def active_children(self):
        self.server_conn.send([child_proc.name for child_proc in active_children()])

    def report_exception(self, proc_name: str, exception: Exception):
        self.logger.error("Received exception report from %s: %s", proc_name, exception)
        if proc_name == TrainingProcess.name:
            # todo: restart training proess
            pass
        elif proc_name == InferenceProcess.name:
            # todo: restart inference process
            pass
        else:
            raise NotImplementedError("Did not expect exception report form %s", proc_name)

    def shutdown(self):
        logger = logging.getLogger(self.name + ".shutdown")
        logger.debug("Shutting down...")
        self._shutting_down = True
        try:
            self.inference_conn.send(SHUTDOWN)
        except Exception as e:
            logger.error(e)
        try:
            self.training_conn.send(SHUTDOWN)
        except Exception as e:
            logger.error(e)

        self.server_conn.send(SHUTDOWN_ANSWER)

        shutdown_time = 20

        def shutdown_child(conn, proc):
            while proc.is_alive():
                # look for shutting down answer
                if conn.poll(timeout=shutdown_time):
                    answer = conn.recv()
                    if answer == SHUTDOWN_ANSWER:
                        # give child process extra time to shutdown
                        proc.join(timeout=shutdown_time)
                else:
                    logger.error("Failed to shutdown %s gracefully. Sending kill...", proc.name)
                    proc.kill()
                    proc.join(timeout=shutdown_time)

            logger.debug("%s has shutdown", proc.name)

        try:
            shutdown_child(self.inference_conn, self.inference_proc)
        except Exception as e:
            logger.error(str(e))
        try:
            shutdown_child(self.training_conn, self.training_proc)
        except Exception as e:
            logger.error(str(e))

        shutil.rmtree(self.tempdir)
        logger.debug("Shutdown complete")

    def update_hparams(self, hparams: dict):
        pass

    def generic_relay_to_server(self, method_name: str, **kwargs):
        if not self._shutting_down:
            self.server_conn.send((method_name, kwargs))

    def __getattr__(self, method_name):
        if method_name in ["forward_answer"]:
            return partial(self.generic_relay_to_server, method_name=method_name)
        else:
            raise AttributeError(method_name)

    def report_idle(self, proc_name: str, devices: Iterable):
        # todo: report idle
        if proc_name == TrainingProcess.name:
            pass
        elif proc_name == InferenceProcess.name:
            pass
        else:
            raise NotImplementedError(proc_name)

    # inference
    def update_inference_model(self):
        with io.BytesIO() as bytes_io:
            torch.save(self.training_model.state_dict(), bytes_io)
            bytes_io.seek(0)
            self.inference_model.load_state_dict(torch.load(bytes_io))

        self.inference_model.eval()

    def forward(self, keys: Iterable, data: torch.Tensor) -> None:
        self.logger.debug("forward")
        self.update_inference_model()
        # todo: update inference devices
        self.inference_conn.send(("forward", {"keys": keys, "data": data}))

    # training
    def free_gpu(self):
        self.training_conn.send(('free_gpu', {}))

    def resume_training(self):
        device = "cpu"
        if self.devices["training"]:
            raise NotImplementedError("gpu training")
            # todo: update training devices (probably we should not use CUDA_VISIBLE_DEVICES due to the process already running)
            os.environ["CUDA_VISIBLE_DEVICES"] = self.devices["training"]
            device = "gpu"

        self.training_conn.send((self.resume_training.__name__, {"device": "cpu"}))

    def pause_training(self):
        self.training_conn.send((self.pause_training.__name__, {}))

    def pause_training_answer(self):
        self.devices["idle"] = self.devices["training"]
        self.devices["training"] = []

    def update_training_dataset(self, keys: Iterable, data: torch.Tensor):
        self.training_conn.send(("update_dataset", {"name": TRAINING, "keys": keys, "data": data}))

    def request_state(self):
        # model state
        # optimizer state
        # current config dict
        pass

    # validation
    def update_validation_dataset(self, keys: Iterable, data: torch.Tensor):
        self.training_conn.send(("update_dataset", {"name": VALIDATION, "keys": keys, "data": data}))

    def validate(self):
        pass
