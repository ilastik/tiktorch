from itertools import count
import logging
import time
import multiprocessing as mp
import queue
import random
import bisect

from copy import deepcopy

import torch
import numpy as np

from tiktorch.utils import DynamicShape, assert_, to_list, define_patched_model
from tiktorch.blockinator import Blockinator, th_pad
from tiktorch.trainy import Trainer

# from .dataloader import get_dataloader
# from .trainer import TikTorchTrainer

logger = logging.getLogger("DeviceHandler")


class DeviceMemoryCapacity(object):
    def __init__(self, num_blocks, dynamic_shape, device_id=None):
        self.num_blocks = num_blocks
        self.dynamic_shape = dynamic_shape
        self.device_id = device_id

    @property
    def shape(self):
        return self.dynamic_shape(*self.num_blocks)

    def is_device_capable(self, shape):
        # TODO
        pass

    def __repr__(self):
        return f"DeviceMemoryCapacity(max_shape={self.shape}, num_blocks={self.num_blocks})"


class Processor(object):
    def __init__(self, num_parallel_jobs=0):
        self._num_parallel_jobs = num_parallel_jobs

    @property
    def num_parallel_jobs(self):
        return self._num_parallel_jobs


class ModelHandler(Processor):
    def __init__(
        self, *, model, device_names, channels, dynamic_shape_code, training_hyperparams=None, log_directory=None
    ):
        # Privates
        self._model = None
        self._trainer = None
        self._max_batch_limit = 500
        self._halo = None
        self._channels = channels
        self._device_specs = {}
        self.__num_trial_runs_on_device = {}
        self._parameter_copy = None
        # Publics
        self.device_names = to_list(device_names)
        self.dynamic_shape = DynamicShape(dynamic_shape_code)
        self.valid_shape: list = []
        # Set
        self._set_model(model)
        self._set_trainer(training_hyperparams, log_directory)
        # Init superclass
        super(ModelHandler, self).__init__(num_parallel_jobs=len(self.devices))

    @property
    def channels(self):
        return self._channels

    @property
    def model(self):
        assert self._model is not None
        return self._model

    def _set_model(self, model):
        logger = logging.getLogger("ModelHandler._set_model")
        # Use this only once to prevent amusing bugs
        assert self._model is None
        logger.info("Sending model to %s", self.device)
        self._model = model.to(self.device)
        logger.info("Model on %s", next(model.parameters()).device)

    def _set_trainer(self, hyperparameters, log_directory=None):
        self._trainer = Trainer(handler=self, hyperparameters=hyperparameters, log_directory=log_directory)

    def _evaluate_parameter_diff(self):
        if self._parameter_copy is None:
            self._parameter_copy = [deepcopy(p) for p in self.model.parameters()]
            return 0
        else:
            param_now = [deepcopy(p) for p in self.model.parameters()]
            param_prev = self._parameter_copy
            self._parameter_copy = param_now
            # Compute diff
            with torch.no_grad():
                diff = sum([torch.norm(p_now - p_prev).item() for p_now, p_prev in zip(param_now, param_prev)])
            return diff

    @property
    def trainer(self):
        assert self._trainer is not None
        return self._trainer

    @property
    def device(self):
        return torch.device(self.device_names[0])

    @property
    def devices(self):
        return [torch.device(name) for name in self.device_names]

    @property
    def num_devices(self):
        return len(self.device_names)

    def get_device_spec(self, device_id):
        device_spec = self._device_specs.get(device_id)
        assert_(
            device_spec is not None,
            f"device_id {device_id} not found in specs. Consider calling dry_run() first.",
            RuntimeError,
        )
        return device_spec

    @property
    def halo_in_blocks(self):
        """
        Returns a list, containing the number of dynamic base shape blocks to cover the halo.
        """
        return [
            int(np.ceil(_halo / _block_shape)) for _halo, _block_shape in zip(self.halo, self.dynamic_shape.base_shape)
        ]

    @property
    def halo(self):
        if self._halo is None:
            self._halo = self.compute_halo()
        return self._halo

    @halo.setter
    def halo(self, value):
        if isinstance(value, int):
            self._halo = [value] * len(self.dynamic_shape)
        else:
            assert_(
                len(value) == len(self.dynamic_shape),
                f"Halo of a {len(self.dynamic_shape)}-D network cannot " f"be {len(value)}-D.",
                ValueError,
            )
            self._halo = value

    def train(self, data, labels):
        self.trainer.push(data, labels)
        return self

    def set_hparams(self, hparams):
        self.trainer.push_hparams(hparams)
        return self

    def pause_training(self):
        self.trainer.pause()
        return self

    def resume_training(self):
        self.trainer.resume()
        return self

    def stop_training(self):
        self.trainer.shut_down_training_process()
        return self

    def start_training(self):
        self.trainer.ignition()
        return self

    def training_process_is_alive(self):
        return self.trainer.is_alive()

    def update_state(self):
        logger = logging.getLogger("ModelHandler.update_state")
        if self.trainer.is_ignited:
            logger.info("Updating state...")
            self.trainer.update_handler_model_state()

    def dump_state(self, filename):
        state_dict = self.model.state_dict()
        torch.save(state_dict, filename)
        return self

    def dry_run(self, image_shape, train_flag=False):
        """
        Dry run to determine blockshape.
        Parameters
        ----------
        :param image_volume_shape (list or tuple in order (t, c, z, y, x)) as upper bound for the search space.
        :return: valid block shape (dict): {'shape': [1, c, z_opt, y_opt, x_opt]} (z_opt == 1 for 2d networks!)
        """
        t, c, z, y, x = image_shape
        if not self.valid_shape:
            logger.info("Creating search space....")
            self.create_search_space(c, z, y, x, max_processing_time=2.5)

        logger.info("Searching for optimal shape...")

        optEntry = self.find(c, train_flag=train_flag)[1]
        if len(optEntry) == 1:
            optShape = {"shape": [1, c, 1] + optEntry + [optEntry[-1]]}
        else:
            optShape = {"shape": [1, c] + optEntry + [optEntry[-1]]}
        logger.info("Optimal shape found: %s", optShape["shape"])

        return optShape

    def create_search_space(self, c, z, y, x, max_processing_time=3):
        """
        Generates a sorted list of shapes which self.model can process.
        """
        # check if model is 3d
        def _is_3d(is_3d_queue: mp.Queue):
            # checks if model can process 3d input
            logger = logging.getLogger("is_3d")
            for _z in range(1, 19, 1):
                for _s in range(32, 300, 1):
                    _input = torch.zeros(1, c, _z, _s, _s).to(self.device)
                    try:
                        with torch.no_grad():
                            _out = self.model.to(self.device)(_input)
                        is_3d_queue.put(True)
                        return
                    except RuntimeError:
                        logger.debug("Model cannot process tensors of shape %s", (1, c, _z, _s, _s))
            is_3d_queue.put(False)

        is_3d_queue = mp.Queue()
        _3d_check_process = mp.Process(target=_is_3d, args=(is_3d_queue,))
        _3d_check_process.start()
        _3d_check_process.join()
        is_3d = is_3d_queue.get_nowait()
        logger.debug("Is model 3d? %s", is_3d)
        ndim = 3 if is_3d else 2

        # create a search space of valid shapes
        def _create_search_space(shape_queue: mp.Queue):
            def _forward(*args):
                try:
                    start = time.time()
                    with torch.no_grad():
                        _out = self._model.to(self.device)(torch.zeros(1, c, *args).to(self.device))
                    del _out
                    if time.time() - start > max_processing_time:
                        return True
                    else:
                        logger.debug("Add shape %s to search space", args)
                        if is_3d:
                            shape_queue.put([args[0], args[-1]])
                        else:
                            shape_queue.put([args[-1]])
                        return False
                except RuntimeError:
                    logger.debug("Model cannot process tensors of shape %s. Vary size!", [1, c, *args])
                    for __s in range(np.max(args[-1] - 15, 0), np.min([args[-1] + 15, x, y])):
                        try:
                            _input = torch.zeros(1, c, args[0], __s, __s) if is_3d else torch.zeros(1, c, __s, __s)
                            start = time.time()
                            with torch.no_grad():
                                _out = self._model.to(self.device)(_input.to(self.device))
                            del _input, _out
                            if time.time() - start > max_processing_time:
                                return True
                            else:
                                if is_3d:
                                    logger.debug("Add shape %s to search space", (args[0], __s, __s))
                                    shape_queue.put([args[0], __s])
                                else:
                                    logger.debug("Add shape %s to search space", (__s, __s))
                                    shape_queue.put([__s])
                                return False
                        except RuntimeError:
                            del _input
                            _var_msg = [1, c, args[0], __s, __s] if is_3d else [1, c, __s, __s]
                            logger.debug("Model cannot process tensors of shape %s", _var_msg)

            if is_3d:
                for _z in range(np.min([1, z]), np.min([20, z]), 1):
                    for _s in range(np.min([32, x, y]), np.min([512, x, y]), 85):
                        if _forward(_z, _s, _s):
                            break
            else:
                for _s in range(np.min([32, x, y]), np.min([2000, x, y]), 80):
                    if _forward(_s, _s):
                        break

        shape_queue = mp.Queue()
        search_space_process = mp.Process(target=_create_search_space, args=(shape_queue,))
        search_space_process.start()
        search_space_process.join()

        while True:
            try:
                shape = shape_queue.get_nowait()
                key = shape[0] * shape[1] if is_3d else shape[0]
                bisect.insort(self.valid_shape, [key, shape])
            except queue.Empty:
                break

    def find(self, c, lo=0, hi=None, train_flag=False):
        """
        Recursive search for the largest valid shape that self.model can process.

        """
        assert self.model is not None
        if not self.valid_shape:
            raise ValueError()

        if hi is None:
            hi = len(self.valid_shape)

        if lo > hi:
            raise ValueError()

        mid = lo + (hi - lo) // 2

        def _forward(i: int, q: mp.Queue):
            # data to torch.tensor
            x = self.valid_shape[i][1] + [self.valid_shape[i][1][-1]]
            x = torch.zeros(1, c, *x).to(self.device)
            try:
                if train_flag:
                    y = self.model.to(self.device)(x)
                    target = torch.randn(*y.shape)
                    loss = torch.nn.MSELoss()
                    loss(y, target).backward()
                else:
                    with torch.no_grad():
                        y = self.model.to(self.device)(x)
                q.put(True)
            except RuntimeError:
                q.put(False)

        q = mp.Queue()

        if hi - lo <= 1:
            p = mp.Process(target=_forward, args=(lo, q))
            p.start()
            p.join()
            try:
                shape_found = q.get_nowait()
            except queue.Empty:
                logger.debug("Queue is empty!")
                raise RuntimeError("No valid shapes found.")
            if shape_found:
                return self.valid_shape[lo]
            else:
                raise RuntimeError("No valid shape found.")

        p = mp.Process(target=_forward, args=(mid, q))
        p.start()
        p.join()

        try:
            processable_shape = q.get_nowait()
        except queue.Empty:
            logger.debug("Queue is empty!")
            return self.find(c, lo, mid, train_flag)

        if processable_shape:
            return self.find(c, mid, hi, train_flag)
        else:
            return self.find(c, lo, mid, train_flag)

    @property
    def num_parallel_jobs(self):
        return self.num_devices

    def compute_halo(self, device_id=0, set_=True):
        device = self.devices[device_id]
        # Evaluate model on the smallest possible image to keep it quick
        input_tensor = torch.zeros(1, self.channels, *self.dynamic_shape.base_shape)
        output_tensor = torch.zeros(
            1, self.channels, *self.dynamic_shape.base_shape
        )  # self.model.to(device)(input_tensor.to(device))
        # Assuming NCHW or NCDHW, the first two axes are not relevant for computing halo
        input_spatial_shape = input_tensor.shape[2:]
        output_spatial_shape = output_tensor.shape[2:]
        shape_difference = [_ishape - _oshape for _ishape, _oshape in zip(input_spatial_shape, output_spatial_shape)]
        # Support for only symmetric halos for now
        assert_(
            all(_shape_diff % 2 == 0 for _shape_diff in shape_difference),
            "Only symmetric halos are supported.",
            RuntimeError,
        )
        # Compute halo
        halo = [_shape_diff // 2 for _shape_diff in shape_difference]
        if set_:
            self.halo = halo
        return halo

    def crop_output_tensor(self, tensor, num_channel_axes=2):
        base_shape = self.dynamic_shape.base_shape
        roi_shape = []
        for size, halo, blocks in zip(base_shape, self.halo, self.halo_in_blocks):
            if halo > 0:
                roi_shape.append(slice(size * blocks, -size * blocks))
            else:
                roi_shape.append(slice(None))
        return tensor[[slice(None)] * num_channel_axes + roi_shape]

    def crop_halo(self, tensor, num_channel_axes=2):
        base_shape = self.dynamic_shape.base_shape
        roi_shape = []
        for size, halo, blocks in zip(base_shape, self.halo, self.halo_in_blocks):
            if halo > 0:
                roi_shape.append(slice(size * blocks - halo, -(size * blocks - halo)))
            else:
                roi_shape.append(slice(None))
        return tensor[[slice(None)] * num_channel_axes + roi_shape]

    def forward(self, input_tensor: torch.Tensor):
        """
        Parameters
        ----------
        input_tensor: torch.Tensor
        """
        logger = logging.getLogger("ModelHandler.forward")
        self.update_state()
        logger.info("Params have changed by norm %s since last forward.", self._evaluate_parameter_diff())
        block = Blockinator(input_tensor, self.dynamic_shape.base_shape, num_channel_axes=2, pad_fn=th_pad)
        with block.attach(self):
            output_tensor = block.process()
        return output_tensor

    def to_device(self, obj):
        return obj.to(self.device)

    def process_tensor(self, tensor):
        tensor = self.to_device(tensor)
        return self.model(tensor)
