import torch
import numpy as np
import importlib
from importlib import util as imputils
import yaml
import os
import logging
from . import utils


logger = logging.getLogger('TikTorch')


class TikTorch(object):
    def __init__(self, build_directory):
        # Privates
        self._build_directory = None
        self._model = None
        self._config = {}
        # Publics
        self.build_directory = build_directory
        self.read_config()

    @property
    def build_directory(self):
        if self._build_directory is not None:
            return self._build_directory
        else:
            raise ValueError("Trying to access `build_directory`, but it's not set yet.")

    @build_directory.setter
    def build_directory(self, value):
        if not os.path.exists(value):
            raise FileNotFoundError(f"Build directory does not exist: {value}")
        self._build_directory = value

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    def read_config(self):
        config_file_name = os.path.join(self.build_directory, 'tiktorch_config.yml')
        if not os.path.exists(config_file_name):
            raise FileNotFoundError(f"Config file not found in "
                                    f"build_directory: {self.build_directory}.")
        with open(config_file_name, 'r') as f:
            self._config.update(yaml.load(f))
        return self

    def get(self, tag, default=None, assert_exist=False):
        if assert_exist:
            assert tag in self._config, f"Tag '{tag}' not found in configuration."
        return self._config.get(tag, default)

    def set_devices(self, devices):
        # TODO Validate
        self._config.update({'devices': devices})
        return self

    @property
    def devices(self):
        devices = self.get('devices', None)
        if isinstance(devices, list):
            raise NotImplementedError("Multi-GPU support is not implemented yet.")
        if devices is None:
            return torch.device('cpu')
        elif isinstance(devices, int):
            return torch.device(f'cuda:{devices}')
        elif isinstance(devices, str):
            return torch.device(devices)
        else:
            raise ValueError

    def ensure_model_on_device(self):
        return self.model.to(self.devices)

    def load_model(self):
        # Dynamically import file.
        model_file_name = os.path.join(self.build_directory, 'model.py')
        module_spec = imputils.spec_from_file_location('model', model_file_name)
        module = imputils.module_from_spec(module_spec)
        # Build model from file
        model: torch.nn.Module = \
            getattr(module, self.get('model_class_name'))(**self.get('model_init_kwargs'))
        # Load parameters
        state_path = os.path.join(self.build_directory, 'state.nn')
        state_dict = torch.load(state_path)
        model.load_state_dict(state_dict)
        # Save attribute and return
        self._model = model
        return self

    def batch_inputs(self, inputs):
        input_shapes = self.get('input_shape', assert_exist=True)
        assert isinstance(input_shapes, (list, tuple))
        # input_shapes can either be a list of shapes or a shape. Make sure it's the latter
        if isinstance(input_shapes[0], int):
            input_shapes = [input_shapes] * len(inputs)
        elif isinstance(input_shapes[0], (list, tuple)):
            pass
        else:
            raise TypeError(f"`input_shapes` must be a list/tuple of ints or "
                            f"lists/tuples or ints. Got list/tuple of {type(input_shapes[0])}.")
        utils.assert_(len(input_shapes) == len(inputs),
                      f"Expecting {len(inputs)} inputs, got {len(input_shapes)} input shapes.",
                      ValueError)
        batches = [input.batcher(input_shape)
                   for input, input_shape in zip(inputs, input_shapes)]
        return batches

    def parse_inputs(self, inputs):
        if isinstance(inputs, TikIn):
            inputs = [inputs]
        elif isinstance(inputs, (np.ndarray, torch.Tensor)):
            inputs = [TikIn([inputs])]
        elif isinstance(inputs, (list, tuple)):
            utils.assert_(all(isinstance(input, TikIn) for input in inputs),
                          "Inputs must all be TikIn objects.")
        else:
            raise TypeError("Inputs must be list TikIn objects.")
        return inputs

    def forward(self, inputs: list):
        """
        Parameters
        ----------
        inputs: list of TikIn
            List of TikIn objects.
        """
        inputs = self.parse_inputs(inputs)
        # Batch inputs
        batches = self.batch_inputs(inputs)
        # Send batch to the right device
        batches = [batch.to(self.devices) for batch in batches]
        # Make sure model is in right device and feedforward
        output_batches = self.ensure_model_on_device()(*batches)
        if not isinstance(output_batches, (list, tuple)):
            output_batches = [output_batches]
        else:
            output_batches = list(output_batches)
        outputs = [TikOut(batch) for batch in output_batches]
        return outputs


class TikIO(object):

    class ShapeError(Exception):
        pass

    def __init__(self):
        self._tensors = None

    @staticmethod
    def to_torch_tensors(tensors):
        torch_tensors = []
        for tensor_num, tensor in enumerate(tensors):
            if isinstance(tensor, np.ndarray):
                torch_tensor = torch.from_numpy(tensor)
            elif isinstance(tensor, torch.Tensor):
                torch_tensor = tensor
            else:
                raise TypeError(f"Tensor at position must be numpy.ndarray or "
                                f"torch.Tensors, got {type(tensor)} instead.")
            torch_tensors.append(torch_tensor)
        return torch_tensors

    @property
    def tensors(self):
        utils.assert_(self._tensors is not None,
                      "Trying to acess `TikIn.tensors` with it yet to be defined.",
                      ValueError)
        return self._tensors

    @tensors.setter
    def tensors(self, value):
        self._tensors = self.to_torch_tensors(self.validate_shape(value))

    @staticmethod
    def validate_shape(tensors):
        utils.assert_([tensor.shape == tensors[0].shape for tensor in tensors],
                      f"Input `tensors` to TikIn must all have the same shape. "
                      f"Got tensors of shape: {[tensor.shape for tensor in tensors]}",
                      TikIn.ShapeError)
        utils.assert_(len(tensors[0].shape) in [2, 3, 4], f"Tensors must be of dimensions "
                                                          f"2 (HW), 3 (CHW/DHw), or 4 (CDHW). "
                                                          f"Got {len(tensors[0].shape)} instead.",
                      TikIO.ShapeError)
        return tensors


class TikIn(TikIO):

    def __init__(self, tensors):
        """
        Input Object for TikTorch.

        Parameters
        ----------
        tensors: list of numpy.ndarray or list of torch.Tensor
            Input Batch to TikTorch. Must be a list of tensors of the same shapes in any of the
            following formats:
            (C = Channel, H = Height, W = Width, D = Depth)
                a. (C, H, W)
                b. (C, D, H, W)
                c. (H, W)
                d. (D, H, W)
            C defaults to 1 if not provided. To resolve between (a) and (d), TikTorch uses
            the `input_shape` provided for model at build time.
        """
        super(TikIn, self).__init__()
        self.tensors = tensors

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def format(self):
        return {2: 'HW', 3: 'CHW/DHW', 4: 'CDHW'}[len(self.shape)]

    def reshape(self, *shape):
        return [tensor.reshape(*shape) for tensor in self.tensors]

    # MUSHROOMS MUUSHROOMS!
    def batcher(self, network_input_shape: list):
        """
        Build batch for the network.

        Parameters
        ----------
        network_input_shape: list
            Input shape to the network.

        Returns
        -------
        torch.Tensor
        """
        network_input_format = {3: 'CHW', 4: 'CDHW'}[len(network_input_shape)]
        if self.format == 'HW':
            utils.assert_(network_input_format == 'CHW',
                          f"Input format is HW, which is not compatible with the network "
                          f"input format {network_input_format} (must be CHW).",
                          TikIn.ShapeError)
            utils.assert_(network_input_shape[0] == 1,
                          f"Input format is HW, for which the number of input channels (C)"
                          f"to the network must be 1. Got C = {network_input_shape[0]} instead.",
                          TikIn.ShapeError)
            pre_cat = self.reshape(1, *self.shape)
        elif self.format == 'CDHW':
            utils.assert_(network_input_format == 'CDHW',
                          f"Input format (CDHW) is not compatible with network input format "
                          f"({network_input_format}).",
                          TikIn.ShapeError)
            utils.assert_(self.shape[0] == network_input_shape[0],
                          f"Number of input channels in input ({self.shape[0]}) is not "
                          f"consistent with what the network expects ({network_input_shape[0]}).",
                          TikIn.ShapeError)
            pre_cat = self.tensors
        elif self.format == 'CHW/DHW':
            if network_input_format == 'CHW':
                # input format is CHW
                utils.assert_(self.shape[0] == network_input_shape[0],
                              f"Number of input channels in input ({self.shape[0]}) is "
                              f"not compatible with the number of input channels to "
                              f"the network ({network_input_shape[0]})",
                              TikIn.ShapeError)
                pre_cat = self.tensors
            elif network_input_format == 'CDHW':
                utils.assert_(network_input_shape[0] == 1,
                              f"Input format DHW requires that the number of input channels (C) "
                              f"to the network is 1. Got C = {network_input_shape[0]} instead.",
                              TikIn.ShapeError)
                pre_cat = self.reshape(1, *self.shape)
            else:
                raise TikIn.ShapeError(f"Input format {self.format} is not compatible with "
                                       f"the network input format {network_input_format}.")
        else:
            raise ValueError("Internal Error: Invalid Format.")
        # Concatenate to a batch
        batch = torch.stack(pre_cat, dim=0)
        return batch


class TikOut(TikIO):
    def __init__(self, batch):
        super(TikOut, self).__init__()
        self.tensors = self.unbatcher(batch)

    def unbatcher(self, batch):
        utils.assert_(len(batch.shape) in [4, 5], f"`batch` must either be a NCHW or NCDHW tensor, "
                                                  f"got one with dimension {len(batch.shape)} "
                                                  f"instead.",
                      TikIO.ShapeError)
        return list(batch)
