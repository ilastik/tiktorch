import logging
import os
from importlib import util as imputils

import numpy as np
import torch
import yaml

from tiktorch.io import TikIn, TikOut
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
