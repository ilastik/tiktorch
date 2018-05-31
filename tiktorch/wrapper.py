import torch
import numpy as np
import importlib
from importlib import util as imputils
import yaml
import os
import logging


logger = logging.getLogger('TikTorch')


class Tiktorch(object):
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
            self._config = yaml.load(f)
        return self

    def get(self, tag, default=None):
        return self._config.get(tag, default)

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
