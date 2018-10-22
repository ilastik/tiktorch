import os
import shutil
from importlib import util as imputils

import yaml
import torch


class FileExtensionError(Exception):
    pass


class BuildSpec(object):
    def __init__(self, build_directory, device='cpu'):
        """
        Parameters
        ----------
        build_directory: str
            Path to tiktorch configuration directory. Will be created if it doesn't exist.
        """
        self._build_directory = None
        self.build_directory = build_directory
        # TODO validate device
        self.device = device

    @property
    def build_directory(self):
        self.assert_(self._build_directory is not None,
                     "Trying to access `build_directory`, but it's not defined.",
                     ValueError)
        return self._build_directory

    @build_directory.setter
    def build_directory(self, value):
        os.makedirs(value, exist_ok=True)
        self._build_directory = value

    @classmethod
    def parse_args(cls):
        pass

    @classmethod
    def assert_(cls, condition, message='', exception_type=Exception):
        if not condition:
            raise exception_type(message)
        return cls

    def validate_path(self, path, extension=None):
        self.assert_(os.path.exists(path), f'Path not found: {path}', FileExistsError)
        if extension is not None:
            self.assert_(path.endswith(extension), f'Expected a .{extension} file for {path}.',
                         FileExtensionError)
        return self

    def copy_to_build_directory(self, path, name_in_build):
        # Validate path
        self.validate_path(path)
        # Check if file in build dir already
        if os.path.dirname(path) == self.build_directory:
            # Nothing to do
            return self
        destination = os.path.join(self.build_directory, name_in_build)
        shutil.copy(path, destination)
        return self

    def dump_config(self, config_dict):
        file_name = 'tiktorch_config.yml'
        dump_file_name = os.path.join(self.build_directory, file_name)
        with open(dump_file_name, 'w') as f:
            yaml.dump(config_dict, f)

    @staticmethod
    def _to_dynamic_shape(minimal_increment):
        if len(minimal_increment) == 2:
            dynamic_shape = '(%i * (nH + 1), %i * (nW + 1))' % minimal_increment
        elif len(minimal_increment) == 3:
            dynamic_shape = '(%i * (nD + 1), %i * (nH + 1), %i * (nW + 1))' % minimal_increment
        else:
            raise ValueError("Invald length %i for minimal increment" % len(minimal_increment))
        return dynamic_shape

    def _validate_spec(self, spec):
        # first, try to load the model
        try:
            module_spec = imputils.spec_from_file_location(spec.code_path)
            module = imputils.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            model: torch.nn.Module =\
                getattr(module, self.get(spec.model_class_name))(**spec.model_init_kwargs)
            model.to(self.device)
        except:
            raise ValueError(f'Could not load model {spec.model_class_name} from {spec.code_path}')
        # next, try to load the state
        try:
            state_dict = torch.load(spec.state_path)
            model.load_state_dict(state_dict)
        except:
            raise ValueError(f'Could not load model state from {spec.state_path}')
        # next, pipe iput of given shape through the network
        with torch.no_grad():
            try:
                input_ = torch.zeros(*spec.input_shape, dtype=torch.float())
                out = model(input_)
            except:
                raise ValueError(f'Input of shape {spec.input_shape} invalid for model')
        return tuple(out.shape)


    def build(self, spec):
        """
        Build tiktorch configuration.

        Parameters
        ----------
        spec: TikTorchSpec
            Specification Object
        """

        output_shape = self._validate_spec(spec)

        # Validate and copy code path
        self.validate_path(spec.code_path, 'py').copy_to_build_directory(spec.code_path,
                                                                         'model.py')
        # ... and weights path
        self.copy_to_build_directory(spec.state_path, 'state.nn')

        # Build and dump configuration dict
        # TODO why would we need the build directory in the config ?
        tiktorch_config = {# 'build_directory': self.build_directory,
                           'input_shape': spec.input_shape,
                           'output_shape': output_shape,
                           'dynamic_input_shape': self._to_dynamic_shape(minimal_increment),
                           'model_class_name': spec.model_class_name,
                           'model_init_kwargs': spec.model_init_kwargs,
                           'torch_version': torch.__version__}
        self.dump_config(tiktorch_config)
        # Done
        return self


class TikTorchSpec(object):
    def __init__(self, code_path=None, model_class_name=None, state_path=None,
                 input_shape=None, minimal_increment=None,
                 model_init_kwargs=None):
        """
        Parameters
        ----------
        code_path: str
            Path to the .py file where the model lives.
        model_class_name: str
            Name of the model class in code_path.
        state_path: str
            Path to where the state_dict is pickled. .
        input_shape: tuple or list
            Input shape of the model. Must be `CHW` (for 2D models) or `CDHW` (for 3D models).
        minimal_increment: tuple or list
            Minimal values by which to increment / decrement the input shape for it to still be valid.
        model_init_kwargs: dict
            Kwargs to the model constructor (if any).
        """
        self.code_path = code_path
        self.model_class_name = model_class_name
        self.state_path = state_path
        self.input_shape = input_shape
        self.minimal_increment = minimal_increment
        self.model_init_kwargs = model_init_kwargs or {}

        self.validate()

    @classmethod
    def assert_(cls, condition, message='', exception_type=Exception):
        if not condition:
            raise exception_type(message)
        return cls

    def validate(self):

        # TODO in the long run we should support both ways of serializing a model:
        # https://pytorch.org/docs/master/notes/serialization.html
        self.assert_(os.path.exists(self.state_path), f'Path not found: {self.state_path}', FileExistsError)
        self.assert_(os.path.exists(self.code_path), f'Path not found: {self.code_path}', FileExistsError)
        self.assert_(isinstance(self.model_class_name, str), "Model Class Name must be a string", ValueError)

        # TODO why do we care if this is list, tuple or whatever ?
        # self.assert_(isinstance(self.input_shape, list), "input_shape must be a list", ValueError)

        ndim = len(self.input_shape)
        self.assert_(ndim in (3, 4),
         f"input_shape has length {len(self.input_shape)} but should have lenght 3 or 4", ValueError)
        self.assert_(ndim - 1 == len(self.minimal_increment),
         f"minimal increment must have 1 entry less than input shape")

        self.assert_(isinstance(self.model_init_kwargs, dict), "model_init_kwargs must be a dictionary", ValueError)
        return self

def test_TikTorchSpec():
    code_path = '/Users'
    model_class_name = "DUNet2D"
    state_path = "/Users"
    input_shape = [1, 512, 512]
    output_shape = [1, 512, 512]
    dynamic_input_shape = '(32 * (nH + 1), 32 * (nW + 1))'
    devices = ['cpu:0']
    model_init_kwargs = {}

    spec = TikTorchSpec(code_path, model_class_name, state_path, input_shape,
                        output_shape, dynamic_input_shape, devices, model_init_kwargs)

def test_BuildyMcBuildface():
    spec = TikTorchSpec(code_path='/export/home/jhugger/sfb1129/test_configs_tiktorch/dunet2D.py',
                        model_class_name='DUNet2D',
                        state_path='/export/home/jhugger/sfb1129/test_configs_tiktorch/dunet2D_weights.nn',
                        input_shape=[3, 256, 256],
                        output_shape=[3, 256, 256],
                        dynamic_input_shape='(1 * (nD + 1), 32 * (nH + 1), 32 * (nW + 1))',
                        devices=['cuda:0', 'cuda:1'],
                        model_init_kwargs={'in_channels': 1, 'out_channels': 1})
    build = BuildyMcBuildface(build_directory='/export/home/jhugger/sfb1129/test_configs_tiktorch/config') \
            .build(spec)

if __name__ == '__main__':
    #test_TikTorchSpec()
    test_BuildyMcBuildface()
