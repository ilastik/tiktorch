import os
import shutil
import yaml


class FileExtensionError(Exception):
    pass


class BuildyMcBuildface(object):
    def __init__(self, build_directory):
        """
        Parameters
        ----------
        build_directory: str
            Path to tiktorch configuration directory. Will be created if it doesn't exist.
        """
        self._build_directory = None
        self.build_directory = build_directory

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

    def build(self, spec):
        """
        Build tiktorch configuration.

        Parameters
        ----------
        spec: TikTorchSpec
            Specification Object
        """
        # Validate and copy code path
        self.validate_path(spec.code_path, 'py').copy_to_build_directory(spec.code_path,
                                                                         'model.py')
        # ... and weights path
        self.validate_path(spec.state_path, 'nn').copy_to_build_directory(spec.state_path,
                                                                          'state.nn')
        # Build and dump configuration dict
        tiktorch_config = spec.__dict__
        tiktorch_config.update({'build_directory': self.build_directory})
        self.dump_config(tiktorch_config)
        # Done
        return self


class TikTorchSpec(object):
    def __init__(self, code_path=None, model_class_name=None, state_path=None,
                 input_shape=None, output_shape=None, dynamic_input_shape=None,
                 devices=None, model_init_kwargs=None):
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
        output_shape: tuple or list
            Shape of the model output. Must be `CHW` (for 2D models) or `CDHW` (for 3D models)
        dynamic_input_shape: str
            String specifying how to select dynamic values for (D,) H, W in C(D)HW shapes.
            The string must have a "n{H, W, D}" in it somewhere, which will be interpreted as
            integers starting at 0.
            For instance, "(64 * 2 ** nD, 32 * 2 ** nH, 32 * 2 ** nW)" would mean the following
            possible DHW shapes:
                (nD=0, nH = 0, nW = 0) --> (64, 32, 32),
                (nD=1, nH = 0, nW = 0) --> (128, 32, 32),
                (nD=0, nH = 1, nW = 2) --> (64, 64, 128),
                (nD=1, nH = 3, nW = 4) --> (128, 256, 512),
                ...
        devices: list
            List of devices to use (e.g. 'cpu:0' or ['cuda:0', 'cuda:1']).
        model_init_kwargs: dict
            Kwargs to the model constructor (if any).
        """
        self.code_path = code_path
        self.model_class_name = model_class_name
        self.state_path = state_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dynamic_input_shape = dynamic_input_shape
        self.devices = devices
        self.model_init_kwargs = model_init_kwargs or {}

    def validate(self):
        # TODO Validate arguments
        return self
