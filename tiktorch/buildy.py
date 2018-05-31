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
            yaml.dump(config_dict, dump_file_name)

    def build(self, code_path, model_class_name, state_path,
              input_shape, output_shape, **model_init_kwargs):
        """
        Build tiktorch configuration.

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
        model_init_kwargs: dict
            Kwargs to the model constructor (if any).
        """
        # Validate and copy code path
        self.validate_path(code_path, 'py').copy_to_build_directory(code_path, 'model.py')
        # ... and weights path
        self.validate_path(state_path, 'nn').copy_to_build_directory(state_path, 'state.nn')
        # Build and dump configuration dict
        tiktorch_config = {'build_directory': self.build_directory,
                           'model_class_name': model_class_name,
                           'input_shape': input_shape,
                           'output_shape': output_shape,
                           'model_init_kwargs': model_init_kwargs}
        self.dump_config(tiktorch_config)
        # Done
        return self


