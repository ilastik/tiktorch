import logging
import os
import shutil
from importlib import util as imputils

import torch
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BuildSpec")


class FileExtensionError(Exception):
    pass


class BuildSpec(object):
    def __init__(self, build_directory, device="cpu"):
        """
        Parameters
        ----------
        build_directory: str
            Path to tiktorch configuration directory. Will be created if it doesn't exist.
        """
        self._build_directory = None
        self.build_directory = build_directory
        self._device = None
        self.device = device

    @property
    def device(self):
        self.assert_(self._device is not None, "Trying to access `device`, but it's not defined.", ValueError)
        return self._device

    @device.setter
    def device(self, value):
        self.assert_("cpu" in value or "cuda" in value, "Value for `device` is not valid.", ValueError)
        self._device = value

    @property
    def build_directory(self):
        self.assert_(
            self._build_directory is not None, "Trying to access `build_directory`, but it's not defined.", ValueError
        )
        return self._build_directory

    @build_directory.setter
    def build_directory(self, value):
        os.makedirs(value, exist_ok=True)
        self._build_directory = value

    @classmethod
    def parse_args(cls):
        pass

    @classmethod
    def assert_(cls, condition, message="", exception_type=Exception):
        if not condition:
            raise exception_type(message)
        return cls

    def validate_path(self, path, extension=None):
        self.assert_(os.path.exists(path), f"Path not found: {path}", FileExistsError)
        if extension is not None:
            self.assert_(path.endswith(extension), f"Expected a .{extension} file for {path}.", FileExtensionError)
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
        file_name = "tiktorch_config.yml"
        dump_file_name = os.path.join(self.build_directory, file_name)
        with open(dump_file_name, "w") as f:
            yaml.dump(config_dict, f)

    @staticmethod
    def _to_dynamic_shape(minimal_increment):
        assert isinstance(minimal_increment, list) or isinstance(minimal_increment, tuple)
        if len(minimal_increment) == 2:
            dynamic_shape = "(%i * (nH + 1), %i * (nW + 1))" % tuple(minimal_increment)
        elif len(minimal_increment) == 3:
            dynamic_shape = "(%i * (nD + 1), %i * (nH + 1), %i * (nW + 1))" % tuple(minimal_increment)
        else:
            raise ValueError("Invald length %i for minimal increment" % len(minimal_increment))
        return dynamic_shape

    def _validate_spec(self, spec):
        # first, try to load the model
        logger.info("Validate TikTorchSpec object.")
        try:
            module_spec = imputils.spec_from_file_location("model", spec.code_path)
            module = imputils.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            model: torch.nn.Module = getattr(module, spec.model_class_name)(**spec.model_init_kwargs)
        except:
            raise ValueError(f"Could not load model {spec.model_class_name} from {spec.code_path}")
        # next, try to load the state
        try:
            state_dict = torch.load(spec.state_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
            model.to(self.device)
            logger.info("Model successfully loaded.")
        except:
            raise ValueError(f"Could not load model state from {spec.state_path}")
        # next, pipe iput of given shape through the network
        with torch.no_grad():
            try:
                input_ = torch.zeros(*([1] + list(spec.input_shape)), dtype=torch.float32, device=self.device)
                logger.info(f"Forward pass with tensor of size {input_.shape}.")
                out = model(input_)
                halo = tuple((i - o) // 2 for i, o in zip(tuple(input_[0, 0].shape), tuple(out[0, 0].shape)))
                logger.info(f"Model outputs tensors of size {out.shape} and has a halo of {halo}.")
            except:
                raise ValueError(f"Input of shape {spec.input_shape} invalid for model")

        return tuple(out[0].shape), halo

    def build(self, spec):
        """
        Build tiktorch configuration.

        Parameters
        ----------
        spec: TikTorchSpec
            Specification Object
        """
        logging.getLogger("BuildSpec.build")

        output_shape, halo_shape = self._validate_spec(spec)

        # Validate and copy code path
        self.validate_path(spec.code_path, "py").copy_to_build_directory(spec.code_path, "model.py")
        # ... and weights path
        self.copy_to_build_directory(spec.state_path, "state.nn")

        # Build and dump configuration dict
        tiktorch_config = {
            "input_shape": tuple(spec.input_shape),
            "output_shape": tuple(output_shape),
            "halo": tuple(halo_shape),
            "dynamic_input_shape": self._to_dynamic_shape(spec.minimal_increment),
            "model_class_name": spec.model_class_name,
            "model_init_kwargs": spec.model_init_kwargs,
            "torch_version": torch.__version__,
        }
        if spec.description is not None:
            tiktorch_config["description"] = spec.description
        if spec.data_source is not None:
            tiktorch_config["data_source"] = spec.data_source
        self.dump_config(tiktorch_config)
        # Done
        return self


class TikTorchSpec(object):
    def __init__(
        self,
        code_path=None,
        model_class_name=None,
        state_path=None,
        input_shape=None,
        minimal_increment=None,
        model_init_kwargs=None,
        description=None,
        data_source=None,
    ):
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
            Must be `HW` (for 2D inputs) or `DHW` (for 3D inputs).
        model_init_kwargs: dict
            Kwargs to the model constructor (if any).
        description: str
            description of the pre-trained mode, optional
        data_source: str
            url to data used for pre-session, optional
        """
        self.code_path = code_path
        self.model_class_name = model_class_name
        self.state_path = state_path
        self.input_shape = input_shape
        self.minimal_increment = minimal_increment
        self.model_init_kwargs = model_init_kwargs or {}
        self.description = description
        self.data_source = data_source

        self.validate()

    @classmethod
    def assert_(cls, condition, message="", exception_type=Exception):
        if not condition:
            raise exception_type(message)
        return cls

    def validate(self):
        # TODO in the long run we should support both ways of serializing a model:
        # https://pytorch.org/docs/master/notes/serialization.html
        self.assert_(os.path.exists(self.state_path), f"Path not found: {self.state_path}", FileExistsError)
        self.assert_(os.path.exists(self.code_path), f"Path not found: {self.code_path}", FileExistsError)
        self.assert_(isinstance(self.model_class_name, str), "Model Class Name must be a string", ValueError)

        # TODO why do we care if this is list, tuple or whatever ?
        # self.assert_(isinstance(self.input_shape, list), "input_shape must be a list", ValueError)

        ndim = len(self.input_shape)
        self.assert_(
            ndim in (3, 4), f"input_shape has length {len(self.input_shape)} but should have lenght 3 or 4", ValueError
        )
        self.assert_(
            ndim - 1 == len(self.minimal_increment), f"minimal increment must have 1 entry less than input shape"
        )

        self.assert_(isinstance(self.model_init_kwargs, dict), "model_init_kwargs must be a dictionary", ValueError)

        if self.description is not None:
            self.assert_(isinstance(self.description, str), "description must be a string", ValueError)
        if self.data_source is not None:
            self.assert_(isinstance(self.data_source, str), "data_source must be a string", ValueError)
        return self


def test_build_spec():
    spec = TikTorchSpec(
        code_path="/home/jo/CREMI_DUNet_pretrained/model.py",
        model_class_name="DUNet2D",
        state_path="/home/jo/CREMI_DUNet_pretrained/state.nn",
        input_shape=[1, 512, 512],
        minimal_increment=[32, 32],
        model_init_kwargs={"in_channels": 1, "out_channels": 1},
    )

    build = BuildSpec(build_directory="/home/jo/sfb1129/ilastik_debug/tiktorch_build")
    build.build(spec)


if __name__ == "__main__":
    test_build_spec()
