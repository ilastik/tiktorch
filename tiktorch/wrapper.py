import logging
import os
from importlib import util as imputils

import numpy as np
import torch
import yaml

from tiktorch.tio import TikIn, TikOut
import tiktorch.utils as utils
from tiktorch.device_handler import ModelHandler

logger = logging.getLogger("TikTorch")


class TikTorch(object):
    def __init__(self, build_directory, device=None):
        # Privates
        self._build_directory = None
        self._handler: ModelHandler = None
        self._model = None
        self._config = {}
        if device is None:
            # The default behaviour is to select a GPU if one is availabe.
            # This can be overriden by providing device in the constructor.
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        # Publics
        self.build_directory = build_directory
        self.read_config()

    @property
    def output_shape(self):
        return self.get("output_shape")

    @property
    def halo(self):
        """
        Returns the halo in dynamic base shape blocks
        """
        assert self.handler is not None
        halo_block = self.handler.halo_in_blocks
        base_shape = self.handler.dynamic_shape.base_shape
        return [shape * block for shape, block in zip(base_shape, halo_block)]

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
        return self.handler.model

    @property
    def device(self):
        return self.handler.device

    @property
    def handler(self):
        if self._handler is None:
            self.load_model()
        return self._handler

    def train(self, data, labels):
        assert self.handler is not None
        self.handler.train(data, labels)
        pass

    def dry_run(self, image_shape, train=False):
        """
        Initiates dry run.
        Parameters
        ----------
        image_shape: list or tuple
        shape of an image in the dataset (e.g `HW` for 2D or `DHW` for 3D)
        """
        assert self.handler is not None
        return self.handler.binary_dry_run(list(image_shape), train_flag=train)

    def read_config(self):
        config_file_name = os.path.join(self.build_directory, "tiktorch_config.yml")
        if not os.path.exists(config_file_name):
            raise FileNotFoundError(f"Config file not found in " f"build_directory: {self.build_directory}.")
        with open(config_file_name, "r") as f:
            self._config.update(yaml.load(f))
        return self

    def _set_handler(self, model):
        assert self.get("input_shape") is not None
        # Pass
        self._handler = ModelHandler(
            model=model,
            device_names=self._device,
            channels=self.get("input_shape")[0],
            dynamic_shape_code=self.get("dynamic_input_shape"),
        )

    def get(self, tag, default=None, assert_exist=False):
        if assert_exist:
            assert tag in self._config, f"Tag '{tag}' not found in configuration."
        return self._config.get(tag, default)

    def load_model(self):
        # Dynamically import file.
        model_file_name = os.path.join(self.build_directory, "model.py")
        module_spec = imputils.spec_from_file_location("model", model_file_name)
        module = imputils.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        # Build model from file
        model: torch.nn.Module = getattr(module, self.get("model_class_name"))(**self.get("model_init_kwargs"))
        # Load parameters
        state_path = os.path.join(self.build_directory, "state.nn")
        try:
            state_dict = torch.load(state_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except:
            raise FileNotFoundError(f"Model weights could not be found at location '{state_path}'!")
        # Build handler
        self._set_handler(model)
        return self

    def batch_inputs(self, inputs):
        input_shapes = self.get("input_shape", assert_exist=True)
        assert isinstance(input_shapes, (list, tuple))
        # input_shapes can either be a list of shapes or a shape. Make sure it's the latter
        if isinstance(input_shapes[0], int):
            input_shapes = [input_shapes] * len(inputs)
        elif isinstance(input_shapes[0], (list, tuple)):
            pass
        else:
            raise TypeError(
                f"`input_shapes` must be a list/tuple of ints or "
                f"lists/tuples or ints. Got list/tuple of {type(input_shapes[0])}."
            )
        utils.assert_(
            len(input_shapes) == len(inputs),
            f"Expecting {len(inputs)} inputs, got {len(input_shapes)} input shapes.",
            ValueError,
        )
        batches = [input.batcher(input_shape) for input, input_shape in zip(inputs, input_shapes)]
        return batches

    def parse_inputs(self, inputs):
        if isinstance(inputs, TikIn):
            inputs = [inputs]
        elif isinstance(inputs, (np.ndarray, torch.Tensor)):
            inputs = [TikIn([inputs])]
        elif isinstance(inputs, (list, tuple)):
            utils.assert_(all(isinstance(input, TikIn) for input in inputs), "Inputs must all be TikIn objects.")
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
        inputs = self.parse_inputs(TikIn(inputs))
        # Batch inputs
        batches = self.batch_inputs(inputs)
        # Send batch to the right device
        batches = [batch for batch in batches]
        # Make sure model is in right device and feedforward
        # throws an error if inputs is a TikIn list with more than 1 element!
        output_batches = self.handler.forward(*batches)
        return output_batches.numpy()


def test_full_pipeline():
    import h5py
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from inferno.io.transform import Compose
    from inferno.io.transform.generic import Normalize, Cast, AsTorchBatch

    # tiktorch = TikTorch('/export/home/jhugger/sfb1129/test_configs_tiktorch/config/')
    tiktorch = TikTorch("/home/jo/config/")

    # with h5py.File('/export/home/jhugger/sfb1129/sample_C_20160501.hdf') as f:
    with h5py.File("/home/jo/sfb1129/sample_C_20160501.hdf") as f:
        cremi_raw = f["volumes"]["raw"][:, 0:512, 0:512]

    transform = Compose(Normalize(), Cast("float32"))
    inputs = [transform(cremi_raw[i : i + 1]) for i in range(1)]

    halo = tiktorch.halo
    max_shape = tiktorch.dry_run([512, 512])

    print(f"Halo: {halo}")
    print(f"max_shape: {max_shape}")

    out = tiktorch.forward(inputs)

    return 0


def test_dunet():
    import h5py
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from inferno.io.transform import Compose
    from inferno.io.transform.generic import Normalize, Cast, AsTorchBatch

    # tiktorch = TikTorch('/export/home/jhugger/sfb1129/test_configs_tiktorch/config/')
    tiktorch = TikTorch("/home/jo/config/")

    # with h5py.File('/export/home/jhugger/sfb1129/sample_C_20160501.hdf') as f:
    with h5py.File("/home/jo/sfb1129/sample_C_20160501.hdf") as f:
        cremi_raw = f["volumes"]["raw"][:, 0:1024, 0:1024]

    transform = Compose(Normalize(), Cast("float32"))
    tikin_list = [TikIn([transform(cremi_raw[i : i + 1]) for i in range(1)])]
    inputs = [transform(cremi_raw[i : i + 1]) for i in range(2)]

    out = tiktorch.forward(inputs)
    return 0


if __name__ == "__main__":
    # test_TikTorch_init()
    # test_forward()
    # test_dunet()
    test_full_pipeline()
