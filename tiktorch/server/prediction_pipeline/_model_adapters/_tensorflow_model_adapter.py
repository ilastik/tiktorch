from typing import Callable, List

import numpy as np
import tensorflow as tf
from pybio.spec import nodes
from pybio.spec.utils import get_instance
from xarray import DataArray

from ._model_adapter import ModelAdapter


class TensorflowModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        pybio_model: nodes.Model,
        devices=List[str],
    ):
        spec = pybio_model
        self.name = spec.name

        _input = spec.inputs[0]
        _output = spec.outputs[0]
        # FIXME: TF probably uses different axis names
        self._internal_output_axes = _output.axes

        self.model = get_instance(pybio_model)
        self.devices = []
        tf_model = tf.keras.models.load_model(spec.weights["tensorflow_saved_model_bundle"].source)
        self.model.set_model(tf_model)

    def forward(self, input_tensor):
        tf_tensor = tf.convert_to_tensor(input_tensor.data)

        res = self.model.forward(tf_tensor)

        if not isinstance(res, np.ndarray):
            res = tf.make_ndarray(res)

        return DataArray(res, dims=tuple(self._internal_output_axes))

    @property
    def max_num_iterations(self) -> int:
        return 0

    @property
    def iteration_count(self) -> int:
        return 0

    def set_break_callback(self, thunk: Callable[[], bool]) -> None:
        pass

    def set_max_num_iterations(self, val: int) -> None:
        pass
