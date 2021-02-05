from typing import Callable, List

import numpy as np
import onnxruntime as rt
import tensorflow as tf
from pybio.spec import nodes
from pybio.spec.utils import get_instance

from ._base import ModelAdapter
from ._preprocessing import make_preprocessing
from ._utils import has_batch_dim


def _noop(tensor):
    return tensor


def _remove_batch_dim(batch: List):
    return [t.reshape(t.shape[1:]) for t in batch]


def _add_batch_dim(tensor):
    return tensor.reshape((1,) + tensor.shape)


class ONNXModelAdapter(ModelAdapter):
    class ONNXWrapper:
        def __init__(self, weights):
            self._session = rt.InferenceSession(weights)
            inputs = self._session.get_inputs()
            if len(inputs) != 1:
                raise ValueError("Only supports models with 1 input")
            self._input_name = inputs[0].name

        def forward(self, input):
            return self._session.run(None, {self._input_name: input})

    def __init__(
        self,
        *,
        pybio_model: nodes.Model,
        devices=List[str],
    ):
        spec = pybio_model
        self.name = spec.name

        if len(spec.inputs) != 1 or len(spec.outputs) != 1:
            raise NotImplementedError("Only single input, single output models are supported")

        assert len(spec.inputs) == 1
        assert len(spec.outputs) == 1

        _input = spec.inputs[0]
        _output = spec.outputs[0]

        self._internal_input_axes = _input.axes
        self._internal_output_axes = _output.axes

        if has_batch_dim(self._internal_input_axes):
            self.input_axes = self._internal_input_axes[1:]
            self._input_batch_dimension_transform = _noop
            _input_shape = _input.shape[1:]
        else:
            self.input_axes = self._internal_input_axes
            self._input_batch_dimension_transform = _noop
            _input_shape = _input.shape

        self.input_shape = list(zip(self.input_axes, _input_shape))

        _halo = _output.halo or [0 for _ in _output.axes]

        if has_batch_dim(self._internal_output_axes):
            self.output_axes = self._internal_output_axes[1:]
            self._output_batch_dimension_transform = _remove_batch_dim
            _halo = _halo[1:]
        else:
            self.output_axes = self._internal_output_axes
            self._output_batch_dimension_transform = _noop

        self._prediction_preprocess = make_preprocessing(_input.preprocessing)
        self._prediction_postprocess = _noop

        self.halo = list(zip(self.output_axes, _halo))
        self.model = self.ONNXWrapper(str(spec.weights["onnx"].source))
        self.devices = []

    def forward(self, batch):
        batch = self._input_batch_dimension_transform(batch)
        batch = self._prediction_preprocess(batch)
        batch = self.model.forward(batch)
        batch = self._prediction_postprocess(batch)
        batch = self._output_batch_dimension_transform(batch)
        return batch

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
