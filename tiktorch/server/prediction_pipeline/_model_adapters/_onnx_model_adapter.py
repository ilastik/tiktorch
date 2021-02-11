import logging
from typing import Callable, List

import numpy as np
import onnxruntime as rt
import xarray
from pybio.spec import nodes
from xarray import DataArray

from tiktorch.server.prediction_pipeline._model_adapters._model_adapter import ModelAdapter
from tiktorch.server.prediction_pipeline._preprocessing import make_preprocessing
from tiktorch.server.prediction_pipeline._utils import has_batch_dim

logger = logging.getLogger(__name__)


class ONNXModelAdapter(ModelAdapter):
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

        self._internal_output_axes = _output.axes

        self._session = rt.InferenceSession(str(spec.weights["onnx"].source))
        onnx_inputs = self._session.get_inputs()
        assert len(onnx_inputs) == 1, f"expected onnx model to have one input got {len(onnx_inputs)}"
        self._input_name = onnx_inputs[0].name
        self.devices = []

    def forward(self, input: DataArray) -> DataArray:
        result = self._session.run(None, {self._input_name: input.data})[0]
        return xarray.DataArray(result, dims=tuple(self._internal_output_axes))

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
