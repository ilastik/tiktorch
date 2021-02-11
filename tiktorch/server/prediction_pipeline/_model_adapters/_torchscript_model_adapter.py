from typing import Callable, List

import numpy as np
import torch
import xarray
from pybio.spec import nodes

from ._model_adapter import ModelAdapter


class TorchscriptModelAdapter(ModelAdapter):
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

        self._internal_input_axes = _input.axes
        self._internal_output_axes = _output.axes

        self.devices = devices
        weight_path = str(spec.weights["pytorch_script"].source.resolve())
        self.model = torch.jit.load(weight_path)

    def forward(self, batch):
        with torch.no_grad():
            torch_tensor = torch.from_numpy(batch.data)
            result = self.model.forward(torch_tensor)

            if not isinstance(result, np.ndarray):
                result = result.cpu().numpy()

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
