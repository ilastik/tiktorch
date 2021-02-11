import logging
from typing import Any, List, Sequence

import torch
from pybio.spec import nodes
from pybio.spec.utils import get_instance
from xarray import DataArray

from ._model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class PytorchModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        pybio_model: nodes.Model,
        devices=Sequence[str],
    ):
        self._max_num_iterations = 0
        self._iteration_count = 0
        self._internal_output_axes = pybio_model.outputs[0].axes
        spec = pybio_model
        self.model = get_instance(pybio_model)
        self.devices = [torch.device(d) for d in devices]
        self.model.to(self.devices[0])
        assert isinstance(self.model, torch.nn.Module)
        weights = spec.weights.get("pytorch_state_dict")
        if weights is not None and weights.source:
            state = torch.load(weights.source, map_location=self.devices[0])
            self.model.load_state_dict(state)

    @property
    def max_num_iterations(self) -> int:
        return self._max_num_iterations

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    def forward(self, input_tensor: DataArray) -> DataArray:
        with torch.no_grad():
            tensor = torch.from_numpy(input_tensor.data)
            tensor = tensor.to(self.devices[0])
            result = self.model(*[tensor])
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()

        return DataArray(result, dims=tuple(self._internal_output_axes))

    def set_max_num_iterations(self, max_num_iterations: int) -> None:
        self._max_num_iterations = max_num_iterations

    def set_break_callback(self, cb):
        return NotImplementedError

    def fit(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
