import logging
from typing import Any, List, Sequence

import torch
#from pybio.core.transformations.base import make_concatenated_apply
from pybio.spec import nodes
from pybio.spec.utils import get_instance

logger = logging.getLogger(__name__)
# @dataclass
# class ValidationOutput(IterationOutput):
#     pass

# @dataclass
# class TrainingOutput(IterationOutput):
#     pass


def _noop(tensor):
    return tensor


def _remove_batch_dim(batch: List):
    return [t.reshape(t.shape[1:]) for t in batch]


def _add_batch_dim(tensor):
    return tensor.reshape((1,) + tensor.shape)


def _check_batch_dim(axes: str) -> bool:
    try:
        index = axes.index("b")
    except ValueError:
        return False
    else:
        if index != 0:
            raise ValueError("Batch dimension is only supported in first position")
        return True


def _make_cast(dtype):
    def _cast(tensor):
        return tensor.astype(dtype)
    return _cast


def _to_torch(tensor):
    return torch.from_numpy(tensor)


def _zero_mean_unit_variance(tensor, eps=1.0e-6):
    mean, std = tensor.mean(), tensor.std()
    return (tensor - mean) / (std + 1.0e-6)


def _sigmoid(tensor):
    return torch.sigmoid(tensor)


KNOWN_PREPROCESSING = {
    "zero_mean_unit_variance": _zero_mean_unit_variance,
}

KNOWN_POSTPROCESSING = {
    "sigmoid": _sigmoid,
}

def chain(*functions):
    def _chained_function(tensor):
        tensor = tensor
        for fn in functions:
            tensor = fn(tensor)

        return tensor

    return _chained_function


class Exemplum:
    def __init__(
        self,
        *,
        pybio_model: nodes.Model,
        batch_size: int = 1,
        num_iterations_per_update: int = 2,
        _devices=Sequence[torch.device],
    ):
        self.max_num_iterations = 0
        self.iteration_count = 0
        self.devices = _devices
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

        if _check_batch_dim(self._internal_input_axes):
            self.input_axes = self._internal_input_axes[1:]
            self._input_batch_dimension_transform = _add_batch_dim
            _input_shape = _input.shape[1:]
        else:
            self.input_axes = self._internal_input_axes
            self._input_batch_dimension_transform = _noop
            _input_shape = _input.shape

        self.input_shape = list(zip(self.input_axes, _input_shape))

        _halo = _output.halo or [0 for _ in _output.axes]

        if _check_batch_dim(self._internal_output_axes):
            self.output_axes = self._internal_output_axes[1:]
            self._output_batch_dimension_transform = _remove_batch_dim
            _halo = _halo[1:]
        else:
            self.output_axes = self._internal_output_axes
            self._output_batch_dimension_transform = _noop

        self.halo = list(zip(self.output_axes, _halo))

        self.model = get_instance(pybio_model)
        self.model.to(self.devices[0])
        if spec.framework == "pytorch":
            assert isinstance(self.model, torch.nn.Module)
            weights = spec.weights.get("pytorch_state_dict")
            if weights is not None and weights.source:
                state = torch.load(weights.source, map_location=self.devices[0])
                self.model.load_state_dict(state)
        else:
            raise NotImplementedError

        preprocessing_functions = [
            _make_cast(_input.data_type),
            _to_torch,
        ]
        
        for preprocessing_step in _input.preprocessing:
            fn = KNOWN_PREPROCESSING.get(preprocessing_step.name)
            if fn is None:
                raise NotImplementedError(f"Preprocessing {preprocessing_step.name}")

            preprocessing_functions.append(fn)

        self._prediction_preprocess = chain(*preprocessing_functions)

        postprocessing_functions = []
        for postprocessing_step in _output.postprocessing:
            fn = KNOWN_POSTPROCESSING.get(postprocessing_step.name)
            if fn is None:
                raise NotImplementedError(f"Postprocessing {postprocessing_step.name}")

            postprocessing_functions.append(fn)

        self._prediction_postprocess = chain(*postprocessing_functions)

    def forward(self, batch) -> List[Any]:
        with torch.no_grad():
            batch = self._input_batch_dimension_transform(batch)
            batch = self._prediction_preprocess(batch)
            batch = [b.to(self.devices[0]) for b in batch]
            batch = self.model(*batch)
            batch = self._prediction_postprocess(batch)
            batch = self._output_batch_dimension_transform(batch)
            assert all([bs > 0 for bs in batch[0].shape]), batch[0].shape
            return batch[0]

    def set_max_num_iterations(self, max_num_iterations: int) -> None:
        self.max_num_iterations = max_num_iterations

    def set_break_callback(self, cb):
        return NotImplementedError

    def fit(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
