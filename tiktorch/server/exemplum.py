import time
from typing import Any, Sequence, List

import torch

from pybio.core.transformations.base import make_concatenated_apply
from pybio.spec import nodes
from pybio.spec.utils import get_instance


# @dataclass
# class ValidationOutput(IterationOutput):
#     pass

# @dataclass
# class TrainingOutput(IterationOutput):
#     pass


def _noop(tensor):
    return tensor


def _remove_batch_dim(tensor):
    return tensor.reshape(tensor.shape[1:])


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
        spec = pybio_model.spec
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
            if spec.prediction.weights is not None:
                state = torch.load(spec.prediction.weights.source, map_location=self.devices[0])
                self.model.load_state_dict(state)
        else:
            raise NotImplementedError

        self._prediction_preprocess = make_concatenated_apply([get_instance(tf) for tf in spec.prediction.preprocess])
        self._prediction_postprocess = make_concatenated_apply([get_instance(tf) for tf in spec.prediction.postprocess])
        # inference_engine = ignite.engine.Engine(self._inference_step_function)
        # .add_event_handler(Events.STARTED, self.prepare_engine)
        # .add_event_handler(Events.COMPLETED, self.log_compute_time)

    # def _validation_step_function(self) -> ValidationOutput:
    #     return ValidationOutput()
    #
    #
    # def _training_step_function(self) -> TrainingOutput:
    #     return TrainingOutput()

    def forward(self, batch) -> List[Any]:
        with torch.no_grad():
            batch = self._input_batch_dimension_transform(batch)
            batch = self._prediction_preprocess(batch)
            batch = [b.to(self.devices[0]) for b in batch]
            batch = self.model(*batch)
            batch = self._output_batch_dimension_transform(batch)
            batch = self._prediction_postprocess(batch)
            return batch[0]

    def set_max_num_iterations(self, max_num_iterations: int) -> None:
        self.max_num_iterations = max_num_iterations

    def set_break_callback(self, cb):
        return NotImplementedError

    def fit(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
