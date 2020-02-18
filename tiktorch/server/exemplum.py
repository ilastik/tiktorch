from dataclasses import dataclass
from typing import Any, Sequence, List

import numpy
import torch

from pybio.core.transformations.base import ConcatenatedPyBioTransformation
from pybio.spec import nodes
from pybio.spec.utils import get_instance


@dataclass
class IterationOutput:
    network_output: List[Any]


@dataclass
class InferenceOutput(IterationOutput):
    pass


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
        warmstart: bool = False,
        batch_size: int = 1,
        max_num_iterations: int = 0,
        num_iterations_per_update: int = 2,
        _devices=Sequence[torch.device],
    ):
        self.batch_size = batch_size
        self.max_num_iterations = max_num_iterations
        self.num_iterations_per_update = num_iterations_per_update
        self.iteration_count = 0
        self.devices = _devices
        spec = pybio_model.spec
        self.name = spec.name

        if len(spec.inputs) != 1 or len(spec.outputs) != 1:
            raise NotImplementedError("Only single input, single output models are supported")

        _input = spec.inputs[0]
        _output = spec.outputs[0]

        self._internal_input_axes = _input.axes
        self._internal_output_axes = _output.axes

        if _check_batch_dim(self._internal_input_axes):
            self.input_axes = self._internal_input_axes[1:]
            self._input_transform = _add_batch_dim
            _input_shape = _input.shape[1:]
        else:
            self.input_axes = self._internal_input_axes
            self._input_transform = _noop
            _input_shape = _input.shape

        self.input_shape = list(zip(self.input_axes, _input_shape))

        _halo = _output.halo or [0 for _ in _output.axes]

        if _check_batch_dim(self._internal_output_axes):
            self.output_axes = self._internal_output_axes[1:]
            self._output_transform = _remove_batch_dim
            _halo = _halo[1:]
        else:
            self.output_axes = self._internal_output_axes
            self._output_transform = _noop

        self.halo = list(zip(self.output_axes, _halo))

        self.model = get_instance(pybio_model)
        self.model.to(self.devices[0])
        if spec.framework == "pytorch":
            assert isinstance(self.model, torch.nn.Module)
            if warmstart:
                state = torch.load(spec.prediction.weights.source, map_location=self.devices[0])
                self.model.load_state_dict(state)
        else:
            raise NotImplementedError

        self._prediction_preprocess = ConcatenatedPyBioTransformation(
            [get_instance(trf) for trf in spec.prediction.preprocess]
        ).apply
        self._prediction_postprocess = ConcatenatedPyBioTransformation(
            [get_instance(trf) for trf in spec.prediction.postprocess]
        ).apply
        # inference_engine = ignite.engine.Engine(self._inference_step_function)
        # .add_event_handler(Events.STARTED, self.prepare_engine)
        # .add_event_handler(Events.COMPLETED, self.log_compute_time)

    def _inference_step_function(self, batch) -> InferenceOutput:
        assert all(isinstance(b, torch.Tensor) for b in batch)
        batch = [b.to(self.devices[0]) for b in batch]
        network_output = self.model(*batch)
        if not isinstance(network_output, (list, tuple)):
            network_output = [network_output]

        return InferenceOutput(network_output=network_output)

    # def _validation_step_function(self) -> ValidationOutput:
    #     return ValidationOutput()
    #
    #
    # def _training_step_function(self) -> TrainingOutput:
    #     return TrainingOutput()

    def forward(self, *batch) -> List[Any]:
        with torch.no_grad():
            batch = self._prediction_preprocess(*batch)
            out = self._inference_step_function(batch)
            batch = out.network_output
            batch = self._prediction_postprocess(*batch)
            return batch

    def set_max_num_iterations(self, max_num_iterations: int) -> None:
        self.max_num_iterations = max_num_iterations

    def set_break_callback(self, cb):
        return NotImplementedError

    def fit(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
