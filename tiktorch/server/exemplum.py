from dataclasses import dataclass
from typing import Any, Sequence

import torch

from pybio.spec import nodes
from pybio.spec.utils import get_instance


@dataclass
class IterationOutput:
    prediction: Any


@dataclass
class InferenceOutput(IterationOutput):
    pass


# @dataclass
# class ValidationOutput(IterationOutput):
#     pass

# @dataclass
# class TrainingOutput(IterationOutput):
#     pass


class Exemplum:
    def __init__(
        self,
        *,
        pybio_model: nodes.Model,
        warmstart: bool = False,
        batch_size: int = 1,
        num_iterations_per_update: int = 2,
        _devices=Sequence[torch.device],
    ):
        self.max_num_iterations = 0
        self.iteration_count = 0
        self.devices = _devices
        spec = pybio_model.spec
        self.model = get_instance(pybio_model)
        self.model.to(self.devices[0])
        if spec.framework == "pytorch":
            assert isinstance(self.model, torch.nn.Module)
            if warmstart:
                state = torch.load(spec.prediction.weights.source, map_location=self.devices[0])
                self.model.load_state_dict(state)
        else:
            raise NotImplementedError

        # inference_engine = ignite.engine.Engine(self._inference_step_function)
        # .add_event_handler(Events.STARTED, self.prepare_engine)
        # .add_event_handler(Events.COMPLETED, self.log_compute_time)

    def _inference_step_function(self, batch) -> InferenceOutput:
        prediction = self.model(batch.to(self.devices[0]))
        return InferenceOutput(prediction=prediction)

    # def _validation_step_function(self) -> ValidationOutput:
    #     return ValidationOutput()
    #
    #
    # def _training_step_function(self) -> TrainingOutput:
    #     return TrainingOutput()

    def forward(self, batch) -> InferenceOutput:
        out = self._inference_step_function(batch)
        return out.prediction

    def set_max_num_iterations(self, max_num_iterations: int) -> None:
        self.max_num_iterations = max_num_iterations

    def set_break_callback(self, cb):
        return NotImplementedError

    def fit(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
