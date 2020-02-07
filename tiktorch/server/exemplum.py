from dataclasses import dataclass
from typing import Any, Dict, Sequence, Union

import ignite
import torch

from pybio.spec import node
from pybio.spec.utils import get_instance


@dataclass
class Config:
    # max_iterations: int
    # batch_size: int
    # num_iterations_per_update: int
    warmstart: bool


class IterationOutput:
    prediction: Any


class InferenceOutput(IterationOutput):
    pass


# class ValidationOutput(IterationOutput):
#     pass
#
# class TrainingOutput(IterationOutput):
#     pass


class Exemplum:
    def __init__(self, pybio_model: node.Model, config: Union[Dict[str, Any], Config], devices: Sequence[torch.device]):
        spec = pybio_model.spec
        self.model = get_instance(pybio_model)
        if spec.framework == "pytorch":
            assert isinstance(self.model, torch.nn.Module)
            if config.warmstart:
                state = torch.load(spec.prediction.weights.source, map_location=devices[0])
                self.model.load_state_dict(state)
        else:
            raise NotImplementedError

        # inference_engine = ignite.engine.Engine(self._inference_step_function)
        # .add_event_handler(Events.STARTED, self.prepare_engine)
        # .add_event_handler(Events.COMPLETED, self.log_compute_time)

    #
    # def _inference_step_function(self, engine: ignite.engine.Engine, batch) -> InferenceOutput:
    #     prediction = self.model(batch)
    #     return InferenceOutput(prediction=prediction)

    # def _validation_step_function(self) -> ValidationOutput:
    #     return ValidationOutput()
    #
    #
    # def _training_step_function(self) -> TrainingOutput:
    #     return TrainingOutput()

    def forward(self, batch) -> InferenceOutput:
        prediction = self.model(batch)
        return InferenceOutput(prediction=prediction)

    @property
    def max_num_iterations(self):
        raise NotImplementedError

    def set_max_num_iterations(self):
        raise NotImplementedError

    def iteration_count(self):
        raise NotImplementedError

    def set_break_callback(self):
        return NotImplementedError
