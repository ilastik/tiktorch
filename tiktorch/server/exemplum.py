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
    def __init__(self, pybio_model: node.Model, config: Union[Dict[str, Any], Config]):
        self.devices = [torch.device("cpu")]
        spec = pybio_model.spec
        self.model = get_instance(pybio_model)
        self.model.to(self.devices[0])
        if spec.framework == "pytorch":
            assert isinstance(self.model, torch.nn.Module)
            if config.warmstart:
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

    @property
    def max_num_iterations(self) -> int:
        return 0

    def set_max_num_iterations(self, max_num_iterations: int) -> None:
        return None

    @property
    def iteration_count(self) -> int:
        return 0

    def set_break_callback(self):
        return NotImplementedError

    def set_devices(self, devices: Sequence[torch.device]) -> None:
        main_device = devices[0]
        self.model = self.model.to(main_device)
        self.devices = devices

    def fit(self):
        return None
