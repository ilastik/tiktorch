from typing import Generic, List, TypeVar

from tiktorch.converters import Sample
from tiktorch.rpc import RPCInterface, Shutdown, exposed
from tiktorch.tiktypes import TikTensorBatch
from tiktorch.types import ModelState

ModelType = TypeVar("ModelType")


class IRPCModelSession(RPCInterface, Generic[ModelType]):
    def __init__(self, model: ModelType):
        super().__init__()
        self._model = model

    @property
    def model(self):
        return self._model

    @exposed
    def shutdown(self) -> Shutdown:
        raise NotImplementedError

    @exposed
    def resume_training(self) -> None:
        raise NotImplementedError

    @exposed
    def pause_training(self) -> None:
        raise NotImplementedError

    @exposed
    def get_idle(self) -> bool:
        raise NotImplementedError

    @exposed
    def update_dataset(self, name: str, data: TikTensorBatch, labels: TikTensorBatch) -> None:
        raise NotImplementedError

    @exposed
    def update_config(self, partial_config: dict) -> None:
        raise NotImplementedError

    @exposed
    def get_state(self) -> ModelState:
        raise NotImplementedError

    @exposed
    def remove_data(self, name: str, ids: List[str]) -> None:
        raise NotImplementedError

    @exposed
    def create_dataset_description(self, mean, stddev) -> str:
        raise NotImplementedError

    @exposed
    def forward(self, input_tensors: Sample):
        raise NotImplementedError
