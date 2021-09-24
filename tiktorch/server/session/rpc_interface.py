from typing import List

from tiktorch.rpc import RPCInterface, Shutdown, exposed
from tiktorch.tiktypes import TikTensorBatch
from tiktorch.types import ModelState


class IRPCModelSession(RPCInterface):
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
    def forward(self, input_tensors):
        raise NotImplementedError

    @exposed
    def get_model_info(self):
        raise NotImplementedError
