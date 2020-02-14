from typing import List

import torch

from tiktorch.rpc import RPCInterface, RPCFuture, exposed, Shutdown
from tiktorch.tiktypes import TikTensorBatch
from tiktorch.types import ModelState


class ITraining(RPCInterface):
    @exposed
    def set_devices(self, devices: List[torch.device]) -> List[torch.device]:
        raise NotImplementedError

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
    def wait_for_idle(self) -> RPCFuture:
        raise NotImplementedError

    @exposed
    def forward(self, input_tensor):
        raise NotImplementedError

    @exposed
    def get_model_info(self):
        raise NotImplementedError
