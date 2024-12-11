from typing import List

import torch

from tiktorch.converters import Sample
from tiktorch.rpc import RPCInterface, exposed
from tiktorch.rpc.exceptions import Shutdown
from tiktorch.tiktypes import TikTensorBatch
from tiktorch.trainer import TrainerState
from tiktorch.types import ModelState


class IRPCModelSession(RPCInterface):
    @exposed
    def init(self, model_bytes: bytes, devices: List[str]):
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


class IRPCTrainer(RPCInterface):
    @exposed
    def init(self, trainer_yaml_config: str):
        raise NotImplementedError

    @exposed
    def forward(self, input_tensors: List[torch.Tensor]):
        raise NotImplementedError

    @exposed
    def resume_training(self) -> None:
        raise NotImplementedError

    @exposed
    def pause_training(self) -> None:
        raise NotImplementedError

    @exposed
    def start_training(self) -> None:
        raise NotImplementedError

    @exposed
    def shutdown(self) -> Shutdown:
        raise NotImplementedError

    @exposed
    def save(self):
        raise NotImplementedError

    @exposed
    def export(self):
        raise NotImplementedError

    @exposed
    def get_state(self) -> TrainerState:
        raise NotImplementedError
