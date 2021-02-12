from typing import List, Tuple

from tiktorch.rpc import RPCFuture, RPCInterface, exposed
from tiktorch.types import Model, ModelState, NDArray, NDArrayBatch, SetDeviceReturnType


class IFlightControl(RPCInterface):
    @exposed
    def ping(self) -> bytes:
        raise NotImplementedError

    @exposed
    def last_ping(self) -> float:
        raise NotImplementedError

    @exposed
    def shutdown(self) -> None:
        raise NotImplementedError


class INeuralNetworkAPI(RPCInterface):
    @exposed
    def load_model(self, model: Model, state: ModelState, devices: list) -> RPCFuture[SetDeviceReturnType]:
        raise NotImplementedError

    @exposed
    def update_config(self, partial_config: dict) -> None:
        raise NotImplementedError

    # inference
    @exposed
    def forward(self, batch: NDArray) -> RPCFuture[NDArray]:
        raise NotImplementedError

    # training
    @exposed
    def pause_training(self) -> None:
        raise NotImplementedError

    @exposed
    def resume_training(self) -> None:
        raise NotImplementedError

    @exposed
    def update_training_data(self, data: NDArrayBatch, labels: NDArrayBatch) -> None:
        raise NotImplementedError

    @exposed
    def update_validation_data(self, data: NDArrayBatch, labels: NDArrayBatch) -> None:
        raise NotImplementedError

    # for information
    @exposed
    def get_available_devices(self) -> List[Tuple[str, str]]:
        raise NotImplementedError

    @exposed
    def is_valid_device_name(self, device_name: str) -> bool:
        raise NotImplementedError

    # for debugging
    @exposed
    def active_children(self) -> list:
        raise NotImplementedError

    @exposed
    def log(self, msg: str) -> None:
        raise NotImplementedError

    @exposed
    def get_model_state(self) -> ModelState:
        raise NotImplementedError

    @exposed
    def remove_data(self, dataset_name: str, ids: List[str]) -> None:
        raise NotImplementedError
