from tiktorch.rpc import RPCInterface, exposed
from tiktorch.types import NDArray, NDArrayBatch

from typing import List, Tuple

class IFlightControl(RPCInterface):
    @exposed
    def ping(self) -> bytes:
        raise NotImplementedError

    @exposed
    def shutdown(self) -> None:
        raise NotImplementedError


class INeuralNetworkAPI(RPCInterface):
    @exposed
    def load_model(
        self, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes, devices: list
    ) -> None:
        raise NotImplementedError

    @exposed
    def set_hparams(self, params: dict) -> None:
        raise NotImplementedError

    @exposed
    def forward(self, batch: NDArray) -> NDArray:
        raise NotImplementedError

    @exposed
    def pause(self) -> None:
        raise NotImplementedError

    @exposed
    def resume(self) -> None:
        raise NotImplementedError

    @exposed
    def train(self, data: NDArrayBatch, labels: NDArrayBatch) -> None:
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
