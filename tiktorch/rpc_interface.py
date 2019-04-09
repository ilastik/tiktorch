from typing import List, Tuple, Union

from tiktorch.rpc import RPCInterface, exposed, RPCFuture
from tiktorch.types import NDArray, LabeledNDArrayBatch
from tiktorch.tiktypes import Point2D, Point3D, Point4D


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
    ) -> RPCFuture[
        Union[
            Tuple[Point2D, List[Point2D], Point2D],
            Tuple[Point3D, List[Point3D], Point3D],
            Tuple[Point4D, List[Point4D], Point4D],
        ]
    ]:
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
    def update_training_data(self, data: LabeledNDArrayBatch) -> None:
        raise NotImplementedError

    @exposed
    def update_validation_data(self, data: LabeledNDArrayBatch) -> None:
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
