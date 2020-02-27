import logging
import time
from typing import Generator, Iterable, List, Optional, Tuple, Union

from tiktorch.configkeys import MINIMAL_CONFIG
from tiktorch.rpc import RPCFuture, TCPConnConf, Timeout
from tiktorch.rpc_interface import IFlightControl, INeuralNetworkAPI
from tiktorch.types import LabeledNDArrayBatch, NDArray, SetDeviceReturnType

logger = logging.getLogger(__name__)


class DummyServerForFrontendDev(INeuralNetworkAPI, IFlightControl):
    RANK = 1
    SIZE = 2

    def __init__(self) -> None:
        self._last_ping = None
        logger.info("started")

    def ping(self) -> bytes:
        self._last_ping = time.time()
        return b"pong"

    def last_ping(self):
        return self._last_ping

    def shutdown(self) -> None:
        logger.info("stopped")

    def load_model(
        self, config: dict, model_file: bytes, model_state: bytes, optimizer_state: bytes, devices: list
    ) -> RPCFuture[SetDeviceReturnType]:
        logger.info("load model")
        logger.debug(
            "config %s\n model_file %s\n model_state %s\n optimizer_state %s\n devices %s\n",
            config,
            model_file,
            model_state,
            optimizer_state,
            devices,
        )
        fut = RPCFuture()
        ret = SetDeviceReturnType((1, 1, 1, 10, 10), [(1, 1, 1, 10, 10), (1, 1, 1, 20, 20)], (0, 0, 0, 4, 4))
        fut.set_result(ret)
        return fut

    def update_config(self, partial_config: dict) -> None:
        logger.info("update config with %s", partial_config)

    def forward(self, batch: NDArray) -> RPCFuture[NDArray]:
        logger.info("forward for array of shape %s", batch.shape)
        fut = RPCFuture()
        fut.set_result(batch)
        return fut

    def pause_training(self) -> None:
        logger.info("pause session")

    def resume_training(self) -> None:
        logger.info("resume session")

    def update_training_data(self, data: LabeledNDArrayBatch) -> None:
        logger.info("update session data with batch of length %d", len(data))

    def update_validation_data(self, data: LabeledNDArrayBatch) -> None:
        logger.info("update validation data with batch of length %d", len(data))

    def get_available_devices(self) -> List[Tuple[str, str]]:
        logger.info("get available devices")
        return [("cpu", "CPU (description)")]

    def is_valid_device_name(self, device_name: str) -> bool:
        logger.info("is_valid_device_name got device_name=%s", device_name)
        return device_name == "cpu"

    def active_children(self) -> list:
        active_children = ["Training", "Inference", "DryRun"]
        logger.info("pretend to have active children %s", active_children)
        return active_children

    def log(self, msg: str) -> None:
        logger.info("log msg: %s", msg)
