from logging import Logger

from tiktorch.configkeys import CONFIG, MINIMAL_CONFIG
from tiktorch.rpc import RPCFuture
from tiktorch.tiktypes import TikTensor
from tiktorch.types import NDArray
from tiktorch.configkeys import TRAINING, LOSS_CRITERION_CONFIG

from typing import Callable


def get_error_msg_for_invalid_config(config: dict) -> str:
    for key in config.keys():
        if key not in CONFIG:
            return f"Unknown config key={key}"

        if isinstance(CONFIG[key], dict):
            if not isinstance(config[key], dict):
                return f"config[key={key}] needs to be a dictionary"
            else:
                for subkey in config[key].keys():
                    if subkey not in CONFIG[key]:
                        return f"Unknown subkey={subkey}"
                    elif subkey == LOSS_CRITERION_CONFIG:
                        if "method" not in config[TRAINING][LOSS_CRITERION_CONFIG]:
                            return f"'method' entry missing in config[{TRAINING}][{LOSS_CRITERION_CONFIG}]"

    return ""


def get_error_msg_for_incomplete_config(config: dict) -> str:
    invalid_msg = get_error_msg_for_invalid_config(config)
    if invalid_msg:
        return invalid_msg

    for key in MINIMAL_CONFIG.keys():
        if key not in config:
            return f"Missing key={key}"

        if isinstance(MINIMAL_CONFIG[key], dict):
            assert isinstance(config[key], dict), "should have been checked by 'get_error_msg_for_invalid_config'"
            for subkey in MINIMAL_CONFIG[key].keys():
                if subkey not in config[key]:
                    return f"Missing subkey={subkey} in config[key={key}]"

    return ""


def convert_tik_fut_to_ndarray_fut(tik_fut: RPCFuture[TikTensor]) -> RPCFuture[NDArray]:
    ndarray_fut = RPCFuture()

    def convert(tik_fut: RPCFuture[TikTensor]):
        try:
            res = tik_fut.result()
        except Exception as e:
            ndarray_fut.set_exception(e)
        else:
            ndarray_fut.set_result(NDArray(res.as_numpy()))

    tik_fut.add_done_callback(convert)
    return ndarray_fut


def add_logger(logger: Logger) -> Callable:
    def with_logging(target: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            logger.info("started")
            try:
                target(*args, **kwargs)
            except Exception as e:
                logger.error(e)
            logger.info("stopped")

        return wrapper

    return with_logging
