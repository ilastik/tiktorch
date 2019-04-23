from logging import Logger

from tiktorch.configkeys import CONFIG, MINIMAL_CONFIG
from tiktorch.rpc import RPCFuture
from tiktorch.tiktypes import TikTensor
from tiktorch.types import NDArray, PointBase, Point2D, Point3D, Point4D, SetDeviceReturnType
from tiktorch.configkeys import TRAINING, LOSS_CRITERION_CONFIG

from typing import Callable, Union, Tuple, List


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


def convert_points_to_5d_tuples(obj: Union[tuple, list, PointBase], missing_axis_value: int):
    if isinstance(obj, tuple):
        return tuple([convert_points_to_5d_tuples(p, missing_axis_value) for p in obj])
    elif isinstance(obj, list):
        return [convert_points_to_5d_tuples(p, missing_axis_value) for p in obj]
    elif isinstance(obj, PointBase):
        return tuple([getattr(obj, a) if a in obj.order else missing_axis_value for a in "tczyx"])
    else:
        return obj


def convert_to_SetDeviceReturnType(
    res: Union[
        Tuple[Point2D, List[Point2D], Point2D],
        Tuple[Point3D, List[Point3D], Point3D],
        Tuple[Point4D, List[Point4D], Point4D],
    ]
):
    return SetDeviceReturnType(
        convert_points_to_5d_tuples(res[0], 1),  # training shape needs singleton 1
        convert_points_to_5d_tuples(res[1], 1),  # valid shapes need singleton 1
        convert_points_to_5d_tuples(res[2], 0),  # shrinkage needs 0 values as it is a difference
    )


def add_logger(logger: Logger) -> Callable:
    def with_logging(target: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            logger.info("started")
            try:
                target(*args, **kwargs)
            except Exception as e:
                logger.exception(e)
            logger.info("stopped")

        return wrapper

    return with_logging
