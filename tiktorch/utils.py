import logging

from tiktorch.configkeys import CONFIG, MINIMAL_CONFIG
from tiktorch.types import Point, SetDeviceReturnType
from tiktorch.configkeys import TRAINING, LOSS_CRITERION_CONFIG

from inferno.io.transform import (
    Transform,
    Compose,
    generic as generic_transforms,
    image as image_transforms,
    volume as volume_transforms,
)

from typing import Callable, Union, Tuple, List

logger = logging.getLogger()


def get_error_msg_for_invalid_config(config: dict) -> str:
    for key in config.keys():
        if key not in CONFIG:
            logger.warning("Encountered unknown config key: %s", key)
            # return f"Unknown config key={key}"
        elif isinstance(CONFIG[key], dict):
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


def convert_points_to_5d_tuples(obj: Union[tuple, list, Point], missing_axis_value: int):
    if isinstance(obj, tuple):
        return tuple([convert_points_to_5d_tuples(p, missing_axis_value) for p in obj])
    elif isinstance(obj, list):
        return [convert_points_to_5d_tuples(p, missing_axis_value) for p in obj]
    elif isinstance(obj, Point):
        return tuple(obj)
    else:
        return obj


def convert_to_SetDeviceReturnType(res: Tuple[Point, List[Point], Point]):
    return SetDeviceReturnType(
        convert_points_to_5d_tuples(res[0], 1),  # training shape needs singleton 1
        convert_points_to_5d_tuples(res[1], 1),  # valid shapes need singleton 1
        convert_points_to_5d_tuples(res[2], 0),  # shrinkage needs 0 values as it is a difference
    )


def add_logger(logger: logging.Logger) -> Callable:
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


def get_transform(name: str, **transform_kwargs) -> Transform:
    for module in [generic_transforms, image_transforms, volume_transforms]:
        ret = getattr(module, name, None)
        if ret is not None:
            try:
                return ret(**transform_kwargs)
            except Exception as e:
                logger.exception(e)
                raise ValueError(f"Transform {name} could not be initialized with kwargs {transform_kwargs}")

    raise NotImplementedError(f"Tranformation {name} could not be found")
