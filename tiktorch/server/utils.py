from inferno.io.transform import (
    Transform,
    generic as generic_transforms,
    image as image_transforms,
    volume as volume_transforms,
)

from . import tobeimported


def get_transform(name: str, **transform_kwargs) -> Transform:
    for module in [generic_transforms, image_transforms, volume_transforms, tobeimported]:
        ret = getattr(module, name, None)
        if ret is not None:
            try:
                return ret(**transform_kwargs)
            except Exception as e:
                logger.exception(e)
                raise ValueError(f"Transform {name} could not be initialized with kwargs {transform_kwargs}")

    raise NotImplementedError(f"Tranformation {name} could not be found")
