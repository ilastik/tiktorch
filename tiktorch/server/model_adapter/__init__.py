from typing import List

from pybio.spec import nodes

from ._base import ModelAdapter

__all__ = ["ModelAdapter", "create_model_adapter"]


def create_model_adapter(*, pybio_model: nodes.Model, devices=List[str]):
    spec = pybio_model
    if spec.framework == "pytorch":
        from ._exemplum import Exemplum

        return Exemplum(pybio_model=pybio_model, devices=devices)
    elif spec.framework == "tensorflow":
        from ._tensorflow_model_adapter import TensorflowModelAdapter

        return TensorflowModelAdapter(pybio_model=pybio_model, devices=devices)
    else:
        raise NotImplementedError(f"Unknown framework: {spec.framework}")
