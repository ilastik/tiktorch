from ._base import ModelAdapter
from pybio.spec import nodes
from typing import List

__all__ = ["ModelAdapter", "create_model_adapter"]


def create_model_adapter(*, pybio_model: nodes.Model, devices = List[str]):
    spec = pybio_model.spec
    if spec.framework == "pytorch":
        from ._exemplum import Exemplum
        return Exemplum(pybio_model=pybio_model, devices=devices)
