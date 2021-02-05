from typing import List, Optional

from pybio.spec import nodes

from ._base import ModelAdapter

__all__ = ["ModelAdapter", "create_model_adapter"]


def _get_model_adapter(weight_format: str):
    if weight_format == "pytorch_state_dict":
        from ._exemplum import Exemplum
        return Exemplum
    elif weight_format == "tensorflow_saved_model_bundle":
        from ._tensorflow_model_adapter import TensorflowModelAdapter
        return TensorflowModelAdapter
    elif weight_format == "onnx":
        from ._onnx_model_adapter import ONNXModelAdapter
        return ONNXModelAdapter
    elif weight_format == "pytorch_script":
        from ._torchscript_model_adapter import TorchscriptModelAdapter
        return TorchscriptModelAdapter
    else:
        raise ValueError(f"Weight format {weight_format} is not supported.")


def create_model_adapter(*, pybio_model: nodes.Model, devices=List[str], weight_format: Optional[str] = None):
    weight_formats = [
        "pytorch_state_dict",
        "tensorflow_saved_model_bundle",
        "pytorch_script",
        "onnx"
    ]

    spec = pybio_model
    weights = pybio_model.weights

    if weight_format is not None:
        if weight_format not in weight_formats:
            raise ValueError(f"Weight format {weight_format} is not in supported formats {weight_formats}")
        weight_formats = [weight_format]

    for weight in weight_formats:
        if weight in weights:
            adapter = _get_model_adapter(weight)
            return adapter(pybio_model=pybio_model, devices=devices)

    raise NotImplementedError(f"No supported weight_formats in {spec.weights.keys()}")
