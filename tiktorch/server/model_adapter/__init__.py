from typing import List

from pybio.spec import nodes

from ._base import ModelAdapter

__all__ = ["ModelAdapter", "create_model_adapter"]


def create_model_adapter(*, pybio_model: nodes.Model, devices=List[str]):
    spec = pybio_model
    weights = pybio_model.weights
    if "pytorch_state_dict" in weights:
        from ._pytorch_model_adapter import PytorchModelAdapter

        return PytorchModelAdapter(pybio_model=pybio_model, devices=devices)
    elif "tensorflow_saved_model_bundle" in weights:
        from ._tensorflow_model_adapter import TensorflowModelAdapter

        return TensorflowModelAdapter(pybio_model=pybio_model, devices=devices)
    elif "onnx" in weights:
        from ._onnx_model_adapter import ONNXModelAdapter

        return ONNXModelAdapter(pybio_model=pybio_model, devices=devices)
    elif "pytorch_script" in weights:
        from ._torchscript_model_adapter import TorchscriptModelAdapter

        return TorchscriptModelAdapter(pybio_model=pybio_model, devices=devices)
    else:
        raise NotImplementedError(f"No supported weight_formats in {spec.weights.keys()}")
