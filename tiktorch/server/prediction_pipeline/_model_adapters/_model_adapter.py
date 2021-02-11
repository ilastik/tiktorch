import abc
from typing import Callable, List

import xarray
from pybio.spec import nodes


class ModelAdapter(abc.ABC):
    """
    Represents model *without* any preprocessing and postprocessing
    """

    @abc.abstractmethod
    def forward(self, input_tensor: xarray.DataArray) -> xarray.DataArray:
        ...

    @property
    @abc.abstractmethod
    def max_num_iterations(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def iteration_count(self) -> int:
        ...

    @abc.abstractmethod
    def set_break_callback(self, thunk: Callable[[], bool]) -> None:
        ...

    @abc.abstractmethod
    def set_max_num_iterations(self, val: int) -> None:
        ...


def create_model_adapter(*, pybio_model: nodes.Model, devices=List[str]) -> ModelAdapter:
    """
    Creates model adapter based on the passed spec
    Note: All specific adapters should happen inside this function to prevent different framework
    initializations interfering with each other
    """
    spec = pybio_model
    weights = pybio_model.weights
    if "pytorch_state_dict" in weights:
        from tiktorch.server.prediction_pipeline._model_adapters._pytorch_model_adapter import PytorchModelAdapter

        return PytorchModelAdapter(pybio_model=pybio_model, devices=devices)

    elif "tensorflow_saved_model_bundle" in weights:
        from tiktorch.server.prediction_pipeline._model_adapters._tensorflow_model_adapter import TensorflowModelAdapter

        return TensorflowModelAdapter(pybio_model=pybio_model, devices=devices)

    elif "onnx" in weights:
        from tiktorch.server.prediction_pipeline._model_adapters._onnx_model_adapter import ONNXModelAdapter

        return ONNXModelAdapter(pybio_model=pybio_model, devices=devices)

    elif "pytorch_script" in weights:
        from tiktorch.server.prediction_pipeline._model_adapters._torchscript_model_adapter import (
            TorchscriptModelAdapter,
        )

        return TorchscriptModelAdapter(pybio_model=pybio_model, devices=devices)

    else:
        raise NotImplementedError(f"No supported weight_formats in {spec.weights.keys()}")
