import abc
from typing import Callable, List

import xarray as xr
from pybio.spec import nodes


class ModelAdapter(abc.ABC):
    """
    Represents model *without* any preprocessing and postprocessing
    """

    @abc.abstractmethod
    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
        """
        Run forward pass of model to get model predictions
        Note: model is responsible converting it's data representation to
        xarray.DataArray
        """
        ...

    # Training methods
    @property
    def max_num_iterations(self) -> int:
        return 0

    @property
    def iteration_count(self) -> int:
        return 0

    def set_break_callback(self, thunk: Callable[[], bool]) -> None:
        pass

    def set_max_num_iterations(self, val: int) -> None:
        pass


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
