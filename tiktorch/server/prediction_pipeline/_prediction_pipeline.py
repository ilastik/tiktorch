import abc
from typing import Callable, List, Tuple

import xarray as xr
from pybio.spec import nodes

from ._model_adapters import ModelAdapter, create_model_adapter
from ._postprocessing import REMOVE_BATCH_DIM, make_postprocessing
from ._preprocessing import ADD_BATCH_DIM, make_ensure_dtype_preprocessing, make_preprocessing
from ._types import Transform
from ._utils import has_batch_dim


class PredictionPipeline(ModelAdapter):
    """
    Represents model *without* any preprocessing and postprocessing
    """

    @abc.abstractmethod
    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
        ...

    @property
    @abc.abstractmethod
    def name(self):
        ...

    @property
    @abc.abstractmethod
    def input_axes(self):
        ...

    @property
    @abc.abstractmethod
    def input_shape(self):
        ...

    @property
    @abc.abstractmethod
    def output_axes(self):
        ...

    @property
    @abc.abstractmethod
    def halo(self):
        ...


class _PredictionPipelineImpl(PredictionPipeline):
    def __init__(
        self,
        *,
        name: str,
        input_axes: str,
        input_shape: List[Tuple[str, int]],
        output_axes: str,
        halo: List[Tuple[str, int]],
        preprocessing: Transform,
        model: ModelAdapter,
        postprocessing: Transform,
    ) -> None:
        self._name = name
        self._halo = halo
        self._input_axes = input_axes
        self._output_axes = output_axes
        self._input_shape = input_shape
        self._preprocessing = preprocessing
        self._model: ModelAdapter = model
        self._postprocessing = postprocessing

    @property
    def name(self):
        return self._name

    @property
    def halo(self):
        return self._halo

    @property
    def input_axes(self):
        return self._input_axes

    @property
    def output_axes(self):
        return self._output_axes

    @property
    def input_shape(self):
        return self._input_shape

    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
        preprocessed = self._preprocessing(input_tensor)
        prediction = self._model.forward(preprocessed)
        return self._postprocessing(prediction)

    @property
    def max_num_iterations(self) -> int:
        return self._model.max_num_iterations

    @property
    def iteration_count(self) -> int:
        return self._model.iteration_count

    def set_break_callback(self, thunk: Callable[[], bool]) -> None:
        self._model.set_break_callback(thunk)

    def set_max_num_iterations(self, val: int) -> None:
        self._model.set_max_num_iterations(val)


def create_prediction_pipeline(
    *, pybio_model: nodes.Model, devices=List[str], preserve_batch_dim=False
) -> PredictionPipeline:
    """
    Creates prediction pipeline which includes:
    * preprocessing
    * model prediction
    * postprocessing
    """
    if len(pybio_model.inputs) != 1 or len(pybio_model.outputs) != 1:
        raise NotImplementedError(f"Only models with single input and output are supported")

    model_adapter: ModelAdapter = create_model_adapter(pybio_model=pybio_model, devices=devices)

    input = pybio_model.inputs[0]
    input_shape = input.shape
    input_axes = input.axes
    preprocessing_spec = input.preprocessing.copy()
    if has_batch_dim(input_axes) and not preserve_batch_dim:
        preprocessing_spec.insert(0, ADD_BATCH_DIM)
        input_axes = input_axes[1:]
        input_shape = input_shape[1:]

    preprocessing_spec.insert(0, make_ensure_dtype_preprocessing(input.data_type))
    input_named_shape = list(zip(input_axes, input_shape))
    preprocessing: Transform = make_preprocessing(preprocessing_spec)

    output = pybio_model.outputs[0]
    halo_shape = output.halo or [0 for _ in output.axes]
    output_axes = pybio_model.outputs[0].axes
    postprocessing_spec = output.postprocessing.copy()
    if has_batch_dim(output_axes) and not preserve_batch_dim:
        postprocessing_spec.append(REMOVE_BATCH_DIM)
        output_axes = output_axes[1:]
        halo_shape = halo_shape[1:]

    halo_named_shape = list(zip(output_axes, halo_shape))
    postprocessing: Transform = make_postprocessing(postprocessing_spec)

    return _PredictionPipelineImpl(
        name=pybio_model.name,
        input_axes=input_axes,
        input_shape=input_named_shape,
        output_axes=output_axes,
        halo=halo_named_shape,
        preprocessing=preprocessing,
        model=model_adapter,
        postprocessing=postprocessing,
    )
