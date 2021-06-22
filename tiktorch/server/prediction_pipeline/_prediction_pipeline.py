import abc
from typing import Callable, List, Optional, Tuple

import xarray as xr
from bioimageio.spec import nodes
from marshmallow import missing

from ._model_adapters import ModelAdapter, create_model_adapter
from ._postprocessing import REMOVE_BATCH_DIM, make_postprocessing
from ._preprocessing import ADD_BATCH_DIM, make_ensure_dtype_preprocessing, make_preprocessing
from ._types import Transform
from ._utils import has_batch_dim


class PredictionPipeline(ModelAdapter):
    """
    Represents model computation including preprocessing and postprocessing
    """

    @abc.abstractmethod
    def forward(self, input_tensor: xr.DataArray) -> xr.DataArray:
        """
        Compute predictions
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the pipeline
        """
        ...

    @property
    @abc.abstractmethod
    def input_axes(self) -> str:
        """
        Input axes excepted by this pipeline
        Note: one character axes names
        """
        ...

    @property
    @abc.abstractmethod
    def input_shape(self) -> List[Tuple[str, int]]:
        """
        Named input dimensions
        """
        ...

    @property
    @abc.abstractmethod
    def output_axes(self) -> str:
        """
        Output axes of this pipeline
        Note: one character axes names
        """
        ...

    @property
    @abc.abstractmethod
    def halo(self) -> List[Tuple[str, int]]:
        """
        Size of output borders that have unreliable data due to artifacts
        """
        ...

    @property
    @abc.abstractmethod
    def scale(self) -> List[Tuple[str, float]]:
        """
        Scale of output tensor relative to input
        """
        ...

    @property
    @abc.abstractmethod
    def offset(self) -> List[Tuple[str, int]]:
        """
        Offset of output tensor relative to input
        """
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
        scale: List[Tuple[str, float]],
        offset: List[Tuple[str, int]],
        preprocessing: Transform,
        model: ModelAdapter,
        postprocessing: Transform,
    ) -> None:
        self._name = name
        self._halo = halo
        self._scale = scale
        self._offset = offset
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
    def scale(self):
        return self._scale

    @property
    def offset(self):
        return self._offset

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
    *, bioimageio_model: nodes.Model, devices=List[str], preserve_batch_dim=False, weight_format: Optional[str] = None
) -> PredictionPipeline:
    """
    Creates prediction pipeline which includes:
    * preprocessing
    * model prediction
    * postprocessing
    """
    if len(bioimageio_model.inputs) != 1 or len(bioimageio_model.outputs) != 1:
        raise NotImplementedError("Only models with single input and output are supported")

    model_adapter: ModelAdapter = create_model_adapter(
        bioimageio_model=bioimageio_model, devices=devices, weight_format=weight_format
    )

    input = bioimageio_model.inputs[0]
    input_shape = input.shape
    input_axes = input.axes
    preprocessing_spec = [] if input.preprocessing is missing else input.preprocessing.copy()
    if has_batch_dim(input_axes) and not preserve_batch_dim:
        preprocessing_spec.insert(0, ADD_BATCH_DIM)
        input_axes = input_axes[1:]
        input_shape = input_shape[1:]

    preprocessing_spec.insert(0, make_ensure_dtype_preprocessing(input.data_type))
    input_named_shape = list(zip(input_axes, input_shape))
    preprocessing: Transform = make_preprocessing(preprocessing_spec)

    output = bioimageio_model.outputs[0]
    halo_shape = output.halo or [0 for _ in output.axes]
    output_axes = bioimageio_model.outputs[0].axes
    postprocessing_spec = [] if output.postprocessing is missing else output.postprocessing.copy()
    if has_batch_dim(output_axes) and not preserve_batch_dim:
        postprocessing_spec.append(REMOVE_BATCH_DIM)
        output_axes = output_axes[1:]
        halo_shape = halo_shape[1:]

    halo_named_shape = list(zip(output_axes, halo_shape))

    if isinstance(output.shape, list):
        raise NotImplementedError("Expected implicit output shape")
    else:
        scale = output.shape.scale or [0 for _ in output.axes]
        offset = output.shape.offset or [0 for _ in output.axes]

    named_scale = list(zip(output_axes, scale))
    named_offset = list(zip(output_axes, offset))

    postprocessing: Transform = make_postprocessing(postprocessing_spec)

    return _PredictionPipelineImpl(
        name=bioimageio_model.name,
        input_axes=input_axes,
        input_shape=input_named_shape,
        output_axes=output_axes,
        halo=halo_named_shape,
        scale=named_scale,
        offset=named_offset,
        preprocessing=preprocessing,
        model=model_adapter,
        postprocessing=postprocessing,
    )
