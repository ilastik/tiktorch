from __future__ import annotations

import dataclasses
import multiprocessing as _mp
import os
import pathlib
import tempfile
import uuid
from concurrent.futures import Future
from multiprocessing.connection import Connection
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

import bioimageio.spec as spec
import numpy
from bioimageio.core import load_resource_description
from bioimageio.core.prediction_pipeline import PredictionPipeline, create_prediction_pipeline
from marshmallow import missing

from tiktorch import log
from tiktorch.rpc import Shutdown
from tiktorch.rpc import mp as _mp_rpc
from tiktorch.rpc.mp import MPServer

from .backend import base
from .rpc_interface import IRPCModelSession

Value = TypeVar("Value")


class AxisWithValue(Generic[Value]):
    VALID_AXES = "itbczyx"
    SPATIAL_AXES = "xyz"

    def __init__(self, axes: str, values: Tuple[Value, ...]):
        self._axes = axes
        self._values = values
        assert len(self._axes) == len(self._values)
        assert all(self._check_axis(axis) for axis in axes)
        assert all(value >= 0 for value in values)
        self._mapping = {axis: size for axis, size in zip(self._axes, self._values)}
        self._xyz = {axis: size for axis, size in self._mapping.items() if axis in self.SPATIAL_AXES}
        self._spatial_axes = "".join(self._xyz.keys())
        self._spatial_values = tuple(list(self._xyz.values()))

    @classmethod
    def from_axes_size_map(cls, dict: Dict[str, Value]) -> AxisWithValue:
        return AxisWithValue("".join(dict.keys()), tuple(list((dict.values()))))

    @property
    def spatial_axes(self) -> str:
        return self._spatial_axes

    @property
    def spatial_values(self) -> Tuple[Value, ...]:
        return self._spatial_values

    @property
    def axes(self) -> str:
        return self._axes

    @property
    def values(self) -> Tuple[Value, ...]:
        return self._values

    @property
    def xyz(self) -> Dict[str, Value]:
        return self._xyz

    @property
    def is_3d(self) -> bool:
        return len(self.xyz) == 3

    def _check_axis(self, axis: str):
        return axis in self.VALID_AXES

    def is_same_axes(self, other: AxisWithValue) -> bool:
        return self._mapping.keys() == other._mapping.keys()

    def __getitem__(self, item):
        if item in self._mapping:
            return self._mapping[item]
        else:
            return 1

    def __iter__(self):
        return iter(self._mapping.items())

    def __len__(self):
        return len(self.axes)

    def __str__(self):
        return f"{self._mapping}"


@dataclasses.dataclass(frozen=True)
class ShapeWithHalo:
    shape: AxisWithValue[int]
    halo: AxisWithValue[int]

    def __post_init__(self):
        assert self.shape.is_same_axes(self.halo)

    @classmethod
    def from_values(cls, shape: Tuple[int, ...], halo: Tuple[int, ...], axes: str):
        return ShapeWithHalo(shape=AxisWithValue(axes, shape), halo=AxisWithValue(axes, halo))

    @property
    def axes(self) -> str:
        return self.shape.axes


@dataclasses.dataclass(frozen=True)
class ShapeWithReference:
    reference_tensor: str
    offset: AxisWithValue[int]
    scale: AxisWithValue[float]
    halo: AxisWithValue[int]

    def __post_init__(self):
        assert self.offset.is_same_axes(self.scale) and self.scale.is_same_axes(self.halo)

    @classmethod
    def from_values(
        cls, reference_tensor: str, offset: Tuple[int, ...], scale: Tuple[float, ...], halo: Tuple[int, ...], axes: str
    ):
        return ShapeWithReference(
            reference_tensor=reference_tensor,
            offset=AxisWithValue(axes, offset),
            scale=AxisWithValue(axes, scale),
            halo=AxisWithValue(axes, halo),
        )

    @property
    def axes(self) -> str:
        return self.offset.axes


class ParameterizedShape:
    def __init__(self, min_shape: AxisWithValue, steps: AxisWithValue):
        self._min_shape = min_shape
        self._steps = steps
        assert self._min_shape.is_same_axes(self._steps)
        assert all(step == 0 for axis, step in steps if axis not in AxisWithValue.SPATIAL_AXES)  # todo: ?
        self._default_multiplier = self._enforce_min_shape()
        self._custom_multiplier: Optional[int] = None
        self._total_shape = self.get_total_shape()

    @classmethod
    def from_values(cls, min_shape: Tuple[int, ...], steps: Tuple[int, ...], axes: str) -> ParameterizedShape:
        return ParameterizedShape(AxisWithValue(axes, min_shape), AxisWithValue(axes, steps))

    @property
    def axes(self) -> str:
        return self._min_shape.axes

    @property
    def spatial_axes(self) -> str:
        return self._min_shape.spatial_axes

    @property
    def min_shape(self) -> AxisWithValue:
        return self._min_shape

    @property
    def steps(self) -> AxisWithValue:
        return self._steps

    @property
    def default_multiplier(self) -> int:
        return self._default_multiplier

    @property
    def multiplier(self) -> int:
        if self._custom_multiplier is not None:
            return self._custom_multiplier
        else:
            return self._default_multiplier

    @multiplier.setter
    def multiplier(self, value):
        self._check_multiplier(value)
        self._custom_multiplier = value

    def get_total_shape(self, multiplier: Optional[int] = None) -> AxisWithValue:
        if multiplier is not None:
            self._check_multiplier(multiplier)
            self._custom_multiplier = multiplier
        else:
            multiplier = self._default_multiplier if self._custom_multiplier is None else self._custom_multiplier
        total_size = [size + multiplier * self._steps[axis] for axis, size in self._min_shape]
        self._total_shape = AxisWithValue(self.axes, tuple(total_size))
        return self._total_shape

    def _enforce_min_shape(self) -> int:
        """Hack: pick a bigger shape than min shape

        Some models come with super tiny minimal shapes, that make the processing
        too slow. While dryrun is not implemented, we'll "guess" a sensible shape
        and hope it will fit into memory.
        """
        MIN_SIZE_2D = 512
        MIN_SIZE_3D = 64

        spacial_increments = sum(i != 0 for i, a in self._steps.xyz.items())
        if spacial_increments > 2:
            target_size = MIN_SIZE_3D
        else:
            target_size = MIN_SIZE_2D

        factors = [
            int(numpy.ceil((target_size - size) / self._steps[axis]))
            for axis, size in self._min_shape.xyz.items()
            if self._steps[axis] != 0
        ]
        # we assume shape is "large" enough if one of the axes is larger than min_size
        if any(f <= 0 for f in factors):
            return 0

        # choose the smallest increment to make at least one size >= target_size
        m = min([x for x in factors])
        return m

    def _check_multiplier(self, value: int):
        if value < 0:
            raise ValueError(f"Multiplier value {value}. It should be >= 0")

    def __str__(self):
        return (
            f"{self.min_shape.spatial_values} + {self.multiplier} * {self.steps.spatial_values} "
            f"= {self.get_total_shape().spatial_values}"
        )


InputShapes = Union[AxisWithValue, ParameterizedShape]
OutputShapes = Union[ShapeWithHalo, ShapeWithReference]


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """Intermediate representation of bioimageio neural network model

    TODO (k-dominik): ModelInfo only used in inference_servicer to convert to
    protobuf modelinfo.

    """

    name: str
    input_shapes: Dict[str, InputShapes]
    output_shapes: Dict[str, OutputShapes]

    @classmethod
    def from_prediction_pipeline(cls, prediction_pipeline: PredictionPipeline) -> ModelInfo:
        input_shapes: Dict[str, InputShapes] = {}
        for input_spec in prediction_pipeline.input_specs:
            name = input_spec.name
            axes = "".join(input_spec.axes)
            shape = input_spec.shape
            if isinstance(shape, spec.shared.raw_nodes.ParametrizedInputShape):
                input_shapes[name] = ParameterizedShape.from_values(
                    min_shape=tuple(shape.min), steps=tuple(shape.step), axes=axes
                )
            elif isinstance(shape, list):
                input_shapes[name] = AxisWithValue(axes=axes, values=tuple(shape))
            else:
                raise ValueError(f"Not expected shape {shape}")

        output_shapes = {}
        for output_spec in prediction_pipeline.output_specs:
            name = output_spec.name
            axes = "".join(output_spec.axes)
            shape = output_spec.shape
            # halo is not required by spec. We could alternatively make it optional in the
            # respective grpc message types and handle missing values in ilastik
            halo = tuple([0 for _ in axes]) if output_spec.halo == missing else output_spec.halo
            if isinstance(shape, spec.shared.raw_nodes.ImplicitOutputShape):
                output_shapes[name] = ShapeWithReference.from_values(
                    reference_tensor=output_spec.shape.reference_tensor,
                    scale=tuple(shape.scale),
                    offset=shape.offset,
                    halo=halo,
                    axes=axes,
                )

            elif isinstance(output_spec.shape, spec.shared.raw_nodes.ExplicitShape):
                output_shapes[name] = ShapeWithHalo.from_values(shape=shape, halo=tuple(halo), axes=axes)

            else:
                raise ValueError(f"Not expected shape {shape}")

        return cls(
            name=prediction_pipeline.name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
        )


class ModelSessionProcess(IRPCModelSession):
    def __init__(self, model_zip: bytes, devices: List[str]) -> None:
        _tmp_file = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        _tmp_file.write(model_zip)
        _tmp_file.close()
        model = load_resource_description(pathlib.Path(_tmp_file.name))
        os.unlink(_tmp_file.name)
        self._model: PredictionPipeline = create_prediction_pipeline(bioimageio_model=model, devices=devices)
        self._datasets = {}
        self._worker = base.SessionBackend(self._model)

    def forward(self, input_tensors: numpy.ndarray) -> Future:
        res = self._worker.forward(input_tensors)
        return res

    def create_dataset(self, mean, stddev):
        id_ = uuid.uuid4().hex
        self._datasets[id_] = {"mean": mean, "stddev": stddev}
        return id_

    def get_model_info(self) -> ModelInfo:
        return ModelInfo.from_prediction_pipeline(self._model)

    def shutdown(self) -> Shutdown:
        self._worker.shutdown()
        return Shutdown()


def _run_model_session_process(
    conn: Connection, model_zip: bytes, devices: List[str], log_queue: Optional[_mp.Queue] = None
):
    try:
        # from: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    except ModuleNotFoundError:
        pass  # probably running on windows

    if log_queue:
        log.configure(log_queue)

    session_proc = ModelSessionProcess(model_zip, devices)
    srv = MPServer(session_proc, conn)
    srv.listen()


def start_model_session_process(
    model_zip: bytes, devices: List[str], log_queue: Optional[_mp.Queue] = None
) -> Tuple[_mp.Process, IRPCModelSession]:
    client_conn, server_conn = _mp.Pipe()
    proc = _mp.Process(
        target=_run_model_session_process,
        name="ModelSessionProcess",
        kwargs={"conn": server_conn, "devices": devices, "log_queue": log_queue, "model_zip": model_zip},
    )
    proc.start()
    return proc, _mp_rpc.create_client(IRPCModelSession, client_conn)
