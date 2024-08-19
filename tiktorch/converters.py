from __future__ import annotations

import dataclasses
from typing import Dict, Iterator, List, Tuple

import numpy as np
import xarray as xr
from bioimageio.core.resource_io import nodes
from bioimageio.core.resource_io.nodes import ParametrizedInputShape

from tiktorch.proto import inference_pb2

# pairs of axis-shape for a single tensor
NamedInt = Tuple[str, int]
NamedFloat = Tuple[str, float]
NamedShape = List[NamedInt]
NamedVec = List[NamedFloat]


class InputTensorValidator:
    def __init__(self, input_specs: List[nodes.InputTensor]):
        self._input_specs = input_specs

    def check_tensors(self, sample: Sample):
        for tensor_id, tensor in sample.tensors.items():
            self.check_shape(tensor_id, tensor.dims, tensor.shape)

    def _get_input_tensors_with_names(self) -> Dict[str, nodes.InputTensor]:
        return {tensor.name: tensor for tensor in self._input_specs}

    def check_shape(self, tensor_id: str, axes: Tuple[str, ...], shape: Tuple[int, ...]):
        shape = self.get_axes_with_size(axes, shape)
        spec = self._get_input_spec(tensor_id)
        if isinstance(spec.shape, list):
            self._check_shape_explicit(spec, shape)
        elif isinstance(spec.shape, ParametrizedInputShape):
            self._check_shape_parameterized(spec, shape)
        else:
            raise ValueError(f"Unexpected shape {spec.shape}")

    def _get_input_spec(self, tensor_id: str) -> nodes.InputTensor:
        self._check_spec_exists(tensor_id)
        specs = [spec for spec in self._input_specs if spec.name == tensor_id]
        assert len(specs) == 1, "ids of tensor specs should be unique"
        return specs[0]

    def _check_spec_exists(self, tensor_id: str):
        spec_names = [spec.name for spec in self._input_specs]
        if tensor_id not in spec_names:
            raise ValueError(f"Spec {tensor_id} doesn't exist for specs {spec_names}")

    def _check_shape_explicit(self, spec: nodes.InputTensor, tensor_shape: Dict[str, int]):
        assert self.is_shape_explicit(spec)
        reference_shape = {name: size for name, size in zip(spec.axes, spec.shape)}
        self.check_same_axes(reference_shape, tensor_shape)
        if reference_shape != tensor_shape:
            raise ValueError(f"Incompatible shapes found {tensor_shape}, expected {reference_shape}")

    def _check_shape_parameterized(self, spec: nodes.InputTensor, tensor_shape: Dict[str, int]):
        assert isinstance(spec.shape, ParametrizedInputShape)
        if not self.is_shape(tensor_shape.values()):
            raise ValueError(f"Invalid shape's sizes {tensor_shape}")

        min_shape = self.get_axes_with_size(spec.axes, tuple(spec.shape.min))
        step = self.get_axes_with_size(spec.axes, tuple(spec.shape.step))
        self.check_same_axes(tensor_shape, min_shape)

        tensor_shapes_arr = np.array(list(tensor_shape.values()))
        min_shape_arr = np.array(list(min_shape.values()))
        step_arr = np.array(list(step.values()))
        diff = tensor_shapes_arr - min_shape_arr
        if any(size < 0 for size in diff):
            raise ValueError(f"Tensor shape {tensor_shape} smaller than min shape {min_shape}")

        non_zero_idx = np.nonzero(step_arr)
        multipliers = diff[non_zero_idx] / step_arr[non_zero_idx]
        multiplier = np.unique(multipliers)
        if len(multiplier) == 1 and self.is_natural_number(multiplier[0]):
            return
        raise ValueError(f"Tensor shape {tensor_shape} not valid for spec {spec}")

    @staticmethod
    def check_same_axes(source: Dict[str, int], target: Dict[str, int]):
        if source.keys() != target.keys():
            raise ValueError(f"Incompatible axes for tensor {target} and reference {source}")

    @staticmethod
    def is_natural_number(n) -> bool:
        return n % 1 == 0.0 and n >= 0

    @staticmethod
    def is_shape(shape: Iterator[int]) -> bool:
        return all(InputTensorValidator.is_natural_number(dim) for dim in shape)

    @staticmethod
    def get_axes_with_size(axes: Tuple[str, ...], shape: Tuple[int, ...]) -> Dict[str, int]:
        assert len(axes) == len(shape)
        return {name: size for name, size in zip(axes, shape)}

    @staticmethod
    def is_shape_explicit(spec: nodes.InputTensor) -> bool:
        return isinstance(spec.shape, list)


@dataclasses.dataclass(frozen=True)
class Sample:
    tensors: Dict[str, xr.DataArray]

    @classmethod
    def from_pb_tensors(cls, pb_tensors: List[inference_pb2.Tensor]) -> Sample:
        return Sample({tensor.tensorId: pb_tensor_to_xarray(tensor) for tensor in pb_tensors})

    @classmethod
    def from_xr_tensors(cls, tensor_ids: List[str], tensors_data: List[xr.DataArray]) -> Sample:
        assert len(tensor_ids) == len(tensors_data)
        return Sample({tensor_id: tensor_data for tensor_id, tensor_data in zip(tensor_ids, tensors_data)})

    def to_pb_tensors(self) -> List[inference_pb2.Tensor]:
        return [xarray_to_pb_tensor(tensor_id, res_tensor) for tensor_id, res_tensor in self.tensors.items()]


def numpy_to_pb_tensor(array: np.ndarray, axistags=None) -> inference_pb2.Tensor:
    if axistags:
        shape = [inference_pb2.NamedInt(size=dim, name=name) for dim, name in zip(array.shape, axistags)]
    else:
        shape = [inference_pb2.NamedInt(size=dim) for dim in array.shape]
    return inference_pb2.Tensor(dtype=str(array.dtype), shape=shape, buffer=bytes(array))


def xarray_to_pb_tensor(tensor_id: str, array: xr.DataArray) -> inference_pb2.Tensor:
    shape = [inference_pb2.NamedInt(size=dim, name=name) for dim, name in zip(array.shape, array.dims)]
    return inference_pb2.Tensor(tensorId=tensor_id, dtype=str(array.dtype), shape=shape, buffer=bytes(array.data))


def name_int_tuples_to_pb_NamedInts(name_int_tuples) -> inference_pb2.NamedInts:
    return inference_pb2.NamedInts(
        namedInts=[inference_pb2.NamedInt(size=dim, name=name) for name, dim in name_int_tuples]
    )


def name_float_tuples_to_pb_NamedFloats(name_float_tuples) -> inference_pb2.NamedFloats:
    return inference_pb2.NamedFloats(
        namedFloats=[inference_pb2.NamedFloat(size=dim, name=name) for name, dim in name_float_tuples]
    )


def pb_tensor_to_xarray(tensor: inference_pb2.Tensor) -> inference_pb2.Tensor:
    if not tensor.dtype:
        raise ValueError("Tensor dtype is not specified")

    if not tensor.shape:
        raise ValueError("Tensor shape is not specified")

    data = np.frombuffer(tensor.buffer, dtype=tensor.dtype).reshape(*[dim.size for dim in tensor.shape])

    return xr.DataArray(data, dims=[d.name for d in tensor.shape])


def pb_tensor_to_numpy(tensor: inference_pb2.Tensor) -> np.ndarray:
    if not tensor.dtype:
        raise ValueError("Tensor dtype is not specified")

    if not tensor.shape:
        raise ValueError("Tensor shape is not specified")

    return np.frombuffer(tensor.buffer, dtype=tensor.dtype).reshape(*[dim.size for dim in tensor.shape])
