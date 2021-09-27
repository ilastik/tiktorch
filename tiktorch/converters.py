from typing import Union

import numpy as np
import xarray as xr

from tiktorch.proto import inference_pb2
from tiktorch.server.session.process import (
    NamedExplicitOutputShape,
    NamedImplicitOutputShape,
    NamedParametrizedShape,
    NamedShape,
)


def numpy_to_pb_tensor(array: np.ndarray, axistags=None) -> inference_pb2.Tensor:
    if axistags:
        shape = [inference_pb2.NamedInt(size=dim, name=name) for dim, name in zip(array.shape, axistags)]
    else:
        shape = [inference_pb2.NamedInt(size=dim) for dim in array.shape]
    return inference_pb2.Tensor(dtype=str(array.dtype), shape=shape, buffer=bytes(array))


def xarray_to_pb_tensor(array: xr.DataArray) -> inference_pb2.Tensor:
    shape = [inference_pb2.NamedInt(size=dim, name=name) for dim, name in zip(array.shape, array.dims)]
    return inference_pb2.Tensor(dtype=str(array.dtype), shape=shape, buffer=bytes(array.data))


def name_int_tuples_to_pb_shape(name_int_tuples) -> inference_pb2.Shape:
    return inference_pb2.Shape(dims=[inference_pb2.NamedInt(size=dim, name=name) for name, dim in name_int_tuples])


def name_float_tuples_to_pb_scale(name_float_tuples) -> inference_pb2.Shape:
    return inference_pb2.Scale(
        scales=[inference_pb2.NamedFloat(size=dim, name=name) for name, dim in name_float_tuples]
    )


def input_shape_to_pb_input_shape(input_shape: Union[NamedShape, NamedParametrizedShape]) -> inference_pb2.InputShape:

    if isinstance(input_shape, NamedParametrizedShape):
        return inference_pb2.InputShape(
            shapeType=1,
            shape=name_int_tuples_to_pb_shape(input_shape.min_shape),
            stepShape=name_int_tuples_to_pb_shape(input_shape.step_shape),
        )
    else:
        return inference_pb2.InputShape(
            shapeType=0,
            shape=name_int_tuples_to_pb_shape(input_shape),
        )


def output_shape_to_pb_output_shape(
    output_shape: Union[NamedExplicitOutputShape, NamedImplicitOutputShape]
) -> inference_pb2.InputShape:

    if isinstance(output_shape, NamedImplicitOutputShape):
        return inference_pb2.OutputShape(
            shapeType=1,
            halo=name_int_tuples_to_pb_shape(output_shape.halo),
            referenceTensor=output_shape.reference_tensor,
            scale=name_float_tuples_to_pb_scale(output_shape.scale),
            offset=name_int_tuples_to_pb_shape(output_shape.offset),
        )
    elif isinstance(output_shape, NamedExplicitOutputShape):
        return inference_pb2.OutputShape(
            shapeType=0,
            shape=name_int_tuples_to_pb_shape(output_shape.shape),
            halo=name_int_tuples_to_pb_shape(output_shape.halo),
        )
    else:
        raise TypeError(f"Conversion not supported for type {type(output_shape)}")


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
