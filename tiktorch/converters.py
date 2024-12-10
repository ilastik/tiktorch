from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr
from bioimageio.core import Sample, Tensor
from bioimageio.spec.model.v0_5 import TensorId

from tiktorch.proto import inference_pb2, training_pb2
from tiktorch.trainer import TrainerState

trainer_state_to_pb = {
    TrainerState.IDLE: training_pb2.GetStatusResponse.State.Idle,
    TrainerState.RUNNING: training_pb2.GetStatusResponse.State.Running,
    TrainerState.PAUSED: training_pb2.GetStatusResponse.State.Paused,
    TrainerState.FAILED: training_pb2.GetStatusResponse.State.Failed,
    TrainerState.FINISHED: training_pb2.GetStatusResponse.State.Finished,
}

pb_state_to_trainer = {value: key for key, value in trainer_state_to_pb.items()}


def pb_tensors_to_sample(pb_tensors: List[inference_pb2.Tensor]) -> Sample:
    return Sample(
        members={TensorId(tensor.tensorId): Tensor.from_xarray(pb_tensor_to_xarray(tensor)) for tensor in pb_tensors},
        id=None,
        stat={},
    )


def xr_tensors_to_sample(tensor_ids: List[str], tensors_data: List[xr.DataArray]) -> Sample:
    assert len(tensor_ids) == len(tensors_data)
    return Sample(
        members={
            TensorId(tensor_id): Tensor.from_xarray(tensor_data)
            for tensor_id, tensor_data in zip(tensor_ids, tensors_data)
        },
        id=None,
        stat={},
    )


def sample_to_pb_tensors(sample: Sample) -> List[inference_pb2.Tensor]:
    return [xarray_to_pb_tensor(tensor_id, res_tensor.data) for tensor_id, res_tensor in sample.members.items()]


def numpy_to_pb_tensor(tensor_id: str, array: np.ndarray, axistags=None) -> inference_pb2.Tensor:
    if axistags:
        shape = [inference_pb2.NamedInt(size=dim, name=name) for dim, name in zip(array.shape, axistags)]
    else:
        shape = [inference_pb2.NamedInt(size=dim) for dim in array.shape]
    return inference_pb2.Tensor(tensorId=tensor_id, dtype=str(array.dtype), shape=shape, buffer=bytes(array))


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
