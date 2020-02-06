import numpy as np

import inference_pb2


def numpy_to_pb_tensor(array: np.ndarray) -> inference_pb2.Tensor:
    shape = [inference_pb2.TensorDim(size=dim) for dim in array.shape]
    return inference_pb2.Tensor(dtype=str(array.dtype), shape=shape, buffer=bytes(array))


def pb_tensor_to_numpy(tensor: inference_pb2.Tensor) -> np.ndarray:
    if not tensor.dtype:
        raise ValueError("Tensor dtype is not specified")

    if not tensor.shape:
        raise ValueError("Tensor shape is not specified")

    return np.frombuffer(tensor.buffer, dtype=tensor.dtype).reshape(*[dim.size for dim in tensor.shape])
