from typing import Iterator, Tuple

import zmq
import numpy as np

from zmq.utils import jsonapi

from .types import NDArrayBatch, NDArray
from .rpc.serialization import ISerializer, FusedFrameIterator, serializer_for


def _make_ndarray(dtype: str, shape: Tuple[int, ...], id_: Tuple[int, ...], frame: zmq.Frame) -> NDArray:
    arr = np.frombuffer(frame.buffer, dtype=dtype)
    arr.shape = shape
    return NDArray(arr, id_ and tuple(id_))


@serializer_for(NDArrayBatch, tag=b"ndbatch")
class NDArrayBatchSerializer(ISerializer[NDArrayBatch]):
    """
    Serialization/deserialization protocol for NDArrayBatch
    First frame contains metadata encoded and json
    Rest of the frames contain raw buffer data
    """

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> NDArrayBatch:
        meta_frame = next(frames)
        meta = jsonapi.loads(meta_frame.bytes)

        arrays = []
        for item, buf_frame in zip(meta, frames):
            nd_array = _make_ndarray(item["dtype"], item["shape"], item["id"], buf_frame)
            arrays.append(nd_array)

        return NDArrayBatch(arrays)

    @classmethod
    def serialize(cls, obj: NDArrayBatch) -> Iterator[zmq.Frame]:
        yield zmq.Frame(jsonapi.dumps(obj.array_metas()))

        for arr in obj.as_numpy():
            yield zmq.Frame(arr)


@serializer_for(NDArray, tag=b"ndarray")
class NDArraySerializer(ISerializer[NDArray]):
    """
    Serialization/deserialization protocol for NDArray
    First frame contains metadata encoded and json
    Next frame contains raw buffer data
    """

    @classmethod
    def deserialize(cls, frames: FusedFrameIterator) -> NDArrayBatch:
        meta_frame = next(frames)
        meta = jsonapi.loads(meta_frame.bytes)

        buf_frame = next(frames)
        return _make_ndarray(meta["dtype"], meta["shape"], meta["id"], buf_frame)

    @classmethod
    def serialize(cls, obj: NDArray) -> Iterator[zmq.Frame]:
        meta = {"id": obj.id, "shape": obj.shape, "dtype": str(obj.dtype)}
        yield zmq.Frame(jsonapi.dumps(meta))
        yield zmq.Frame(obj.as_numpy())
