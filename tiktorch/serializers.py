from typing import Iterator, Tuple

import zmq
import numpy as np

from zmq.utils import jsonapi

from .types import NDArrayBatch, NDArray
from .rpc.serialization import ISerializer, FusedFrameIterator, serializer_for


@serializer_for(NDArrayBatch)
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
            nd_array = cls._make_ndarray(
                item['dtype'],
                item['shape'],
                item['id'],
                buf_frame
            )
            arrays.append(nd_array)

        return NDArrayBatch(arrays)

    @classmethod
    def serialize(cls, obj: NDArrayBatch) -> Iterator[zmq.Frame]:
        yield zmq.Frame(jsonapi.dumps(obj.array_metas()))

        for arr in obj.as_numpy():
            yield zmq.Frame(arr)

    @staticmethod
    def _make_ndarray(
        dtype: str,
        shape: Tuple[int, ...],
        id_: Tuple[int, ...],
        frame: zmq.Frame
    ) -> NDArray:

        arr = np.frombuffer(frame.buffer, dtype=dtype)
        arr.shape = shape
        return NDArray(arr, id_)
