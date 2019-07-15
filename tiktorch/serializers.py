from typing import Iterator, Tuple

import numpy as np
import zmq
from zmq.utils import jsonapi

from .rpc.serialization import FusedFrameIterator, ISerializer, serializer_for
from .types import Model, ModelState, NDArray, NDArrayBatch, SetDeviceReturnType


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


@serializer_for(SetDeviceReturnType, tag=b"setdevices")
class SetDeviceReturnTypeSerializer(ISerializer[SetDeviceReturnType]):
    @classmethod
    def serialize(cls, obj: SetDeviceReturnType) -> Iterator[zmq.Frame]:
        yield zmq.Frame(
            jsonapi.dumps(
                {"shrinkage": obj.shrinkage, "valid_shapes": obj.valid_shapes, "training_shape": obj.training_shape}
            )
        )

    @classmethod
    def deserialize(cls, frames: "FusedFrameIterator") -> SetDeviceReturnType:
        frm = next(frames)
        data = jsonapi.loads(frm.bytes)
        return SetDeviceReturnType(
            training_shape=tuple(data["training_shape"]),
            valid_shapes=[tuple(el) for el in data["valid_shapes"]],
            shrinkage=tuple(data["shrinkage"]),
        )


@serializer_for(ModelState, tag=b"model_st")
class ModelStateSerializer(ISerializer[ModelState]):
    @classmethod
    def serialize(cls, obj: ModelState) -> Iterator[zmq.Frame]:
        yield zmq.Frame(
            jsonapi.dumps(
                {
                    "epoch": obj.epoch,
                    "loss": obj.loss,
                    "num_iterations_done": obj.num_iterations_done,
                    "num_iterations_max": obj.num_iterations_max,
                }
            )
        )
        yield zmq.Frame(obj.model_state)
        yield zmq.Frame(obj.optimizer_state)

    @classmethod
    def deserialize(cls, frames: "FusedFrameIterator") -> ModelState:
        frm = next(frames)
        epoch_loss = jsonapi.loads(frm.bytes)

        model_state = next(frames)
        optimizer_state = next(frames)

        return ModelState(**epoch_loss, model_state=model_state.bytes, optimizer_state=optimizer_state.bytes)


@serializer_for(Model, tag=b"model")
class ModelSerializer(ISerializer[Model]):
    @classmethod
    def serialize(cls, obj: Model) -> Iterator[zmq.Frame]:
        yield zmq.Frame(obj.code)
        yield zmq.Frame(jsonapi.dumps(obj.config))

    @classmethod
    def deserialize(cls, frames: "FusedFrameIterator") -> ModelState:
        frm_code = next(frames)
        frm_config = next(frames)
        config = jsonapi.loads(frm_config.bytes)
        return Model(code=frm_code.bytes, config=config)
