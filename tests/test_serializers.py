import numpy as np
import zmq


from tiktorch import serializers as ser
from tiktorch.rpc import serialize, deserialize
from tiktorch import types


def test_ndarray_serializer():
    s = ser.NDArraySerializer()
    ndarray = types.NDArray(np.random.rand(17, 12, 13), id_=(3, 5))
    serialized = s.serialize(ndarray)
    deserialized = s.deserialize(iter(serialized))
    np.testing.assert_array_almost_equal(deserialized.as_numpy(), ndarray.as_numpy())
    assert ndarray.id == deserialized.id


def test_set_devices_return_serialization_deserialization():
    ret = types.SetDeviceReturnType(
        training_shape=(1, 2, 3, 4, 5), valid_shapes=[(1, 2, 3, 4, 5), (1001, 2, 3, 4, 100)], shrinkage=(0, 1, 0, 1, 0)
    )

    serialized = list(serialize(ret))
    assert all(isinstance(f, zmq.Frame) for f in serialized)
    assert deserialize(iter(serialized)) == ret


def test_model_state_serializer():
    s = ser.ModelStateSerializer()
    state = types.ModelState(2.3, 12, b"model_state", b"opt_state", 10, 12)
    serialized = s.serialize(state)
    deserialized = s.deserialize(iter(serialized))
    assert state == deserialized


def test_model_serializer():
    s = ser.ModelSerializer()
    model = types.Model(code=b"import os", config={"val1": 10})
    serialized = s.serialize(model)
    deserialized = s.deserialize(iter(serialized))
    assert model == deserialized
