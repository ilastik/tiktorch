import numpy as np


from tiktorch import serializers as ser
from tiktorch import types


def test_ndarray_serializer():
    s = ser.NDArraySerializer()
    ndarray = types.NDArray(np.random.rand(17, 12, 13), id_=(3, 5))
    serialized = s.serialize(ndarray)
    deserialized = s.deserialize(iter(serialized))
    np.testing.assert_array_almost_equal(deserialized.as_numpy(), ndarray.as_numpy())
    assert ndarray.id == deserialized.id
