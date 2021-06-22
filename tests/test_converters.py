import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from tiktorch.converters import numpy_to_pb_tensor, pb_tensor_to_numpy, pb_tensor_to_xarray, xarray_to_pb_tensor
from tiktorch.proto import inference_pb2


def _numpy_to_pb_tensor(arr):
    """
    Makes sure that tensor was serialized/deserialized
    """
    tensor = numpy_to_pb_tensor(arr)
    parsed = inference_pb2.Tensor()
    parsed.ParseFromString(tensor.SerializeToString())
    return parsed


class TestNumpyToPBTensor:
    def test_should_serialize_to_tensor_type(self):
        arr = np.arange(9)
        tensor = _numpy_to_pb_tensor(arr)
        assert isinstance(tensor, inference_pb2.Tensor)

    @pytest.mark.parametrize("np_dtype,dtype_str", [(np.int64, "int64"), (np.uint8, "uint8"), (np.float32, "float32")])
    def test_should_have_dtype_as_str(self, np_dtype, dtype_str):
        arr = np.arange(9, dtype=np_dtype)
        tensor = _numpy_to_pb_tensor(arr)
        assert arr.dtype == tensor.dtype

    @pytest.mark.parametrize("shape", [(3, 3), (1,), (1, 1), (18, 20, 1)])
    def test_should_have_shape(self, shape):
        arr = np.zeros(shape)
        tensor = _numpy_to_pb_tensor(arr)
        assert tensor.shape
        assert list(shape) == [dim.size for dim in tensor.shape]

    def test_should_have_serialized_bytes(self):
        arr = np.arange(9, dtype=np.uint8)
        expected = bytes(arr)
        tensor = _numpy_to_pb_tensor(arr)

        assert expected == tensor.buffer


class TestPBTensorToNumpy:
    def test_should_raise_on_empty_dtype(self):
        tensor = inference_pb2.Tensor(dtype="", shape=[inference_pb2.NamedInt(size=1), inference_pb2.NamedInt(size=2)])
        with pytest.raises(ValueError):
            pb_tensor_to_numpy(tensor)

    def test_should_raise_on_empty_shape(self):
        tensor = inference_pb2.Tensor(dtype="int64", shape=[])
        with pytest.raises(ValueError):
            pb_tensor_to_numpy(tensor)

    def test_should_return_ndarray(self):
        arr = np.arange(9)
        parsed = _numpy_to_pb_tensor(arr)
        result_arr = pb_tensor_to_numpy(parsed)

        assert isinstance(result_arr, np.ndarray)

    @pytest.mark.parametrize("np_dtype,dtype_str", [(np.int64, "int64"), (np.uint8, "uint8"), (np.float32, "float32")])
    def test_should_have_same_dtype(self, np_dtype, dtype_str):
        arr = np.arange(9, dtype=np_dtype)
        tensor = _numpy_to_pb_tensor(arr)
        result_arr = pb_tensor_to_numpy(tensor)

        assert arr.dtype == result_arr.dtype

    @pytest.mark.parametrize("shape", [(3, 3), (1,), (1, 1), (18, 20, 1)])
    def test_should_same_shape(self, shape):
        arr = np.zeros(shape)
        tensor = _numpy_to_pb_tensor(arr)
        result_arr = pb_tensor_to_numpy(tensor)
        assert arr.shape == result_arr.shape

    @pytest.mark.parametrize("shape", [(3, 3), (1,), (1, 1), (18, 20, 1)])
    def test_should_same_data(self, shape):
        arr = np.random.random(shape)
        tensor = _numpy_to_pb_tensor(arr)
        result_arr = pb_tensor_to_numpy(tensor)

        assert_array_equal(arr, result_arr)


class TestXarrayToPBTensor:
    def to_pb_tensor(self, arr):
        """
        Makes sure that tensor was serialized/deserialized
        """
        tensor = xarray_to_pb_tensor(arr)
        parsed = inference_pb2.Tensor()
        parsed.ParseFromString(tensor.SerializeToString())
        return parsed

    def test_should_serialize_to_tensor_type(self):
        xarr = xr.DataArray(np.arange(8).reshape((2, 4)), dims=("x", "y"))
        pb_tensor = self.to_pb_tensor(xarr)
        assert isinstance(pb_tensor, inference_pb2.Tensor)
        assert len(pb_tensor.shape) == 2
        dim1 = pb_tensor.shape[0]
        dim2 = pb_tensor.shape[1]

        assert dim1.size == 2
        assert dim1.name == "x"

        assert dim2.size == 4
        assert dim2.name == "y"

    @pytest.mark.parametrize("shape", [(3, 3), (1,), (1, 1), (18, 20, 1)])
    def test_should_have_shape(self, shape):
        arr = xr.DataArray(np.zeros(shape))
        tensor = self.to_pb_tensor(arr)
        assert tensor.shape
        assert list(shape) == [dim.size for dim in tensor.shape]

    def test_should_have_serialized_bytes(self):
        arr = xr.DataArray(np.arange(9, dtype=np.uint8))
        expected = bytes(arr.data)
        tensor = self.to_pb_tensor(arr)

        assert expected == tensor.buffer


class TestPBTensorToXarray:
    def to_pb_tensor(self, arr):
        """
        Makes sure that tensor was serialized/deserialized
        """
        tensor = xarray_to_pb_tensor(arr)
        parsed = inference_pb2.Tensor()
        parsed.ParseFromString(tensor.SerializeToString())
        return parsed

    def test_should_raise_on_empty_dtype(self):
        tensor = inference_pb2.Tensor(dtype="", shape=[inference_pb2.NamedInt(size=1), inference_pb2.NamedInt(size=2)])
        with pytest.raises(ValueError):
            pb_tensor_to_xarray(tensor)

    def test_should_raise_on_empty_shape(self):
        tensor = inference_pb2.Tensor(dtype="int64", shape=[])
        with pytest.raises(ValueError):
            pb_tensor_to_xarray(tensor)

    def test_should_return_ndarray(self):
        arr = xr.DataArray(np.arange(9))
        parsed = self.to_pb_tensor(arr)
        result_arr = pb_tensor_to_xarray(parsed)

        assert isinstance(result_arr, xr.DataArray)

    @pytest.mark.parametrize("np_dtype,dtype_str", [(np.int64, "int64"), (np.uint8, "uint8"), (np.float32, "float32")])
    def test_should_have_same_dtype(self, np_dtype, dtype_str):
        arr = xr.DataArray(np.arange(9, dtype=np_dtype))
        tensor = self.to_pb_tensor(arr)
        result_arr = pb_tensor_to_xarray(tensor)

        assert arr.dtype == result_arr.dtype

    @pytest.mark.parametrize("shape", [(3, 3), (1,), (1, 1), (18, 20, 1)])
    def test_should_same_shape(self, shape):
        arr = xr.DataArray(np.zeros(shape))
        tensor = self.to_pb_tensor(arr)
        result_arr = pb_tensor_to_xarray(tensor)
        assert arr.shape == result_arr.shape

    @pytest.mark.parametrize("shape", [(3, 3), (1,), (1, 1), (18, 20, 1)])
    def test_should_same_data(self, shape):
        arr = xr.DataArray(np.random.random(shape))
        tensor = self.to_pb_tensor(arr)
        result_arr = pb_tensor_to_xarray(tensor)
        assert_array_equal(arr, result_arr)
