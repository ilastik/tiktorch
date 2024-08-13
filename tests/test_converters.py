import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from tiktorch.converters import (
    NamedExplicitOutputShape,
    NamedImplicitOutputShape,
    NamedParametrizedShape,
    Sample,
    input_shape_to_pb_input_shape,
    numpy_to_pb_tensor,
    output_shape_to_pb_output_shape,
    pb_tensor_to_numpy,
    pb_tensor_to_xarray,
    xarray_to_pb_tensor,
)
from tiktorch.proto import inference_pb2


def _numpy_to_pb_tensor(arr):
    """
    Makes sure that tensor was serialized/deserialized
    """
    tensor = numpy_to_pb_tensor(arr)
    parsed = inference_pb2.Tensor()
    parsed.ParseFromString(tensor.SerializeToString())
    return parsed


def to_pb_tensor(tensor_id: str, arr: xr.DataArray):
    """
    Makes sure that tensor was serialized/deserialized
    """
    tensor = xarray_to_pb_tensor(tensor_id, arr)
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
    def test_should_serialize_to_tensor_type(self):
        xarr = xr.DataArray(np.arange(8).reshape((2, 4)), dims=("x", "y"))
        pb_tensor = to_pb_tensor("input0", xarr)
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
        tensor = to_pb_tensor("input0", arr)
        assert tensor.shape
        assert list(shape) == [dim.size for dim in tensor.shape]

    def test_should_have_serialized_bytes(self):
        arr = xr.DataArray(np.arange(9, dtype=np.uint8))
        expected = bytes(arr.data)
        tensor = to_pb_tensor("input0", arr)

        assert expected == tensor.buffer


class TestPBTensorToXarray:
    def test_should_raise_on_empty_dtype(self):
        tensor = inference_pb2.Tensor(dtype="", shape=[inference_pb2.NamedInt(size=1), inference_pb2.NamedInt(size=2)])
        with pytest.raises(ValueError):
            pb_tensor_to_xarray(tensor)

    def test_should_raise_on_empty_shape(self):
        tensor = inference_pb2.Tensor(dtype="int64", shape=[])
        with pytest.raises(ValueError):
            pb_tensor_to_xarray(tensor)

    def test_should_return_xarray(self):
        arr = xr.DataArray(np.arange(9))
        parsed = to_pb_tensor("input0", arr)
        result_tensor = pb_tensor_to_xarray(parsed)
        assert isinstance(result_tensor, xr.DataArray)

    @pytest.mark.parametrize("np_dtype,dtype_str", [(np.int64, "int64"), (np.uint8, "uint8"), (np.float32, "float32")])
    def test_should_have_same_dtype(self, np_dtype, dtype_str):
        arr = xr.DataArray(np.arange(9, dtype=np_dtype))
        pb_tensor = to_pb_tensor("input0", arr)
        result_arr = pb_tensor_to_xarray(pb_tensor)

        assert arr.dtype == result_arr.dtype

    @pytest.mark.parametrize("shape", [(3, 3), (1,), (1, 1), (18, 20, 1)])
    def test_should_same_shape(self, shape):
        arr = xr.DataArray(np.zeros(shape))
        pb_tensor = to_pb_tensor("input0", arr)
        result_arr = pb_tensor_to_xarray(pb_tensor)
        assert arr.shape == result_arr.shape

    @pytest.mark.parametrize("shape", [(3, 3), (1,), (1, 1), (18, 20, 1)])
    def test_should_same_data(self, shape):
        arr = xr.DataArray(np.random.random(shape))
        pb_tensor = to_pb_tensor("input0", arr)
        result_arr = pb_tensor_to_xarray(pb_tensor)
        assert_array_equal(arr, result_arr)


class TestShapeConversions:
    def to_named_explicit_shape(self, shape, axes, halo):
        return NamedExplicitOutputShape(
            halo=[(name, dim) for name, dim in zip(axes, halo)], shape=[(name, dim) for name, dim in zip(axes, shape)]
        )

    def to_named_implicit_shape(self, axes, halo, offset, scales, reference_tensor):
        return NamedImplicitOutputShape(
            halo=[(name, dim) for name, dim in zip(axes, halo)],
            offset=[(name, dim) for name, dim in zip(axes, offset)],
            scale=[(name, scale) for name, scale in zip(axes, scales)],
            reference_tensor=reference_tensor,
        )

    def to_named_paramtrized_shape(self, min_shape, axes, step):
        return NamedParametrizedShape(
            min_shape=[(name, dim) for name, dim in zip(axes, min_shape)],
            step_shape=[(name, dim) for name, dim in zip(axes, step)],
        )

    @pytest.mark.parametrize(
        "shape,axes,halo",
        [((42,), "x", (0,)), ((42, 128, 5), "abc", (1, 1, 1)), ((5, 4, 3, 2, 1, 42), "btzyxc", (1, 2, 3, 4, 5, 24))],
    )
    def test_explicit_output_shape(self, shape, axes, halo):
        named_shape = self.to_named_explicit_shape(shape, axes, halo)
        pb_shape = output_shape_to_pb_output_shape(named_shape)

        assert pb_shape.shapeType == 0
        assert pb_shape.referenceTensor == ""
        assert len(pb_shape.scale.namedFloats) == 0
        assert len(pb_shape.offset.namedFloats) == 0

        assert [(d.name, d.size) for d in pb_shape.halo.namedInts] == [(name, size) for name, size in zip(axes, halo)]
        assert [(d.name, d.size) for d in pb_shape.shape.namedInts] == [(name, size) for name, size in zip(axes, shape)]

    @pytest.mark.parametrize(
        "axes,halo,offset,scales,reference_tensor",
        [("x", (0,), (10,), (1.0,), "forty-two"), ("abc", (1, 1, 1), (1, 2, 3), (1.0, 2.0, 3.0), "helloworld")],
    )
    def test_implicit_output_shape(self, axes, halo, offset, scales, reference_tensor):
        named_shape = self.to_named_implicit_shape(axes, halo, offset, scales, reference_tensor)
        pb_shape = output_shape_to_pb_output_shape(named_shape)

        assert pb_shape.shapeType == 1
        assert pb_shape.referenceTensor == reference_tensor
        assert [(d.name, d.size) for d in pb_shape.scale.namedFloats] == [
            (name, size) for name, size in zip(axes, scales)
        ]
        assert [(d.name, d.size) for d in pb_shape.offset.namedFloats] == [
            (name, size) for name, size in zip(axes, offset)
        ]

        assert [(d.name, d.size) for d in pb_shape.halo.namedInts] == [(name, size) for name, size in zip(axes, halo)]
        assert len(pb_shape.shape.namedInts) == 0

    def test_output_shape_raises(self):
        shape = [("a", 1)]
        with pytest.raises(TypeError):
            _ = output_shape_to_pb_output_shape(shape)

    @pytest.mark.parametrize(
        "shape,axes",
        [((42,), "x"), ((42, 128, 5), "abc"), ((5, 4, 3, 2, 1, 42), "btzyxc")],
    )
    def test_explicit_input_shape(self, shape, axes):
        named_shape = [(name, dim) for name, dim in zip(axes, shape)]
        pb_shape = input_shape_to_pb_input_shape(named_shape)

        assert pb_shape.shapeType == 0
        assert [(d.name, d.size) for d in pb_shape.shape.namedInts] == [(name, size) for name, size in zip(axes, shape)]

    @pytest.mark.parametrize(
        "min_shape,axes,step",
        [
            ((42,), "x", (5,)),
            ((42, 128, 5), "abc", (1, 2, 3)),
            ((5, 4, 3, 2, 1, 42), "btzyxc", (15, 24, 33, 42, 51, 642)),
        ],
    )
    def test_parametrized_input_shape(self, min_shape, axes, step):
        named_shape = self.to_named_paramtrized_shape(min_shape, axes, step)
        pb_shape = input_shape_to_pb_input_shape(named_shape)

        assert pb_shape.shapeType == 1
        assert [(d.name, d.size) for d in pb_shape.shape.namedInts] == [
            (name, size) for name, size in zip(axes, min_shape)
        ]
        assert [(d.name, d.size) for d in pb_shape.stepShape.namedInts] == [
            (name, size) for name, size in zip(axes, step)
        ]


class TestSample:
    def test_create_sample_from_pb_tensors(self):
        arr_1 = np.arange(32 * 32, dtype=np.int64).reshape(32, 32)
        tensor_1 = inference_pb2.Tensor(
            dtype="int64",
            tensorId="input1",
            buffer=bytes(arr_1),
            shape=[inference_pb2.NamedInt(name="x", size=32), inference_pb2.NamedInt(name="y", size=32)],
        )

        arr_2 = np.arange(64 * 64, dtype=int).reshape(64, 64)
        tensor_2 = inference_pb2.Tensor(
            dtype="int64",
            tensorId="input2",
            buffer=bytes(arr_2),
            shape=[inference_pb2.NamedInt(name="x", size=64), inference_pb2.NamedInt(name="y", size=64)],
        )

        sample = Sample.from_pb_tensors([tensor_1, tensor_2])
        assert len(sample.tensors) == 2
        assert sample.tensors["input1"].equals(xr.DataArray(arr_1, dims=["x", "y"]))
        assert sample.tensors["input2"].equals(xr.DataArray(arr_2, dims=["x", "y"]))

    def test_create_sample_from_raw_data(self):
        arr_1 = np.arange(32 * 32, dtype=np.int64).reshape(32, 32)
        tensor_1 = xr.DataArray(arr_1, dims=["x", "y"])
        arr_2 = np.arange(64 * 64, dtype=np.int64).reshape(64, 64)
        tensor_2 = xr.DataArray(arr_2, dims=["x", "y"])
        tensors_ids = ["input1", "input2"]
        actual_sample = Sample.from_raw_data(tensors_ids, [tensor_1, tensor_2])

        expected_dict = {tensors_ids[0]: tensor_1, tensors_ids[1]: tensor_2}
        expected_sample = Sample(expected_dict)
        assert actual_sample == expected_sample

    def test_sample_to_pb_tensors(self):
        arr_1 = np.arange(32 * 32, dtype=np.int64).reshape(32, 32)
        tensor_1 = xr.DataArray(arr_1, dims=["x", "y"])
        arr_2 = np.arange(64 * 64, dtype=np.int64).reshape(64, 64)
        tensor_2 = xr.DataArray(arr_2, dims=["x", "y"])
        tensors_ids = ["input1", "input2"]
        sample = Sample.from_raw_data(tensors_ids, [tensor_1, tensor_2])

        pb_tensor_1 = inference_pb2.Tensor(
            dtype="int64",
            tensorId="input1",
            buffer=bytes(arr_1),
            shape=[inference_pb2.NamedInt(name="x", size=32), inference_pb2.NamedInt(name="y", size=32)],
        )
        pb_tensor_2 = inference_pb2.Tensor(
            dtype="int64",
            tensorId="input2",
            buffer=bytes(arr_2),
            shape=[inference_pb2.NamedInt(name="x", size=64), inference_pb2.NamedInt(name="y", size=64)],
        )
        expected_tensors = [pb_tensor_1, pb_tensor_2]

        actual_tensors = sample.to_pb_tensors()
        assert expected_tensors == actual_tensors
