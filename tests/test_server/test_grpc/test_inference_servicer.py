from typing import Tuple
from unittest.mock import patch

import grpc
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from tiktorch import converters
from tiktorch.converters import get_axes_with_size, named_shape_to_pb_NamedInts
from tiktorch.proto import inference_pb2, inference_pb2_grpc
from tiktorch.server.data_store import DataStore
from tiktorch.server.device_pool import TorchDevicePool
from tiktorch.server.grpc import InferenceServicer, inference_servicer
from tiktorch.server.session_manager import SessionManager


@pytest.fixture(scope="module")
def data_store():
    return DataStore()


@pytest.fixture(scope="module")
def grpc_add_to_server():
    return inference_pb2_grpc.add_InferenceServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer(data_store):
    return inference_servicer.InferenceServicer(TorchDevicePool(), SessionManager(), data_store)


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    return inference_pb2_grpc.InferenceStub


@pytest.fixture()
def gpu_exists():
    with patch.object(InferenceServicer, "_check_gpu_exists", lambda *args: None):
        yield


def valid_model_request(model_bytes, device_ids=None):
    return inference_pb2.CreateModelSessionRequest(
        model_blob=inference_pb2.Blob(content=model_bytes.getvalue()), deviceIds=device_ids or ["cpu"]
    )


class TestModelManagement:
    @pytest.fixture
    def method_requiring_session(self, request, grpc_stub):
        method_name, req = request.param
        return getattr(grpc_stub, method_name), req

    def test_model_session_creation(self, grpc_stub, bioimageio_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_model_bytes))
        assert model.id
        grpc_stub.CloseModelSession(model)

    def test_model_session_creation_using_upload_id(self, grpc_stub, data_store, bioimageio_dummy_explicit_model_bytes):
        id_ = data_store.put(bioimageio_dummy_explicit_model_bytes.getvalue())

        rq = inference_pb2.CreateModelSessionRequest(model_uri=f"upload://{id_}", deviceIds=["cpu"])
        model = grpc_stub.CreateModelSession(rq)
        assert model.id
        grpc_stub.CloseModelSession(model)

    def test_model_session_creation_using_random_uri(self, grpc_stub):
        rq = inference_pb2.CreateModelSessionRequest(model_uri="randomSchema://", deviceIds=["cpu"])
        with pytest.raises(grpc.RpcError):
            grpc_stub.CreateModelSession(rq)

    def test_model_session_creation_using_non_existent_upload(self, grpc_stub):
        rq = inference_pb2.CreateModelSessionRequest(model_uri="upload://test123", deviceIds=["cpu"])
        with pytest.raises(grpc.RpcError):
            grpc_stub.CreateModelSession(rq)

    def test_predict_call_fails_without_specifying_model_session_id(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as e:
            grpc_stub.Predict(inference_pb2.PredictRequest())

        assert grpc.StatusCode.FAILED_PRECONDITION == e.value.code()
        assert "model-session-id has not been provided" in e.value.details()


class TestDeviceManagement:
    def test_list_devices(self, grpc_stub):
        resp = grpc_stub.ListDevices(inference_pb2.Empty())
        device_by_id = {d.id: d for d in resp.devices}
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

    def _query_devices(self, grpc_stub):
        dev_resp = grpc_stub.ListDevices(inference_pb2.Empty())
        device_by_id = {d.id: d for d in dev_resp.devices}
        return device_by_id

    def test_if_model_create_fails_devices_are_released(self, grpc_stub):
        model_req = inference_pb2.CreateModelSessionRequest(
            model_blob=inference_pb2.Blob(content=b""), deviceIds=["cpu"]
        )

        model = None
        with pytest.raises(Exception):
            model = grpc_stub.CreateModelSession(model_req)

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

        if model:
            grpc_stub.CloseModelSession(model)

    def test_use_device(self, grpc_stub, bioimageio_model_bytes):
        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_model_bytes, device_ids=["cpu"]))

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.IN_USE == device_by_id["cpu"].status

        grpc_stub.CloseModelSession(model)

    def test_using_same_device_fails(self, grpc_stub, bioimageio_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_model_bytes, device_ids=["cpu"]))
        with pytest.raises(grpc.RpcError):
            model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_model_bytes, device_ids=["cpu"]))

        grpc_stub.CloseModelSession(model)

    def test_closing_session_releases_devices(self, grpc_stub, bioimageio_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_model_bytes, device_ids=["cpu"]))

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.IN_USE == device_by_id["cpu"].status

        grpc_stub.CloseModelSession(model)

        device_by_id_after_close = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id_after_close
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id_after_close["cpu"].status


class TestGetLogs:
    def test_returns_ack_message(self, bioimageio_model_bytes, grpc_stub):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_model_bytes))
        resp = grpc_stub.GetLogs(inference_pb2.Empty())
        record = next(resp)
        assert inference_pb2.LogEntry.Level.INFO == record.level
        assert "Sending model logs" == record.content
        grpc_stub.CloseModelSession(model)


class TestForwardPass:
    def test_call_fails_with_unknown_model_session_id(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as e:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId="myid1"))
        assert grpc.StatusCode.FAILED_PRECONDITION == e.value.code()
        assert "model-session with id myid1 doesn't exist" in e.value.details()

    def test_call_predict_valid_explicit(self, grpc_stub, bioimageio_dummy_explicit_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_explicit_model_bytes))
        arr = xr.DataArray(np.arange(128 * 128).reshape(1, 1, 128, 128), dims=("b", "c", "x", "y"))
        expected = arr + 1
        input_tensor_id = "input"
        output_tensor_id = "output"
        input_tensors = [converters.xarray_to_pb_tensor(input_tensor_id, arr)]
        res = grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))

        grpc_stub.CloseModelSession(model)

        assert len(res.tensors) == 1
        assert res.tensors[0].tensorId == output_tensor_id
        assert_array_equal(expected, converters.pb_tensor_to_numpy(res.tensors[0]))

    def test_call_predict_invalid_shape_explicit(self, grpc_stub, bioimageio_dummy_explicit_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_explicit_model_bytes))
        arr = xr.DataArray(np.arange(32 * 32).reshape(1, 1, 32, 32), dims=("b", "c", "x", "y"))
        input_tensors = [converters.xarray_to_pb_tensor("input", arr)]
        with pytest.raises(grpc.RpcError):
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        grpc_stub.CloseModelSession(model)

    @pytest.mark.parametrize(
        "shape",
        [(1, 1, 64, 32), (1, 1, 32, 64), (1, 1, 64, 32), (0, 1, 64, 64), (1, 0, 64, 64)],
    )
    def test_call_predict_invalid_shape_parameterized(self, grpc_stub, shape, bioimageio_dummy_param_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_param_model_bytes))
        arr = xr.DataArray(np.arange(np.prod(shape)).reshape(*shape), dims=("b", "c", "x", "y"))
        input_tensors = [converters.xarray_to_pb_tensor("param", arr)]
        with pytest.raises(grpc.RpcError):
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        grpc_stub.CloseModelSession(model)

    def test_call_predict_invalid_tensor_ids(self, grpc_stub, bioimageio_dummy_model):
        model_bytes, _ = bioimageio_dummy_model
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(32 * 32).reshape(32, 32), dims=("x", "y"))
        input_tensors = [converters.xarray_to_pb_tensor("invalidTensorName", arr)]
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert error.value.details().startswith("Exception calling application: Spec invalidTensorName doesn't exist")
        grpc_stub.CloseModelSession(model)

    def test_call_predict_invalid_axes(self, grpc_stub, bioimageio_dummy_model):
        model_bytes, tensor_id = bioimageio_dummy_model
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(32 * 32).reshape(32, 32), dims=("invalidAxis", "y"))
        input_tensors = [converters.xarray_to_pb_tensor(tensor_id, arr)]
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert error.value.details().startswith("Exception calling application: Incompatible axes")
        grpc_stub.CloseModelSession(model)

    @pytest.mark.parametrize("shape", [(1, 1, 64, 64), (1, 1, 66, 65), (1, 1, 68, 66), (1, 1, 70, 67)])
    def test_call_predict_valid_shape_parameterized(self, grpc_stub, shape, bioimageio_dummy_param_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_param_model_bytes))
        arr = xr.DataArray(np.arange(np.prod(shape)).reshape(*shape), dims=("b", "c", "x", "y"))
        input_tensors = [converters.xarray_to_pb_tensor("param", arr)]
        grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        grpc_stub.CloseModelSession(model)

    @pytest.mark.skip
    def test_call_predict_tf(self, grpc_stub, bioimageio_dummy_tensorflow_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_tensorflow_model_bytes))
        arr = xr.DataArray(np.arange(32 * 32).reshape(1, 1, 32, 32), dims=("b", "c", "x", "y"))
        expected = arr * -1
        input_tensor_id = "input"
        output_tensor_id = "output"
        input_tensors = [converters.xarray_to_pb_tensor(input_tensor_id, arr)]
        res = grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))

        grpc_stub.CloseModelSession(model)

        assert len(res.tensors) == 1
        assert res.tensors[0].tensorId == output_tensor_id
        assert_array_equal(expected, converters.pb_tensor_to_numpy(res.tensors[0]))


class TestCudaMemory:
    MAX_SHAPE = (1, 1, 10, 10)
    AXES = ("b", "c", "y", "x")

    def to_pb_namedInts(self, shape: Tuple[int, ...]) -> inference_pb2.NamedInts:
        return named_shape_to_pb_NamedInts(get_axes_with_size(self.AXES, shape))

    @pytest.mark.parametrize(
        "min_shape, max_shape, step_shape, expected",
        [
            ((1, 1, 5, 5), (1, 1, 11, 11), (0, 0, 1, 1), MAX_SHAPE),
            ((1, 1, 5, 5), (1, 1, 6, 6), (0, 0, 1, 1), [1, 1, 6, 6]),
        ],
    )
    def test_max_cuda_memory(
        self,
        gpu_exists,
        min_shape,
        max_shape,
        step_shape,
        expected,
        grpc_stub,
        bioimageio_dummy_cuda_out_of_memory_model_bytes,
    ):
        min_shape = self.to_pb_namedInts(min_shape)
        max_shape = self.to_pb_namedInts(max_shape)
        step_shape = self.to_pb_namedInts(step_shape)

        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_cuda_out_of_memory_model_bytes))
        res = grpc_stub.MaxCudaMemoryShape(
            inference_pb2.MaxCudaMemoryShapeRequest(
                modelSessionId=model.id,
                tensorId="input",
                deviceId="cuda:0",
                minShape=min_shape,
                maxShape=max_shape,
                stepShape=step_shape,
            )
        )
        grpc_stub.CloseModelSession(model)
        assert res.maxShape == self.to_pb_namedInts(expected)

    @pytest.mark.parametrize(
        "min_shape, max_shape, step_shape, description",
        [
            ((1, 1, 6, 6), (1, 1, 5, 5), (0, 0, 1, 1), "Max shape [1 1 5 5] smaller than min shape [1 1 6 6]"),
            ((1, 1, 5, 5), (1, 1, 6, 6), (0, 0, 2, 1), "Invalid parameterized shape"),
        ],
    )
    def test_max_cuda_memory_invalid_request(
        self,
        description,
        gpu_exists,
        min_shape,
        max_shape,
        step_shape,
        grpc_stub,
        bioimageio_dummy_cuda_out_of_memory_model_bytes,
    ):
        min_shape = self.to_pb_namedInts(min_shape)
        max_shape = self.to_pb_namedInts(max_shape)
        step_shape = self.to_pb_namedInts(step_shape)

        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_cuda_out_of_memory_model_bytes))
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.MaxCudaMemoryShape(
                inference_pb2.MaxCudaMemoryShapeRequest(
                    modelSessionId=model.id,
                    tensorId="input",
                    deviceId="cuda:0",
                    minShape=min_shape,
                    maxShape=max_shape,
                    stepShape=step_shape,
                )
            )
        assert error.value.details().startswith(f"Exception calling application: {description}")
        grpc_stub.CloseModelSession(model)

    def test_max_cuda_memory_not_found(self, gpu_exists, grpc_stub, bioimageio_dummy_cuda_out_of_memory_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_cuda_out_of_memory_model_bytes))
        min_shape = self.to_pb_namedInts((1, 1, 11, 11))
        max_shape = self.to_pb_namedInts((1, 1, 12, 12))
        step = self.to_pb_namedInts((0, 0, 1, 1))
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.MaxCudaMemoryShape(
                inference_pb2.MaxCudaMemoryShapeRequest(
                    modelSessionId=model.id,
                    tensorId="input",
                    deviceId="cuda:0",
                    minShape=min_shape,
                    maxShape=max_shape,
                    stepShape=step,
                )
            )
        assert error.value.code() == grpc.StatusCode.NOT_FOUND
        assert error.value.details() == "no valid shape"
        grpc_stub.CloseModelSession(model)

    @pytest.mark.parametrize(
        "shape, expected",
        [((1, 1, 10, 10), False), ((1, 1, 99, 99), True)],
    )
    def test_is_out_of_memory(
        self, gpu_exists, shape, expected, grpc_stub, bioimageio_dummy_cuda_out_of_memory_model_bytes
    ):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_cuda_out_of_memory_model_bytes))
        shape = self.to_pb_namedInts(shape)
        res = grpc_stub.IsCudaOutOfMemory(
            inference_pb2.IsCudaOutOfMemoryRequest(
                modelSessionId=model.id, tensorId="input", deviceId="cuda:0", shape=shape
            )
        )
        grpc_stub.CloseModelSession(model)
        assert res.isCudaOutOfMemory is expected
