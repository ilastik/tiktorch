import io

import grpc
import numpy as np
import pytest
import torch
import xarray as xr
from numpy.testing import assert_array_equal

from tiktorch import converters
from tiktorch.converters import pb_tensor_to_xarray
from tiktorch.proto import inference_pb2, inference_pb2_grpc, utils_pb2
from tiktorch.server.data_store import DataStore
from tiktorch.server.device_pool import TorchDevicePool
from tiktorch.server.grpc import inference_servicer
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


@pytest.fixture(autouse=True)
def clean(grpc_servicer):
    yield
    grpc_servicer.close_all_sessions()


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    return inference_pb2_grpc.InferenceStub


def valid_model_request(model_bytes: io.BytesIO, device_ids=None):
    ret = inference_pb2.CreateModelSessionRequest(
        model_blob=inference_pb2.Blob(content=model_bytes.getvalue()), deviceIds=device_ids or ["cpu"]
    )
    return ret


class TestModelManagement:
    @pytest.fixture
    def method_requiring_session(self, request, grpc_stub):
        method_name, req = request.param
        return getattr(grpc_stub, method_name), req

    def test_model_session_creation_using_upload_id(
        self, grpc_stub, data_store, bioimage_model_explicit_add_one_siso_v5
    ):
        model_bytes = bioimage_model_explicit_add_one_siso_v5
        id_ = data_store.put(model_bytes.getvalue())

        rq = inference_pb2.CreateModelSessionRequest(model_uri=f"upload://{id_}", deviceIds=["cpu"])
        model = grpc_stub.CreateModelSession(rq)
        assert model.id

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

    def test_model_init_failed_close_session(self, bioimage_model_explicit_add_one_siso_v5, grpc_stub):
        """
        If the model initialization fails, the session should be closed, so we can initialize a new one
        """

        model_req = inference_pb2.CreateModelSessionRequest(
            model_blob=inference_pb2.Blob(content=b""), deviceIds=["cpu"]
        )

        with pytest.raises(Exception):
            grpc_stub.CreateModelSession(model_req)

        model_bytes = bioimage_model_explicit_add_one_siso_v5
        response = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        assert response.id is not None


class TestDeviceManagement:
    def test_list_devices(self, grpc_stub):
        resp = grpc_stub.ListDevices(utils_pb2.Empty())
        device_by_id = {d.id: d for d in resp.devices}
        assert "cpu" in device_by_id
        assert utils_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

    def _query_devices(self, grpc_stub):
        dev_resp = grpc_stub.ListDevices(utils_pb2.Empty())
        device_by_id = {d.id: d for d in dev_resp.devices}
        return device_by_id

    def test_if_model_create_fails_devices_are_released(self, grpc_stub):
        model_req = inference_pb2.CreateModelSessionRequest(
            model_blob=inference_pb2.Blob(content=b""), deviceIds=["cpu"]
        )

        with pytest.raises(Exception):
            grpc_stub.CreateModelSession(model_req)

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert utils_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

    def test_use_device(self, grpc_stub, bioimage_model_explicit_add_one_siso_v5):
        model_bytes = bioimage_model_explicit_add_one_siso_v5
        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert utils_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

        grpc_stub.CreateModelSession(valid_model_request(model_bytes, device_ids=["cpu"]))

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert utils_pb2.Device.Status.IN_USE == device_by_id["cpu"].status

    def test_using_same_device_fails(self, grpc_stub, bioimage_model_explicit_add_one_siso_v5):
        model_bytes = bioimage_model_explicit_add_one_siso_v5
        grpc_stub.CreateModelSession(valid_model_request(model_bytes, device_ids=["cpu"]))
        with pytest.raises(grpc.RpcError):
            grpc_stub.CreateModelSession(valid_model_request(model_bytes, device_ids=["cpu"]))

    def test_closing_session_releases_devices(self, grpc_stub, bioimage_model_explicit_add_one_siso_v5):
        model_bytes = bioimage_model_explicit_add_one_siso_v5
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes, device_ids=["cpu"]))

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert utils_pb2.Device.Status.IN_USE == device_by_id["cpu"].status

        grpc_stub.CloseModelSession(model)

        device_by_id_after_close = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id_after_close
        assert utils_pb2.Device.Status.AVAILABLE == device_by_id_after_close["cpu"].status


class TestGetLogs:
    def test_returns_ack_message(self, bioimage_model_explicit_add_one_siso_v5, grpc_stub):
        model_bytes = bioimage_model_explicit_add_one_siso_v5
        grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        resp = grpc_stub.GetLogs(utils_pb2.Empty())
        record = next(resp)
        assert inference_pb2.LogEntry.Level.INFO == record.level
        assert "Sending model logs" == record.content


class TestForwardPass:
    def test_call_fails_with_unknown_model_session_id(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as e:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId="myid1"))
        assert grpc.StatusCode.FAILED_PRECONDITION == e.value.code()
        assert "model-session with id myid1 doesn't exist" in e.value.details()

    def test_call_predict_valid_explicit(self, grpc_stub, bioimage_model_explicit_add_one_siso_v5):
        model_bytes = bioimage_model_explicit_add_one_siso_v5
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(2 * 10 * 20).reshape(1, 2, 10, 20), dims=("batch", "channel", "x", "y"))
        input_tensor_id = "input"
        input_tensors = [converters.xarray_to_pb_tensor(input_tensor_id, arr)]
        res = grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert len(res.tensors) == 1
        pb_tensor = res.tensors[0]
        assert pb_tensor.tensorId == "output"
        expected_output = xr.DataArray(torch.from_numpy(arr.values + 1).numpy(), dims=arr.dims)
        assert_array_equal(pb_tensor_to_xarray(res.tensors[0]), expected_output)

    def test_call_predict_valid_explicit_v4(self, grpc_stub, bioimage_model_add_one_v4):
        model_bytes = bioimage_model_add_one_v4
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(2 * 10 * 20).reshape(1, 2, 10, 20), dims=("batch", "channel", "x", "y"))
        input_tensor_id = "input"
        input_tensors = [converters.xarray_to_pb_tensor(input_tensor_id, arr)]
        res = grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert len(res.tensors) == 1
        pb_tensor = res.tensors[0]
        assert pb_tensor.tensorId == "output"
        expected_output = xr.DataArray(torch.from_numpy(arr.values + 1).numpy(), dims=arr.dims)
        assert_array_equal(pb_tensor_to_xarray(res.tensors[0]), expected_output)

    def test_call_predict_invalid_shape_explicit(self, grpc_stub, bioimage_model_explicit_add_one_siso_v5):
        model_bytes = bioimage_model_explicit_add_one_siso_v5
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(32 * 32).reshape(1, 1, 32, 32), dims=("batch", "channel", "x", "y"))
        input_tensors = [converters.xarray_to_pb_tensor("input", arr)]
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert error.value.details().startswith("Exception calling application: Incompatible axis")

    def test_call_predict_multiple_inputs_with_reference(self, grpc_stub, bioimage_model_add_one_miso_v5):
        model_bytes = bioimage_model_add_one_miso_v5
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))

        arr1 = xr.DataArray(np.arange(2 * 10 * 20).reshape(1, 2, 10, 20), dims=("batch", "channel", "x", "y"))
        input_tensor_id1 = "input1"

        arr2 = xr.DataArray(np.arange(2 * 12 * 21).reshape(1, 2, 12, 21), dims=("batch", "channel", "x", "y"))
        input_tensor_id2 = "input2"

        arr3 = xr.DataArray(np.arange(2 * 14 * 20).reshape(1, 2, 14, 20), dims=("batch", "channel", "x", "y"))
        input_tensor_id3 = "input3"

        input_tensor_ids = [input_tensor_id1, input_tensor_id2, input_tensor_id3]
        tensors_arr = [arr1, arr2, arr3]
        input_tensors = [
            converters.xarray_to_pb_tensor(tensor_id, arr) for tensor_id, arr in zip(input_tensor_ids, tensors_arr)
        ]

        res = grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        grpc_stub.CloseModelSession(model)
        assert len(res.tensors) == 1
        pb_tensor = res.tensors[0]
        assert pb_tensor.tensorId == "output"
        expected_output = xr.DataArray(torch.from_numpy(arr1.values + 1).numpy(), dims=arr1.dims)
        assert_array_equal(pb_tensor_to_xarray(res.tensors[0]), expected_output)

    @pytest.mark.parametrize("shape", [(1, 2, 10, 20), (1, 2, 12, 20), (1, 2, 10, 23), (1, 2, 12, 23)])
    def test_call_predict_valid_shape_parameterized(self, grpc_stub, shape, bioimage_model_param_add_one_siso_v5):
        model_bytes = bioimage_model_param_add_one_siso_v5
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(np.prod(shape)).reshape(*shape), dims=("batch", "channel", "x", "y"))
        input_tensor_id = "input"
        input_tensors = [converters.xarray_to_pb_tensor(input_tensor_id, arr)]
        grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))

    @pytest.mark.parametrize(
        "shape",
        [(1, 1, 10, 20), (1, 2, 8, 20), (1, 2, 11, 20), (1, 2, 10, 21)],
    )
    def test_call_predict_invalid_shape_parameterized(self, grpc_stub, shape, bioimage_model_param_add_one_siso_v5):
        model_bytes = bioimage_model_param_add_one_siso_v5
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(np.prod(shape)).reshape(*shape), dims=("batch", "channel", "x", "y"))
        input_tensor_id = "input"
        input_tensors = [converters.xarray_to_pb_tensor(input_tensor_id, arr)]
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert error.value.details().startswith("Exception calling application: Incompatible axis")

    def test_call_predict_invalid_tensor_ids(self, grpc_stub, bioimage_model_explicit_add_one_siso_v5):
        model_bytes = bioimage_model_explicit_add_one_siso_v5
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(2 * 10 * 20).reshape(1, 2, 10, 20), dims=("batch", "channel", "x", "y"))
        input_tensors = [converters.xarray_to_pb_tensor("invalidTensorName", arr)]
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert error.value.details().startswith("Exception calling application: Spec 'invalidTensorName' doesn't exist")

    @pytest.mark.parametrize(
        "axes",
        [
            ("channel", "batch", "x", "y"),
            ("time", "channel", "x", "y"),
            ("batch", "channel", "z", "y"),
            ("b", "c", "x", "y"),
        ],
    )
    def test_call_predict_invalid_axes(self, grpc_stub, axes, bioimage_model_explicit_add_one_siso_v5):
        model_bytes = bioimage_model_explicit_add_one_siso_v5
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(2 * 10 * 20).reshape(1, 2, 10, 20), dims=axes)
        input_tensor_id = "input"
        input_tensors = [converters.xarray_to_pb_tensor(input_tensor_id, arr)]
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert error.value.details().startswith("Exception calling application: Incompatible axes names")
