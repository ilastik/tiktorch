import grpc
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from tiktorch import converters
from tiktorch.proto import inference_pb2, inference_pb2_grpc
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

    def test_model_session_creation_using_upload_id(self, grpc_stub, data_store, bioimageio_dummy_explicit_model_bytes):
        id_ = data_store.put(bioimageio_dummy_explicit_model_bytes.getvalue())

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

        with pytest.raises(Exception):
            grpc_stub.CreateModelSession(model_req)

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

    def test_use_device(self, grpc_stub, bioimageio_model_bytes):
        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

        grpc_stub.CreateModelSession(valid_model_request(bioimageio_model_bytes, device_ids=["cpu"]))

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.IN_USE == device_by_id["cpu"].status

    def test_using_same_device_fails(self, grpc_stub, bioimageio_model_bytes):
        grpc_stub.CreateModelSession(valid_model_request(bioimageio_model_bytes, device_ids=["cpu"]))
        with pytest.raises(grpc.RpcError):
            grpc_stub.CreateModelSession(valid_model_request(bioimageio_model_bytes, device_ids=["cpu"]))

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

        assert len(res.tensors) == 1
        assert res.tensors[0].tensorId == output_tensor_id
        assert_array_equal(expected, converters.pb_tensor_to_numpy(res.tensors[0]))

    def test_call_predict_invalid_shape_explicit(self, grpc_stub, bioimageio_dummy_explicit_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_explicit_model_bytes))
        arr = xr.DataArray(np.arange(32 * 32).reshape(1, 1, 32, 32), dims=("b", "c", "x", "y"))
        input_tensors = [converters.xarray_to_pb_tensor("input", arr)]
        with pytest.raises(grpc.RpcError):
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))

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

    def test_call_predict_invalid_tensor_ids(self, grpc_stub, bioimageio_dummy_model):
        model_bytes, _ = bioimageio_dummy_model
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(32 * 32).reshape(32, 32), dims=("x", "y"))
        input_tensors = [converters.xarray_to_pb_tensor("invalidTensorName", arr)]
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert error.value.details().startswith("Exception calling application: Spec invalidTensorName doesn't exist")

    def test_call_predict_invalid_axes(self, grpc_stub, bioimageio_dummy_model):
        model_bytes, tensor_id = bioimageio_dummy_model
        model = grpc_stub.CreateModelSession(valid_model_request(model_bytes))
        arr = xr.DataArray(np.arange(32 * 32).reshape(32, 32), dims=("invalidAxis", "y"))
        input_tensors = [converters.xarray_to_pb_tensor(tensor_id, arr)]
        with pytest.raises(grpc.RpcError) as error:
            grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))
        assert error.value.details().startswith("Exception calling application: Incompatible axes")

    @pytest.mark.parametrize("shape", [(1, 1, 64, 64), (1, 1, 66, 65), (1, 1, 68, 66), (1, 1, 70, 67)])
    def test_call_predict_valid_shape_parameterized(self, grpc_stub, shape, bioimageio_dummy_param_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_param_model_bytes))
        arr = xr.DataArray(np.arange(np.prod(shape)).reshape(*shape), dims=("b", "c", "x", "y"))
        input_tensors = [converters.xarray_to_pb_tensor("param", arr)]
        grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))

    @pytest.mark.skip
    def test_call_predict_tf(self, grpc_stub, bioimageio_dummy_tensorflow_model_bytes):
        model = grpc_stub.CreateModelSession(valid_model_request(bioimageio_dummy_tensorflow_model_bytes))
        arr = xr.DataArray(np.arange(32 * 32).reshape(1, 1, 32, 32), dims=("b", "c", "x", "y"))
        expected = arr * -1
        input_tensor_id = "input"
        output_tensor_id = "output"
        input_tensors = [converters.xarray_to_pb_tensor(input_tensor_id, arr)]
        res = grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensors=input_tensors))

        assert len(res.tensors) == 1
        assert res.tensors[0].tensorId == output_tensor_id
        assert_array_equal(expected, converters.pb_tensor_to_numpy(res.tensors[0]))
