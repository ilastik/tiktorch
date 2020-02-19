import pytest
import grpc
import numpy as np
from numpy.testing import assert_array_equal

from tiktorch import converters
import inference_pb2, inference_pb2_grpc
from tiktorch.server.device_pool import IDevicePool, TorchDevicePool
from tiktorch.server.session_manager import SessionManager


@pytest.fixture(scope="module")
def grpc_add_to_server():
    from tiktorch.proto.inference_pb2_grpc import add_InferenceServicer_to_server

    return add_InferenceServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer():
    from tiktorch.server import grpc_svc

    return grpc_svc.InferenceServicer(TorchDevicePool(), SessionManager())


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    from tiktorch.proto.inference_pb2_grpc import InferenceStub

    return InferenceStub


def valid_model_request(model_bytes, device_ids=None):
    return inference_pb2.CreateModelSessionRequest(
        model_blob=inference_pb2.Blob(content=model_bytes.getvalue()), deviceIds=device_ids or ["cpu"]
    )


class TestModelManagement:
    @pytest.fixture
    def method_requiring_session(self, request, grpc_stub):
        method_name, req = request.param
        return getattr(grpc_stub, method_name), req

    def test_model_session_creation(self, grpc_stub, pybio_unet_zip):
        model = grpc_stub.CreateModelSession(valid_model_request(pybio_unet_zip))
        assert model.id
        grpc_stub.CloseModelSession(model)

    def test_predict_call_fails_without_specifying_model_session_id(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as e:
            res = grpc_stub.Predict(inference_pb2.PredictRequest())

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

    def test_use_device(self, grpc_stub, pybio_unet_zip):
        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

        model = grpc_stub.CreateModelSession(valid_model_request(pybio_unet_zip, device_ids=["cpu"]))

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.IN_USE == device_by_id["cpu"].status

        grpc_stub.CloseModelSession(model)

    def test_using_same_device_fails(self, grpc_stub, pybio_unet_zip):
        model = grpc_stub.CreateModelSession(valid_model_request(pybio_unet_zip, device_ids=["cpu"]))
        with pytest.raises(grpc.RpcError) as e:
            model = grpc_stub.CreateModelSession(valid_model_request(pybio_unet_zip, device_ids=["cpu"]))

        grpc_stub.CloseModelSession(model)

    def test_closing_session_releases_devices(self, grpc_stub, pybio_unet_zip):
        model = grpc_stub.CreateModelSession(valid_model_request(pybio_unet_zip, device_ids=["cpu"]))

        device_by_id = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.IN_USE == device_by_id["cpu"].status

        grpc_stub.CloseModelSession(model)

        device_by_id_after_close = self._query_devices(grpc_stub)
        assert "cpu" in device_by_id_after_close
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id_after_close["cpu"].status


class TestGetLogs:
    def test_returns_ack_message(self, pybio_unet_zip, grpc_stub):
        model = grpc_stub.CreateModelSession(valid_model_request(pybio_unet_zip))
        resp = grpc_stub.GetLogs(inference_pb2.Empty())
        record = next(resp)
        assert inference_pb2.LogEntry.Level.INFO == record.level
        assert "Sending model logs" == record.content
        grpc_stub.CloseModelSession(model)


class TestForwardPass:
    def test_call_fails_with_unknown_model_session_id(self, grpc_stub):
        with pytest.raises(grpc.RpcError) as e:
            res = grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId="myid1"))
        assert grpc.StatusCode.FAILED_PRECONDITION == e.value.code()
        assert "model-session with id myid1 doesn't exist" in e.value.details()

    def test_call_predict(self, grpc_stub, pybio_dummy_zip):
        model = grpc_stub.CreateModelSession(valid_model_request(pybio_dummy_zip))

        arr = np.arange(32 * 32).reshape(1, 1, 32, 32)
        expected = arr + 1
        input_tensor = converters.numpy_to_pb_tensor(arr)
        res = grpc_stub.Predict(inference_pb2.PredictRequest(modelSessionId=model.id, tensor=input_tensor))

        grpc_stub.CloseModelSession(model)

        assert_array_equal(expected, converters.pb_tensor_to_numpy(res.tensor))
