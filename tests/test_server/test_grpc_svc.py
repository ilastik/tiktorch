import pytest
import grpc

from tiktorch.proto import inference_pb2, inference_pb2_grpc
from tiktorch.server.device_manager import IDeviceManager, TorchDeviceManager
from tiktorch.server.session_manager import SessionManager


@pytest.fixture(scope="module")
def grpc_add_to_server():
    from tiktorch.proto.inference_pb2_grpc import add_InferenceServicer_to_server

    return add_InferenceServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer():
    from tiktorch.server import grpc_svc

    return grpc_svc.InferenceServicer(TorchDeviceManager(), SessionManager())


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    from tiktorch.proto.inference_pb2_grpc import InferenceStub

    return InferenceStub


def pytest_generate_tests(metafunc):
    if "method_requiring_session" in metafunc.fixturenames:
        methods = [("ListAvailableDevices", inference_pb2.Empty()), ("Predict", inference_pb2.PredictRequest())]
        metafunc.parametrize(
            "method_requiring_session", methods, ids=[method_name for method_name, _ in methods], indirect=True
        )


class TestSessionManagement:
    @pytest.fixture
    def method_requiring_session(self, request, grpc_stub):
        method_name, req = request.param
        return getattr(grpc_stub, method_name), req

    def test_session_creation(self, grpc_stub):
        session = grpc_stub.CreateSession(inference_pb2.Empty())
        assert session.id

    def test_call_fails_without_specifying_session_id(self, grpc_stub, method_requiring_session):
        method, req = method_requiring_session
        with pytest.raises(grpc.RpcError) as e:
            method(req)

        assert grpc.StatusCode.FAILED_PRECONDITION == e.value.code()
        assert "session-id has not been provided" in e.value.details()

    def test_call_fails_with_unknown_session_id(self, grpc_stub, method_requiring_session):
        method, req = method_requiring_session
        with pytest.raises(grpc.RpcError) as e:
            method(req, metadata=[("session-id", "test")])

        assert grpc.StatusCode.FAILED_PRECONDITION == e.value.code()
        assert "session with id test doesn't exist" in e.value.details()

    def test_call_succeeds_with_valid_session_id(self, grpc_stub, method_requiring_session):
        method, req = method_requiring_session
        session = grpc_stub.CreateSession(inference_pb2.Empty())
        _, call = method.with_call(req, metadata=[("session-id", session.id)])
        assert grpc.StatusCode.OK == call.code()


class TestListAvailableDevicer:
    def test_list_available_devices(self, grpc_stub):
        session = grpc_stub.CreateSession(inference_pb2.Empty())
        resp = grpc_stub.ListAvailableDevices(inference_pb2.Empty(), metadata=[("session-id", session.id)])
        assert "cpu" in [d.id for d in resp.devices]
