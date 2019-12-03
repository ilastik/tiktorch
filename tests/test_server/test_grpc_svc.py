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
        methods = [
            # ("ListDevices", inference_pb2.Empty()), TODO: Devices state in session
            ("Predict", inference_pb2.PredictRequest()),
            ("UseDevices", inference_pb2.Devices()),
            # ("GetLogs", inference_pb2.Empty()), FIXME: Streaming requires one iteration to start before termination
        ]
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
            res = method(req)
            print(type(res))

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

    def test_closing_session(self, grpc_stub, method_requiring_session):
        method, req = method_requiring_session

        session = grpc_stub.CreateSession(inference_pb2.Empty())
        grpc_stub.CloseSession(session)

        with pytest.raises(grpc.RpcError) as e:
            method(req, metadata=[("session-id", session.id)])
            assert f"session with id {session.id} doesn't exist" in e.value.details()


class TestDeviceManagement:
    def test_list_devices(self, grpc_stub):
        session = grpc_stub.CreateSession(inference_pb2.Empty())
        resp = grpc_stub.ListDevices(inference_pb2.Empty(), metadata=[("session-id", session.id)])
        device_by_id = {d.id: d for d in resp.devices}
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

    def test_closing_session_releases_devices(self, grpc_stub):
        session = grpc_stub.CreateSession(inference_pb2.Empty())
        devs = grpc_stub.ListDevices(inference_pb2.Empty(), metadata=[("session-id", session.id)])
        resp = grpc_stub.UseDevices(devs, metadata=[("session-id", session.id)])

        in_use_devices = grpc_stub.ListDevices(inference_pb2.Empty(), metadata=[("session-id", session.id)])
        assert len(in_use_devices.devices) >= 1
        for dev in in_use_devices.devices:
            assert inference_pb2.Device.Status.IN_USE == dev.status

        grpc_stub.CloseSession(session)

        session2 = grpc_stub.CreateSession(inference_pb2.Empty())
        avail_devices = grpc_stub.ListDevices(inference_pb2.Empty(), metadata=[("session-id", session2.id)])

        assert len(avail_devices.devices) >= 1
        for dev in avail_devices.devices:
            assert inference_pb2.Device.Status.AVAILABLE == dev.status

    def test_use_device(self, grpc_stub):
        session = grpc_stub.CreateSession(inference_pb2.Empty())
        resp = grpc_stub.ListDevices(inference_pb2.Empty(), metadata=[("session-id", session.id)])

        device_by_id = {d.id: d for d in resp.devices}
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.AVAILABLE == device_by_id["cpu"].status

        to_use = inference_pb2.Devices(devices=[device_by_id["cpu"]])
        resp = grpc_stub.UseDevices(to_use, metadata=[("session-id", session.id)])

        new_dev_resp = grpc_stub.ListDevices(inference_pb2.Empty(), metadata=[("session-id", session.id)])

        device_by_id = {d.id: d for d in new_dev_resp.devices}
        assert "cpu" in device_by_id
        assert inference_pb2.Device.Status.IN_USE == device_by_id["cpu"].status

        grpc_stub.CloseSession(session)


class TestGetLogs:
    def test_returns_ack_message(self, grpc_stub):
        session = grpc_stub.CreateSession(inference_pb2.Empty())
        resp = grpc_stub.GetLogs(inference_pb2.Empty(), metadata=[("session-id", session.id)])
        record = next(resp)
        assert inference_pb2.LogEntry.Level.INFO == record.level
        assert "Sending session logs" == record.content
