import time
from concurrent import futures

import grpc

from tiktorch.proto import inference_pb2, inference_pb2_grpc
from tiktorch.server.device_manager import IDeviceManager, TorchDeviceManager
from tiktorch.server.session_manager import SessionManager

_ONE_DAY_IN_SECONDS = 24 * 60 * 60


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, device_manager: IDeviceManager, session_manager: SessionManager) -> None:
        self.__device_manager = device_manager
        self.__session_manager = session_manager

    def _get_session(self, context):
        meta = dict(context.invocation_metadata())
        session_id = meta.get("session-id", None)

        if session_id is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "session-id has not been provided by client")

        session = self.__session_manager.get(session_id)

        if session is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"session with id {session_id} doesn't exist")

        return session

    def CreateSession(self, request: inference_pb2.Empty, context) -> inference_pb2.Session:
        session = self.__session_manager.create_session()
        return inference_pb2.Session(id=session.id)

    def ListAvailableDevices(self, request: inference_pb2.Empty, context) -> inference_pb2.Devices:
        session = self._get_session(context)
        devices = self.__device_manager.list_available_devices()
        return inference_pb2.Devices(names=devices)

    def Predict(self, request: inference_pb2.PredictRequest, context) -> inference_pb2.PredictResponse:
        session = self._get_session(context)
        return inference_pb2.PredictResponse()


def serve(host, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    session_svc = InferenceServicer(TorchDeviceManager(), SessionManager())
    inference_pb2_grpc.add_InferenceServicer_to_server(session_svc, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
