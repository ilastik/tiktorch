import time
from concurrent import futures

import grpc

from tiktorch.proto import inference_pb2, inference_pb2_grpc
from tiktorch.server.device_manager import IDeviceManager, TorchDeviceManager
from tiktorch.server.session_manager import SessionManager

_ONE_DAY_IN_SECONDS = 24 * 60 * 60


class SessionServicer(inference_pb2_grpc.SessionProviderServicer):
    def __init__(self, device_manager: IDeviceManager, session_manager: SessionManager) -> None:
        self.__device_manager = device_manager
        self.__session_manager = session_manager

    def CreateSession(self, request: inference_pb2.Empty, context) -> inference_pb2.Session:
        session = self.__session_manager.create_session()
        return inference_pb2.Session(id=session.id)

    def GetDevices(self, request: inference_pb2.Empty, context) -> inference_pb2.Devices:
        devices = self.__device_manager.list_available_devices()
        return inference_pb2.Devices(names=devices)


def serve(host, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    session_svc = SessionServicer(TorchDeviceManager(), SessionManager())
    inference_pb2_grpc.add_SessionProviderServicer_to_server(session_svc, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
