import time
from concurrent import futures
import multiprocessing as mp

import grpc

from tiktorch.rpc.mp import MPClient, MPServer, Shutdown, create_client
from tiktorch.proto import inference_pb2, inference_pb2_grpc
from tiktorch.server.device_pool import IDevicePool, TorchDevicePool, DeviceStatus
from tiktorch.server.session_manager import SessionManager, ISession


_ONE_DAY_IN_SECONDS = 24 * 60 * 60


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, device_pool: IDevicePool, session_manager: SessionManager) -> None:
        self.__device_pool = device_pool
        self.__session_manager = session_manager

    def CreateModelSession(
        self, request: inference_pb2.CreateModelSessionRequest, context
    ) -> inference_pb2.ModelSession:
        lease = self.__device_pool.lease(request.deviceIds)
        session = self.__session_manager.create_session()
        session.on_close(lease.terminate)
        # model.on_close(lease.terminate)
        # handler2inference_conn, inference2handler_conn = mp.Pipe()
        # self._inference_proc = mp.Process(
        #     target=inference.run, name="Inference", kwargs={"conn": inference2handler_conn}
        # )
        # self._inference_proc.start()
        # self._inference = create_client(inference.IInference, handler2inference_conn)
        # self._inference.load_model(request.model_blob).result()
        return inference_pb2.ModelSession(id=session.id)

    def CloseModelSession(self, request: inference_pb2.ModelSession, context) -> inference_pb2.Empty:
        self.__session_manager.close_session(request.id)
        return inference_pb2.Empty()

    def GetLogs(self, request: inference_pb2.Empty, context):
        yield inference_pb2.LogEntry(
            timestamp=int(time.time()), level=inference_pb2.LogEntry.Level.INFO, content="Sending model logs"
        )

    def ListDevices(self, request: inference_pb2.Empty, context) -> inference_pb2.Devices:
        devices = self.__device_pool.list_devices()
        pb_devices = []
        for dev in devices:
            if dev.status == DeviceStatus.AVAILABLE:
                pb_status = inference_pb2.Device.Status.AVAILABLE
            elif dev.status == DeviceStatus.IN_USE:
                pb_status = inference_pb2.Device.Status.IN_USE
            else:
                raise ValueError(f"Unknown status value {dev.status}")

            pb_devices.append(inference_pb2.Device(id=dev.id, status=pb_status))

        return inference_pb2.Devices(devices=pb_devices)

    def Predict(self, request: inference_pb2.PredictRequest, context) -> inference_pb2.PredictResponse:
        session = self._getModelSession(context, request.modelSessionId)
        return inference_pb2.PredictResponse()

    def _getModelSession(self, context, modelSessionId: str) -> ISession:
        if not modelSessionId:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "model-session-id has not been provided by client")

        session = self.__session_manager.get(modelSessionId)

        if session is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"model-session with id {modelSessionId} doesn't exist")

        return session


def serve(host, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    session_svc = InferenceServicer(TorchDevicePool(), SessionManager())
    inference_pb2_grpc.add_InferenceServicer_to_server(session_svc, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
