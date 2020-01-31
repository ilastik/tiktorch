import time
from concurrent import futures
import multiprocessing as mp

import grpc

from tiktorch.rpc.mp import MPClient, MPServer, Shutdown, create_client
from tiktorch.proto import inference_pb2, inference_pb2_grpc
from tiktorch.server.device_manager import IDeviceManager, TorchDeviceManager, DeviceStatus
from tiktorch.server.model_manager import ModelManager, IModel
from tiktorch.server.handler import inference_new as inference


_ONE_DAY_IN_SECONDS = 24 * 60 * 60


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, device_manager: IDeviceManager, model_manager: ModelManager) -> None:
        self.__device_manager = device_manager
        self.__model_manager = model_manager

    def CreateModel(self, request: inference_pb2.CreateModelRequest, context) -> inference_pb2.Model:
        lease = self.__device_manager.lease(request.deviceIds)
        model = self.__model_manager.create_model()
        model.on_close(lease.terminate)
        # model.on_close(lease.terminate)
        # handler2inference_conn, inference2handler_conn = mp.Pipe()
        # self._inference_proc = mp.Process(
        #     target=inference.run, name="Inference", kwargs={"conn": inference2handler_conn}
        # )
        # self._inference_proc.start()
        # self._inference = create_client(inference.IInference, handler2inference_conn)
        # self._inference.load_model(request.model_blob).result()
        return inference_pb2.Model(id=model.id)

    def CloseModel(self, request: inference_pb2.Model, context) -> inference_pb2.Empty:
        self.__model_manager.close_model(request.id)
        return inference_pb2.Empty()

    def GetLogs(self, request: inference_pb2.Empty, context):
        yield inference_pb2.LogEntry(
            timestamp=int(time.time()), level=inference_pb2.LogEntry.Level.INFO, content="Sending model logs"
        )

    def ListDevices(self, request: inference_pb2.Empty, context) -> inference_pb2.Devices:
        devices = self.__device_manager.list_devices()
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
        model = self._getModel(context, request.modelId)
        return inference_pb2.PredictResponse()

    def _getModel(self, context, modelId: str) -> IModel:
        if not modelId:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "model-id has not been provided by client")

        model = self.__model_manager.get(modelId)

        if model is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"model with id {modelId} doesn't exist")

        return model


def serve(host, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    session_svc = InferenceServicer(TorchDeviceManager(), ModelManager())
    inference_pb2_grpc.add_InferenceServicer_to_server(session_svc, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
