import time
from concurrent import futures
import multiprocessing as mp
import threading

import grpc

from tiktorch.rpc.mp import MPClient, MPServer, Shutdown, create_client
from tiktorch.proto import inference_pb2, inference_pb2_grpc, converters
from tiktorch.server.device_pool import IDevicePool, TorchDevicePool, DeviceStatus
from tiktorch.server.session_manager import SessionManager, ISession
from tiktorch.server.training import start_model_process


class InferenceServicer(inference_pb2_grpc.InferenceServicer, inference_pb2_grpc.FlightControlServicer):
    def __init__(self, device_pool: IDevicePool, session_manager: SessionManager, *, done_evt=None) -> None:
        self.__device_pool = device_pool
        self.__session_manager = session_manager
        self.__done_evt = done_evt

    def CreateModelSession(
        self, request: inference_pb2.CreateModelSessionRequest, context
    ) -> inference_pb2.ModelSession:
        lease = self.__device_pool.lease(request.deviceIds)
        _, client = start_model_process(model_zip=request.model_blob.content, devices=[d.id for d in lease.devices])
        session = self.__session_manager.create_session()
        session.on_close(lease.terminate)
        session.on_close(client.shutdown)
        session.client = client
        model_info = session.client.get_model_info()

        pb_valid_shapes = []
        for shape in model_info.valid_shapes:
            pb_shape = []
            for tag, size in shape:
                pb_shape.append(inference_pb2.TensorDim(size=size, name=tag))

            pb_valid_shapes.append(inference_pb2.Shape(dims=pb_shape))

        return inference_pb2.ModelSession(
            id=session.id,
            name=model_info.name,
            inputAxes=model_info.input_axes,
            outputAxes=model_info.output_axes,
            validShapes=pb_valid_shapes,
            hasTraining=False,
            halo=[inference_pb2.TensorDim(size=size, name=tag) for tag, size in model_info.halo],
        )

    def CreateDatasetDescription(
        self, request: inference_pb2.CreateDatasetDescriptionRequest, context
    ) -> inference_pb2.DatasetDescription:
        session = self._getModelSession(context, request.modelSessionId)
        id = session.client.create_dataset_description(mean=request.mean, stddev=request.stddev)
        return inference_pb2.DatasetDescription(id=id)

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
        arr = converters.pb_tensor_to_numpy(request.tensor)
        res = session.client.forward(arr)
        pb_tensor = converters.numpy_to_pb_tensor(res)
        return inference_pb2.PredictResponse(tensor=pb_tensor)

    def _getModelSession(self, context, modelSessionId: str) -> ISession:
        if not modelSessionId:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "model-session-id has not been provided by client")

        session = self.__session_manager.get(modelSessionId)

        if session is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"model-session with id {modelSessionId} doesn't exist")

        return session

    def Ping(self, request: inference_pb2.Empty, context):
        return inference_pb2.Empty()

    def Shutdown(self, request: inference_pb2.Empty, context):
        if self.__done_evt:
            self.__done_evt.set()
        return inference_pb2.Empty()


def serve(host, port):
    done_evt = threading.Event()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    session_svc = InferenceServicer(TorchDevicePool(), SessionManager(), done_evt=done_evt)
    inference_pb2_grpc.add_InferenceServicer_to_server(session_svc, server)
    inference_pb2_grpc.add_FlightControlServicer_to_server(session_svc, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    done_evt.wait()

    server.stop(0).wait()
