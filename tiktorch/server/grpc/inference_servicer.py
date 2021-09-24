import time

import grpc

from tiktorch import converters
from tiktorch.proto import inference_pb2, inference_pb2_grpc
from tiktorch.server.data_store import IDataStore
from tiktorch.server.device_pool import DeviceStatus, IDevicePool
from tiktorch.server.session.process import start_model_session_process
from tiktorch.server.session_manager import ISession, SessionManager


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, device_pool: IDevicePool, session_manager: SessionManager, data_store: IDataStore) -> None:
        self.__device_pool = device_pool
        self.__session_manager = session_manager
        self.__data_store = data_store

    def CreateModelSession(
        self, request: inference_pb2.CreateModelSessionRequest, context
    ) -> inference_pb2.ModelSession:
        if request.HasField("model_uri"):
            if not request.model_uri.startswith("upload://"):
                raise NotImplementedError("Only upload:// URI supported")

            upload_id = request.model_uri.replace("upload://", "")
            content = self.__data_store.get(upload_id)
        else:
            content = request.model_blob.content

        lease = self.__device_pool.lease(request.deviceIds)

        try:
            _, client = start_model_session_process(model_zip=content, devices=[d.id for d in lease.devices])
        except Exception:
            lease.terminate()
            raise

        session = self.__session_manager.create_session()
        session.on_close(lease.terminate)
        session.on_close(client.shutdown)
        session.client = client

        try:
            model_info = session.client.get_model_info()
        except Exception:
            lease.terminate()
            raise

        pb_valid_shapes = []
        for shape in model_info.valid_shapes:
            pb_shape = []
            for tag, size in shape:
                pb_shape.append(inference_pb2.NamedInt(size=size, name=tag))

            pb_valid_shapes.append(inference_pb2.Shape(dims=pb_shape))

        pb_halos = [converters.name_int_tuples_to_pb_shape(halo) for halo in model_info.halos]
        pb_offsets = [converters.name_int_tuples_to_pb_shape(offset) for offset in model_info.offsets]
        pb_valid_scales = [converters.name_float_tuples_to_pb_scale(scale) for scale in model_info.scales]

        return inference_pb2.ModelSession(
            id=session.id,
            name=model_info.name,
            inputAxes=model_info.input_axes,
            outputAxes=model_info.output_axes,
            validShapes=pb_valid_shapes,
            hasTraining=False,
            halos=pb_halos,
            scales=pb_valid_scales,
            offsets=pb_offsets,
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
        arrs = [converters.pb_tensor_to_xarray(tensor) for tensor in request.tensors]
        res = session.client.forward(arrs)
        pb_tensors = [converters.xarray_to_pb_tensor(res_tensor) for res_tensor in res]
        return inference_pb2.PredictResponse(tensors=pb_tensors)

    def _getModelSession(self, context, modelSessionId: str) -> ISession:
        if not modelSessionId:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "model-session-id has not been provided by client")

        session = self.__session_manager.get(modelSessionId)

        if session is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"model-session with id {modelSessionId} doesn't exist")

        return session
