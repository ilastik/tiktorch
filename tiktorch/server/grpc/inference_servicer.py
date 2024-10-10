import time

import grpc

from tiktorch.converters import pb_tensors_to_sample, sample_to_pb_tensors
from tiktorch.proto import inference_pb2, inference_pb2_grpc
from tiktorch.server.data_store import IDataStore
from tiktorch.server.device_pool import DeviceStatus, IDevicePool
from tiktorch.server.session.process import InputSampleValidator, start_model_session_process
from tiktorch.server.session_manager import Session, SessionManager


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
            _, client = start_model_session_process(model_bytes=content, devices=[d.id for d in lease.devices])
        except Exception:
            lease.terminate()
            raise

        session = self.__session_manager.create_session(client)
        session.on_close(lease.terminate)
        session.on_close(client.api.shutdown)

        return inference_pb2.ModelSession(id=session.id)

    def CreateDatasetDescription(
        self, request: inference_pb2.CreateDatasetDescriptionRequest, context
    ) -> inference_pb2.DatasetDescription:
        session = self._getModelSession(context, request.modelSessionId)
        id = session.bio_model_client.api.create_dataset_description(mean=request.mean, stddev=request.stddev)
        return inference_pb2.DatasetDescription(id=id)

    def CloseModelSession(self, request: inference_pb2.ModelSession, context) -> inference_pb2.Empty:
        self.__session_manager.close_session(request.id)
        return inference_pb2.Empty()

    def close_all_sessions(self):
        """
        Not exposed by the API

        Close all sessions ensuring that all devices are not leased
        """
        self.__session_manager.close_all_sessions()
        assert len(self.__device_pool.list_reserved_devices()) == 0

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
        input_sample = pb_tensors_to_sample(request.tensors)
        tensor_validator = InputSampleValidator(session.bio_model_client.input_specs)
        tensor_validator.check_tensors(input_sample)
        res = session.bio_model_client.api.forward(input_sample)
        return inference_pb2.PredictResponse(tensors=sample_to_pb_tensors(res))

    def _getModelSession(self, context, modelSessionId: str) -> Session:
        if not modelSessionId:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "model-session-id has not been provided by client")

        session = self.__session_manager.get(modelSessionId)

        if session is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"model-session with id {modelSessionId} doesn't exist")

        return session
