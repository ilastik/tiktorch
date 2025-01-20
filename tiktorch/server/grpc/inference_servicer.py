import time

from tiktorch.converters import pb_tensors_to_sample, sample_to_pb_tensors
from tiktorch.proto import inference_pb2, inference_pb2_grpc, utils_pb2
from tiktorch.rpc.mp import BioModelClient
from tiktorch.server.data_store import IDataStore
from tiktorch.server.device_pool import IDevicePool
from tiktorch.server.grpc.utils_servicer import get_model_session, list_devices
from tiktorch.server.session.process import InputSampleValidator, start_model_session_process
from tiktorch.server.session_manager import Session, SessionManager


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(
        self, device_pool: IDevicePool, session_manager: SessionManager[BioModelClient], data_store: IDataStore
    ) -> None:
        self.__device_pool = device_pool
        self.__session_manager = session_manager
        self.__data_store = data_store

    def CreateModelSession(self, request: inference_pb2.CreateModelSessionRequest, context) -> utils_pb2.ModelSession:
        if request.HasField("model_uri"):
            if not request.model_uri.startswith("upload://"):
                raise NotImplementedError("Only upload:// URI supported")

            upload_id = request.model_uri.replace("upload://", "")
            content = self.__data_store.get(upload_id)
        else:
            content = request.model_blob.content

        devices = list(request.deviceIds)

        _, client = start_model_session_process(model_bytes=content)
        session = self.__session_manager.create_session(client)
        session.on_close(client.api.shutdown)

        lease = self.__device_pool.lease(devices)
        session.on_close(lease.terminate)

        try:
            client.api.init(model_bytes=content, devices=devices)
        except Exception as e:
            self.__session_manager.close_session(session.id)
            raise e

        return utils_pb2.ModelSession(id=session.id)

    def CreateDatasetDescription(
        self, request: inference_pb2.CreateDatasetDescriptionRequest, context
    ) -> inference_pb2.DatasetDescription:
        session = self._getModelSession(context, request.modelSessionId)
        id = session.client.api.create_dataset_description(mean=request.mean, stddev=request.stddev)
        return inference_pb2.DatasetDescription(id=id)

    def CloseModelSession(self, request: utils_pb2.ModelSession, context) -> utils_pb2.Empty:
        self.__session_manager.close_session(request.id)
        return utils_pb2.Empty()

    def close_all_sessions(self):
        """
        Not exposed by the API

        Close all sessions ensuring that all devices are not leased
        """
        self.__session_manager.close_all_sessions()
        assert len(self.__device_pool.list_reserved_devices()) == 0

    def GetLogs(self, request: utils_pb2.Empty, context):
        yield inference_pb2.LogEntry(
            timestamp=int(time.time()), level=inference_pb2.LogEntry.Level.INFO, content="Sending model logs"
        )

    def ListDevices(self, request: utils_pb2.Empty, context) -> utils_pb2.Devices:
        return list_devices(self.__device_pool)

    def Predict(self, request: utils_pb2.PredictRequest, context) -> utils_pb2.PredictResponse:
        session = self._getModelSession(context, request.modelSessionId)
        input_sample = pb_tensors_to_sample(request.tensors)
        tensor_validator = InputSampleValidator(session.client.input_specs)
        tensor_validator.check_tensors(input_sample)
        res = session.client.api.forward(input_sample).result()
        return utils_pb2.PredictResponse(tensors=sample_to_pb_tensors(res))

    def _getModelSession(self, context, modelSessionId: utils_pb2.ModelSession) -> Session[BioModelClient]:
        return get_model_session(
            session_manager=self.__session_manager, model_session_id=modelSessionId, context=context
        )
