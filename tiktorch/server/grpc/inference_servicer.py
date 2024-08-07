import gc
import time
from typing import List, Optional, Tuple

import grpc
import numpy as np
import torch.cuda
import xarray

from tiktorch import converters
from tiktorch.converters import info2session
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

        return info2session(session.id, model_info)

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

    def MaxCudaMemoryShape(
        self, request: inference_pb2.MaxCudaMemoryShapeRequest, context
    ) -> inference_pb2.MaxCudaMemoryShapeResponse:
        session = self._getModelSession(context, request.modelSessionId)
        max_shape = self._get_max_shape(session, request)
        if max_shape is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "no valid shape")
        return inference_pb2.MaxCudaMemoryShapeResponse(maxShape=max_shape)

    def IsCudaOutOfMemory(
        self, request: inference_pb2.IsCudaOutOfMemoryRequest, context
    ) -> inference_pb2.IsCudaOutOfMemoryResponse:
        session = self._getModelSession(context, request.modelSessionId)
        return inference_pb2.IsCudaOutOfMemoryResponse(
            isCudaOutOfMemory=self._is_cuda_out_of_memory(session, request.shapeReference)
        )

    def _get_max_shape(
        self, session: ISession, request: inference_pb2.MaxCudaMemoryShapeRequest
    ) -> Optional[List[int]]:
        max_shape_reference = np.array(request.maxShapeReference)
        min_shape_reference = np.array(request.minShapeReference)
        step = np.array(request.step)
        diff_reference = max_shape_reference - min_shape_reference
        num_increments = list(set(diff_reference / step))

        assert len(num_increments) == 1 and self._is_natural_num(num_increments[0])
        num_increments = int(num_increments[0])
        assert np.array_equal(min_shape_reference + num_increments * step, max_shape_reference)

        max_shape = max_shape_reference
        for increment in range(num_increments):
            max_shape = max_shape - increment * step
            if not self._is_cuda_out_of_memory(session, max_shape):
                return max_shape
        return None

    def _is_natural_num(self, num) -> bool:
        return np.floor(num) == np.ceil(num) and num >= 0

    def _is_cuda_out_of_memory(self, session: ISession, shape: Tuple[int, ...]) -> bool:
        if not self._is_gpu():
            return False
        is_out_of_memory = False
        dummy_tensor = xarray.DataArray(np.random.rand(*shape))
        try:
            session.client.forward([dummy_tensor])
        except RuntimeError as e:
            if "out of memory" in str(e):
                is_out_of_memory = True
                print(f"Using shape {shape} will cause out of memory.")
            else:
                raise
        finally:
            gc.collect()  # attempt to explicitly deallocate memory
            torch.cuda.empty_cache()
        return is_out_of_memory

    def _is_gpu(self) -> bool:
        return torch.cuda.is_available()

    def _getModelSession(self, context, modelSessionId: str) -> ISession:
        if not modelSessionId:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "model-session-id has not been provided by client")

        session = self.__session_manager.get(modelSessionId)

        if session is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"model-session with id {modelSessionId} doesn't exist")

        return session
