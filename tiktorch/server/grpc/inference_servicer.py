import gc
import time
from typing import Optional

import grpc
import numpy as np
import torch.cuda
import xarray

from tiktorch.converters import (
    InputTensorValidator,
    NamedParametrizedShape,
    NamedShape,
    Sample,
    get_axes_with_size,
    named_shape_to_pb_NamedInts,
    pb_NamedInts_to_named_shape,
)
from tiktorch.proto import inference_pb2, inference_pb2_grpc
from tiktorch.rpc.mp import BioModelClient
from tiktorch.server.data_store import IDataStore
from tiktorch.server.device_pool import DeviceStatus, IDevicePool
from tiktorch.server.session.process import start_model_session_process
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
            _, client = start_model_session_process(model_zip=content, devices=[d.id for d in lease.devices])
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
        input_sample = Sample.from_pb_tensors(request.tensors)
        res = self._validated_forward(session.bio_model_client, input_sample)
        output_tensor_ids = [tensor.name for tensor in session.bio_model_client.output_specs]
        output_sample = Sample.from_xr_tensors(output_tensor_ids, res)
        return inference_pb2.PredictResponse(tensors=output_sample.to_pb_tensors())

    def MaxCudaMemoryShape(
        self, request: inference_pb2.MaxCudaMemoryShapeRequest, context
    ) -> inference_pb2.MaxCudaMemoryShapeResponse:
        session = self._getModelSession(context, request.modelSessionId)
        self._check_gpu_exists(session.bio_model_client, request.deviceId)
        min_shape = pb_NamedInts_to_named_shape(request.minShape)
        step_shape = pb_NamedInts_to_named_shape(request.stepShape)
        max_shape = pb_NamedInts_to_named_shape(request.maxShape)
        max_valid_shape = self._get_max_shape(
            tensor_id=request.tensorId,
            client=session.bio_model_client,
            param_shape=NamedParametrizedShape(min_shape, step_shape),
            max_shape=max_shape,
        )
        if max_valid_shape is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "no valid shape")
        return inference_pb2.MaxCudaMemoryShapeResponse(maxShape=named_shape_to_pb_NamedInts(max_valid_shape))

    def IsCudaOutOfMemory(
        self, request: inference_pb2.IsCudaOutOfMemoryRequest, context
    ) -> inference_pb2.IsCudaOutOfMemoryResponse:
        session = self._getModelSession(context, request.modelSessionId)
        self._check_gpu_exists(session.bio_model_client, request.deviceId)
        return inference_pb2.IsCudaOutOfMemoryResponse(
            isCudaOutOfMemory=self._is_cuda_out_of_memory(
                session.bio_model_client, request.tensorId, pb_NamedInts_to_named_shape(request.shape)
            )
        )

    def _get_max_shape(
        self,
        client: BioModelClient,
        tensor_id: str,
        param_shape: NamedParametrizedShape,
        max_shape: NamedShape,
    ) -> Optional[NamedShape]:
        num_increment = InputTensorValidator.get_num_increments_from_param_shape(param_shape, max_shape)
        if num_increment is None:
            raise ValueError(
                f"Invalid parameterized shape min: {param_shape}, with max: {max_shape}\n"
                f"max != min + n * step, where n belongs to (0, 1, 2, ...)"
            )

        max_shape_arr = np.array(list(max_shape.values()))
        step_shape_arr = np.array(list(param_shape.step_shape.values()))
        for increment in range(num_increment):
            max_shape_arr = max_shape_arr - increment * step_shape_arr
            max_shape = get_axes_with_size(param_shape.axes, max_shape_arr)
            if not self._is_cuda_out_of_memory(client, tensor_id, max_shape):
                return max_shape
        return None

    def _is_cuda_out_of_memory(self, client: BioModelClient, tensor_id: str, shape: NamedShape) -> bool:
        is_out_of_memory = False
        dummy_tensor = xarray.DataArray(np.random.rand(*shape.values()), dims=shape.keys())
        sample = Sample.from_xr_tensors(tensor_ids=[tensor_id], tensors_data=[dummy_tensor])
        try:
            self._validated_forward(client, sample)
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

    def _validated_forward(self, client: BioModelClient, sample: Sample):
        validator = InputTensorValidator(client.input_specs)
        validator.check_tensors(sample)
        return client.api.forward(sample)

    def _check_gpu_exists(self, client: BioModelClient, device_id: str):
        gpu_device_ids = [device.id for device in self.__device_pool.list_devices() if device.id.startswith("cuda")]
        if len(gpu_device_ids) == 0:
            raise ValueError("Not available gpus found")
        if device_id not in client.devices:
            raise ValueError(f"{device_id} not found for model {client.name}")

    def _getModelSession(self, context, modelSessionId: str) -> Session:
        if not modelSessionId:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "model-session-id has not been provided by client")

        session = self.__session_manager.get(modelSessionId)

        if session is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"model-session with id {modelSessionId} doesn't exist")

        return session
