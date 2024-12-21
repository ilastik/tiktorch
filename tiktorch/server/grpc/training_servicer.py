from __future__ import annotations

import logging
import queue
import time
from pathlib import Path
from typing import Callable, List

from tiktorch.converters import pb_tensors_to_sample, sample_to_pb_tensors, trainer_state_to_pb
from tiktorch.proto import training_pb2, training_pb2_grpc, utils_pb2
from tiktorch.server.device_pool import IDevicePool
from tiktorch.server.grpc.utils_servicer import get_model_session, list_devices
from tiktorch.server.session.process import start_trainer_process
from tiktorch.server.session.rpc_interface import IRPCTrainer
from tiktorch.server.session_manager import Session, SessionManager
from tiktorch.trainer import TrainerYamlParser

logger = logging.getLogger(__name__)


class TrainingServicer(training_pb2_grpc.TrainingServicer):
    def __init__(
        self,
        device_pool: IDevicePool,
        session_manager: SessionManager[IRPCTrainer],
    ) -> None:
        self._device_pool = device_pool
        self._logs_queue_stream = queue.Queue()
        self._should_stop_callbacks: List[Callable] = []
        self._session_manager = session_manager

    def ListDevices(self, request: utils_pb2.Empty, context) -> utils_pb2.Devices:
        return list_devices(self._device_pool)

    def Init(self, request: training_pb2.TrainingConfig, context):
        parser = TrainerYamlParser(request.yaml_content)
        device = parser.get_device()

        _, client = start_trainer_process()
        session = self._session_manager.create_session(client)
        session.on_close(client.shutdown)

        lease = self._device_pool.lease([device])
        session.on_close(lease.terminate)

        try:
            client.init(request.yaml_content)
        except Exception as e:
            self._session_manager.close_session(session.id)
            raise e

        return utils_pb2.ModelSession(id=session.id)

    def Start(self, request: utils_pb2.ModelSession, context):
        session = self._getTrainerSession(context, request)
        session.client.start_training()
        return utils_pb2.Empty()

    def Resume(self, request, context):
        session = self._getTrainerSession(context, request)
        session.client.resume_training()
        return utils_pb2.Empty()

    def Pause(self, request: utils_pb2.ModelSession, context):
        session = self._getTrainerSession(context, request)
        session.client.pause_training()
        return utils_pb2.Empty()

    def Save(self, request: training_pb2.SaveRequest, context):
        session = self._getTrainerSession(context, request.modelSessionId)
        session.client.save(Path(request.filePath))
        return utils_pb2.Empty()

    def Export(self, request: training_pb2.ExportRequest, context):
        session = self._getTrainerSession(context, request.modelSessionId)
        session.client.export(Path(request.filePath))
        return utils_pb2.Empty()

    def Predict(self, request: utils_pb2.PredictRequest, context):
        session = self._getTrainerSession(context, request.modelSessionId)
        input_sample = pb_tensors_to_sample(request.tensors)
        predictions = session.client.forward(input_sample).result()
        return utils_pb2.PredictResponse(tensors=sample_to_pb_tensors(predictions))

    def StreamUpdates(self, request: utils_pb2.ModelSession, context):
        raise NotImplementedError

    def GetLogs(self, request: utils_pb2.ModelSession, context):
        raise NotImplementedError

    def IsBestModel(self, request, context):
        session = self._getTrainerSession(context, request)
        prev_best_model_idx = None
        while context.is_active():
            current_best_model_idx = session.client.get_best_model_idx()
            if current_best_model_idx != prev_best_model_idx:
                prev_best_model_idx = current_best_model_idx
                yield utils_pb2.Empty()
            time.sleep(1)
        logger.info("Client disconnected. Stopping stream.")

    def GetStatus(self, request: utils_pb2.ModelSession, context):
        session = self._getTrainerSession(context, request)
        state = session.client.get_state()
        return training_pb2.GetStatusResponse(state=trainer_state_to_pb[state])

    def CloseTrainerSession(self, request: utils_pb2.ModelSession, context) -> utils_pb2.Empty:
        self._session_manager.close_session(request.id)
        return utils_pb2.Empty()

    def close_all_sessions(self):
        self._session_manager.close_all_sessions()

    def _getTrainerSession(self, context, model_session_id: utils_pb2.ModelSession) -> Session[IRPCTrainer]:
        return get_model_session(
            session_manager=self._session_manager, model_session_id=model_session_id, context=context
        )
