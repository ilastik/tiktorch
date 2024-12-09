from __future__ import annotations

import logging
import queue
from typing import Callable, List

import grpc

from tiktorch.converters import trainer_state_to_pb
from tiktorch.proto import training_pb2, training_pb2_grpc
from tiktorch.server.device_pool import IDevicePool
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

    def Init(self, request: training_pb2.TrainingConfig, context):
        parser = TrainerYamlParser(request.yaml_content)
        device = parser.get_device()

        _, client = start_trainer_process()
        session = self._session_manager.create_session(client)
        session.on_close(client.shutdown)

        lease = self._device_pool.lease([device])
        session.on_close(lease.terminate)

        client.init(request.yaml_content)

        return training_pb2.TrainingSessionId(id=session.id)

    def Start(self, request, context):
        session = self._getTrainerSession(context, request.id)
        session.client.start_training()
        return training_pb2.Empty()

    def Resume(self, request, context):
        session = self._getTrainerSession(context, request.id)
        session.client.resume_training()
        return training_pb2.Empty()

    def Pause(self, request: training_pb2.TrainingSessionId, context):
        session = self._getTrainerSession(context, request.id)
        session.client.pause_training()
        return training_pb2.Empty()

    def Save(self, request: training_pb2.TrainingSessionId, context):
        session = self._getTrainerSession(context, request.modelSessionId)
        session.client.save()
        return training_pb2.Empty()

    def Export(self, request: training_pb2.TrainingSessionId, context):
        session = self._getTrainerSession(context, request.modelSessionId)
        session.client.export()
        return training_pb2.Empty()

    def Predict(self, request: training_pb2.TrainingSessionId, context):
        raise NotImplementedError

    def StreamUpdates(self, request: training_pb2.TrainingSessionId, context):
        raise NotImplementedError

    def GetLogs(self, request: training_pb2.TrainingSessionId, context):
        raise NotImplementedError

    def GetStatus(self, request: training_pb2.TrainingSessionId, context):
        session = self._getTrainerSession(context, request.id)
        state = session.client.get_state()
        return training_pb2.GetStatusResponse(state=trainer_state_to_pb[state])

    def CloseTrainerSession(self, request: training_pb2.TrainingSessionId, context) -> training_pb2.Empty:
        self._session_manager.close_session(request.id)
        return training_pb2.Empty()

    def close_all_sessions(self):
        self._session_manager.close_all_sessions()

    def _getTrainerSession(self, context, trainer_session_id: str) -> Session[IRPCTrainer]:
        session = self._session_manager.get(trainer_session_id)

        if session is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, f"trainer-session with id {trainer_session_id} doesn't exist"
            )

        return session
