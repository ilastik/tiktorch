import logging
import threading
from typing import Generic, Set, TypeVar, Union

from bioimageio.core import PredictionPipeline, Sample

from tiktorch.server.session.backend import commands
from tiktorch.server.session.backend.commands import CommandPriorityQueueUtils, ShutdownWithTeardownCmd
from tiktorch.trainer import Callbacks, ErrorCallbacks, Trainer, TrainerState

logger = logging.getLogger(__name__)


class StateTransitionError(Exception):
    def __init__(self, current_state: TrainerState, transitioning_state: TrainerState, valid_states: Set[TrainerState]):
        super().__init__(
            f"Invalid state transition: {current_state} -> {transitioning_state}. Valids are {valid_states}"
        )
        self.current_state = current_state
        self.transitioning_state = transitioning_state
        self.valid_states = valid_states

    def __reduce__(self):
        return (
            self.__class__,
            (self.current_state, self.transitioning_state, self.valid_states),
        )


class TrainerSupervisor:
    """Training supervisor for custom models supported by the 'Trainer' interface.

    Monitoring the training thread and its status.
    """

    def __init__(self, trainer: Trainer) -> None:
        super().__init__()
        self._trainer = trainer
        self._trainer.should_stop_callbacks.register(self._should_stop)
        self._shutdown_event = threading.Event()
        self._state = TrainerState.IDLE
        self._pause_triggered = False
        self._session_thread = threading.Thread(target=self._start_session, name="SessionThread")
        self._command_queue_utils = CommandPriorityQueueUtils()
        self.training_error_callbacks: ErrorCallbacks = Callbacks()

    def get_state(self) -> TrainerState:
        logger.debug(f"Get state called {self._state}")
        return self._state

    def start(self):
        if self._state != TrainerState.IDLE:
            raise StateTransitionError(
                current_state=self._state, transitioning_state=TrainerState.RUNNING, valid_states={TrainerState.IDLE}
            )
        self._session_thread.start()
        self._pause_triggered = False
        start_cmd = commands.SetStartStateTrainingCmd()
        self._command_queue_utils.send_command(start_cmd.awaitable)
        start_cmd.awaitable.wait()

    def _start_session(self):
        logger.info("Starting session worker")
        try:
            while True:
                if self._command_queue_utils.process_commands(self):
                    break

                if self._state == TrainerState.RUNNING:
                    self._fit()
        except Exception as e:
            logger.exception(f"Uncaught exception in session worker. Exception: {e}")
        finally:
            logger.info("Stopped session worker")

    def _fit(self):
        try:
            self._trainer.fit()
        except Exception as e:
            logger.exception(f"Training error: {e}")
            self.training_error_callbacks(e)
            self._command_queue_utils.send_command(commands.ShutdownWithErrorCmd())
            return

        if self.is_training_finished():
            logger.info(f"Training has finished: {self._get_num_iterations_epochs()} ")
            self._command_queue_utils.send_command(commands.NominalShutdownCmd())

    def is_training_finished(self):
        return (
            self._trainer.num_epochs == self._trainer.max_num_epochs
            or self._trainer.num_iterations == self._trainer.max_num_iterations
        )

    def _get_num_iterations_epochs(self) -> str:
        iterations = f"Iterations[{self._trainer.num_iterations}/{self._trainer.max_num_iterations}]"
        epochs = f"Epochs[{self._trainer.num_epochs}/{self._trainer.max_num_epochs}]"
        return f"{iterations}, {epochs}"

    def resume(self):
        self._check_transition_to_state(TrainerState.RUNNING, valid_states={TrainerState.PAUSED})
        self._pause_triggered = False
        resume_cmd = commands.SetResumeStateTrainingCmd()
        self._command_queue_utils.send_command(resume_cmd.awaitable)
        resume_cmd.awaitable.wait()  # make sure that the state has actually changed (acknowledge)
        logger.info(f"Resume training: {self._get_num_iterations_epochs()}")

    def pause(self):
        self._check_transition_to_state(TrainerState.PAUSED, valid_states={TrainerState.RUNNING})
        self._pause_triggered = True
        pause_cmd = commands.SetPauseStateTrainingCmd()
        self._command_queue_utils.send_command(pause_cmd.awaitable)
        pause_cmd.awaitable.wait()  # make sure that the state has actually changed (acknowledge)

    def shutdown(self):
        if not self._session_thread.is_alive():
            # nothing to do if session thread not alive
            return
        self._pause_triggered = True
        self._command_queue_utils.send_command(commands.ShutdownCmd())
        self._session_thread.join()

    def forward(self, input_tensors):
        self.pause()
        self._trainer.forward(input_tensors)
        self.resume()

    def save(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def _should_stop(self):
        return self._pause_triggered

    def transition_to_state(self, new_state: TrainerState, valid_states: Set[TrainerState]):
        """
        Should be used via the ICommands to monitor the state of the training
        """
        self._check_transition_to_state(new_state, valid_states)
        logger.info(f"State transition: {self._state} -> {new_state}")
        self._state = new_state

    def _check_transition_to_state(self, new_state: TrainerState, valid_states: Set[TrainerState]):
        if self._state not in valid_states:
            raise StateTransitionError(
                current_state=self._state, transitioning_state=new_state, valid_states=valid_states
            )


class BioModelSupervisor:
    """Supervisor for bioimageio models

    Currently used only for inference.

    Allows to serialize and offload commands by multiple threads requests.
    """

    def __init__(self, pipeline: PredictionPipeline) -> None:
        super().__init__()
        self._pipeline = pipeline

    def forward(self, sample: Sample):
        results = self._pipeline.predict_sample_without_blocking(sample)
        return results

    def set_max_num_iterations(self, num_iterations: int):
        raise NotImplementedError

    def update_dataset(self):
        raise NotImplementedError

    def shutdown(self):
        pass


Supervisors = Union[BioModelSupervisor, TrainerSupervisor]
SupervisorTypeVar = TypeVar("SupervisorTypeVar", bound=Supervisors)


class QueueTasks(Generic[SupervisorTypeVar]):
    """
    A task queue manager for processing commands with a supervisor.

    Serializes multiple async requests wrapped as commands.
    """

    def __init__(self, supervisor: SupervisorTypeVar) -> None:
        self._command_queue = CommandPriorityQueueUtils()
        self._supervisor = supervisor
        self._thread = threading.Thread(target=self._run, name="QueueTasksWorker")

    def start(self):
        self._thread.start()

    def _run(self):
        logger.info("Starting session worker")
        try:
            while True:
                if self._command_queue.process_commands(self._supervisor):
                    break
        except Exception as e:
            logger.exception(f"Uncaught exception in session worker {e}")
        finally:
            logger.info("Stopped session worker")

    def send_command(self, command: commands.ICommand):
        self._command_queue.send_command(command)

    def shutdown(self):
        if not self._thread.is_alive():
            logger.debug("Worker thread isn't alive")
            return
        logger.debug("Shutting down...")
        stop_cmd = ShutdownWithTeardownCmd()
        self.send_command(stop_cmd.awaitable)
        stop_cmd.awaitable.wait()
        logger.debug("Shutdown complete")
        self._thread.join()
