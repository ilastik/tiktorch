import logging
import time
import torch

from torch.multiprocessing import Queue, Process, TimeoutError
from queue import Empty, Full

from typing import Any, List, Generic, Iterator, Iterable, TypeVar, Mapping, Callable, Dict, Optional, Tuple

from ..types import NDArrayBatch
from .training import TrainingProcess
from .inference import InferenceProcess

logger = logging.getLogger(__name__)


MAX_QUEUE_SIZE = 1000
SHUTDOWN = "shutdown"
SHUTDOWN_ANSWER = "shutting_down"


class ProcessComm:
    """
    Process Communication maintains two queues for bidirectional communication
    """

    def __init__(
        self, send_queue: Queue, recv_queue: Queue, send_timeout: float = 1, recv_timeout: float = 0, name: str = ""
    ) -> None:
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.send_timeout = send_timeout
        self.recv_timeout = recv_timeout
        if name is None:
            name = self.__name__

        self.logger = logging.getLogger(name)

    def send(self, call: str, kwargs: dict = {}, timeout: float = None) -> None:
        """
        :param call: method name of receiver
        :param kwargs: key word arguments for call
        :param timeout: block at most this long to send
        :raises: queue.Full
        """
        if timeout is None:
            timeout = self.send_timeout

        try:
            self.send_queue.put(call, block=False)
        except Full:
            self.logger.error("Send queue full")
            self.send_queue.put(call, block=True, timeout=timeout)

    def recv(self, timeout: float = None) -> Tuple[str, dict]:
        """
        :param timeout: block at most this long to receive
        :raises: queue.Empty
        """
        if timeout is None:
            timeout = self.recv_timeout

        self.recv_queue.get(block=bool(timeout), timeout=timeout)


class HandlerProcess(Process):
    """
    Process to orchestrate the interplay of training/validation and inference
    """

    def __init__(self, server_comm: ProcessComm):
        """
        :param from_server_queue: downstream communication
        :param to_server_queue: upstream communication
        :param timeout: log a warning if no message has been received for this many seconds
        """
        super().__init__(name="HandlerProcess")
        self.server_comm = server_comm

    def run(self):
        # set up training process
        training_comm = ProcessComm(Queue(MAX_QUEUE_SIZE), Queue(MAX_QUEUE_SIZE))
        training_proc = TrainingProcess(ProcessComm(training_comm.recv_queue, training_comm.send_queue))
        self.training_comm = training_comm
        self.training_proc = training_proc
        # set up inference process
        inference_comm = ProcessComm(Queue(MAX_QUEUE_SIZE), Queue(MAX_QUEUE_SIZE))
        inference_proc = InferenceProcess(ProcessComm(inference_comm.recv_queue, inference_comm.send_queue))
        self.inference_comm = inference_comm
        self.inference_proc = inference_proc

        try:
            while True:
                try:
                    call, kwargs = self.server_comm.recv()  # server communication has priority
                except Empty:
                    try:
                        call, kwargs = inference_comm.recv()  # inference has priority over training
                    except Empty:
                        try:
                            call, kwargs = training_comm.recv(timeout=0.01)
                        except:
                            continue

                meth = getattr(self, call, None)
                if meth is None:
                    raise NotImplementedError(call)

                meth(**kwargs)
        finally:
            self.shutdown()

    # general
    def shutdown(self):
        # initiate shutdown in parallel
        resend_shutdown_to_inference = True
        try:
            self.inference_comm.send(SHUTDOWN, timeout=0)
            resend_shutdown_to_inference = False
        except Exception:
            pass

        resend_shutdown_to_training = True
        try:
            self.training_comm.send(SHUTDOWN, timeout=0)
            resend_shutdown_to_training = False
        except Exception:
            pass

        def shutdown_child(comm, proc, resend_shutdown):
            while proc.is_alive():
                if resend_shutdown:
                    try:
                        comm.send(SHUTDOWN, timeout=0)
                        resend_shutdown = False
                    except Exception:
                        pass

                # look for shutting down answer
                while True:
                    try:
                        call, kwargs = comm.recv(timeout=2)
                        if call == "shutting_down":
                            proc.join(timeout=5)
                            if proc.exitcode is None:
                                # proc.join timed out, process is still running
                                raise TimeoutError

                            break
                    except Exception:
                        # could not shutdown gracefully
                        logger.error("Failed to shutdown %s gracefully", proc.name)
                        proc.kill()
                        proc.join()
                        break

        shutdown_child(self.inference_comm, self.inference_proc, resend_shutdown_to_inference)
        shutdown_child(self.training_comm, self.training_proc, resend_shutdown_to_training)

    def update_hparams(self, hparams: dict):
        pass

    # inference
    def forward(self, data: NDArrayBatch):
        raise NotImplementedError()

    # training
    def resume_train(self):
        pass

    def pause_train(self):
        pass

    def update_training_dataset(self):
        pass

    def request_state(self):
        # model state
        # optimizer state
        # current config dict
        pass

    # validation
    def update_validation_dataset(self):
        pass

    def validate(self):
        pass
