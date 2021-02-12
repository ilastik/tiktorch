from __future__ import annotations

import logging
import queue

import numpy as np
import xarray as xr

from tiktorch.server.prediction_pipeline import PredictionPipeline
from tiktorch.server.session import types
from tiktorch.server.session.backend import commands

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(self, pipeline: PredictionPipeline) -> None:
        self._state = types.State.Stopped

        self._command_queue = commands.CommandPriorityQueue()
        self._pipeline = pipeline
        self._pipeline.set_break_callback(self.has_commands)
        self._idle_callbacks = []

    def send_command(self, cmd: commands.ICommand) -> None:
        if not isinstance(cmd, commands.ICommand):
            raise ValueError(f"Expected instance of ICommand got {cmd}")

        logger.debug("Sending command %s", cmd)
        self._command_queue.put(cmd)

    @property
    def state(self):
        return self._state

    def has_commands(self):
        return not self._command_queue.empty()

    def has_work(self):
        return self._pipeline.max_num_iterations and self._pipeline.max_num_iterations > self._pipeline.iteration_count

    def forward(self, input_tensor):
        result = self._pipeline.forward(input_tensor)
        assert isinstance(result, xr.DataArray), f"Not a DataArray, but a {type(result)}"
        return result

    def transition_to(self, new_state: types.State) -> None:
        logger.debug("Attempting transition to state %s", new_state)
        self._state = new_state
        self._update_state()

    def set_max_num_iterations(self, num: int):
        self._pipeline.set_max_num_iterations(num)
        self._update_state()

    def on_idle(self, callback):
        self._idle_callbacks.append(callback)
        self._notify_idle()

    def _notify_idle(self):
        if self._state in (types.State.Idle, types.State.Paused):
            idle_cbs = self._idle_callbacks
            self._idle_callbacks = []
            for cb in idle_cbs:
                try:
                    cb()
                except Exception:
                    logger.exception("Exception during idle callback")

    def run(self):
        logger.info("Starting session worker")
        try:
            self._run()
        except Exception:
            logger.exception("Uncaught exception in session worker")
        finally:
            logger.info("Stopped session worker")

    def _run(self):
        self._set_state(types.State.Paused)

        while True:
            self._process_commands()

            if self.state == types.State.Stopped:
                break

            elif self._state == types.State.Idle or self._state == types.State.Paused:
                with self._command_queue.not_empty:
                    self._command_queue.not_empty.wait()

            elif self._state == types.State.Running:
                self._train()
                self._update_state()

    def _process_commands(self):
        while not self._command_queue.empty():
            try:
                cmd = self._command_queue.get_nowait()
                logger.debug("Executing %s", cmd)
                ctx = commands.Context(supervisor=self)

                try:
                    cmd.execute(ctx)
                except Exception:
                    logger.exception("Failed to execute %s", cmd)
                finally:
                    self._command_queue.task_done()

            except queue.Empty:
                pass

    def _train(self):
        logger.info(
            "Start session for %d iterations", self._pipeline.max_num_iterations - self._pipeline.iteration_count
        )
        try:
            self._pipeline.train()
        except Exception as e:
            logger.error("Exception during session training. Pausing...", exc_info=True)
            # FIXME: Should we use PauseCmd here? Maybe we should only know about ICommand on this level.
            self.send_command(commands.PauseCmd())

        self._update_state()

    def _update_state(self):
        if self._state == types.State.Running:
            should_idle = not self.has_work()
            if should_idle:
                self._set_state(types.State.Idle)

        elif self._state == types.State.Idle:
            should_run = self.has_work()
            if should_run:
                self._set_state(types.State.Running)

    def _set_state(self, new_state: types.State) -> None:
        self._state = new_state
        self._notify_idle()
        logger.debug("Set new state %s", self._state)
