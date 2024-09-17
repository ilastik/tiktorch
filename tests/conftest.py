import faulthandler
import logging.handlers
import multiprocessing as mp
import signal
import sys
import threading
from os import getenv
from random import randint
from typing import Generator, List, Tuple
from unittest.mock import create_autospec, patch

import numpy as np
import pytest
import xarray as xr
from bioimageio.core import AxisId, PredictionPipeline, Sample, Tensor
from bioimageio.spec.model import v0_5
from bioimageio.spec.model.v0_5 import TensorId

from tiktorch.server.session import process


@pytest.fixture
def srv_port():
    return getenv("TEST_TIKTORCH_PORT", randint(5500, 8000))


@pytest.fixture
def pub_port():
    return getenv("TEST_TIKTORCH_PUB_PORT", randint(8000, 9999))


@pytest.fixture(scope="session", autouse=True)
def register_faulthandler():
    if not sys.platform.startswith("win"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)


class QueueListener(logging.handlers.QueueListener):
    def start(self):
        # Redefine to provide meaningful thread name
        self._thread = t = threading.Thread(target=self._monitor, name="QueueListener")
        t.daemon = True
        t.start()


@pytest.fixture(scope="module")
def log_queue():
    q = mp.Queue()

    logger = logging.getLogger()

    listener = QueueListener(q, *logger.handlers)
    listener.start()

    yield q

    listener.stop()


@pytest.fixture(scope="session")
def assert_threads_cleanup():
    yield
    running_threads = [str(t) for t in threading.enumerate() if t != threading.current_thread() and not t.daemon]
    if len(running_threads):
        pytest.fail("Threads still running:\n\t%s" % "\n\t".join(running_threads))


MockedPredictionPipeline = Generator[Tuple[PredictionPipeline, Sample], None, None]


def patched_prediction_pipeline(mocked_prediction_pipeline: PredictionPipeline):
    return patch.object(process, "_get_prediction_pipeline_from_model_bytes", lambda *args: mocked_prediction_pipeline)


@pytest.fixture
def bioimage_model_explicit_siso() -> MockedPredictionPipeline:
    mocked_prediction_pipeline, mocked_output_sample = _bioimage_model_siso(
        [
            v0_5.BatchAxis(),
            v0_5.ChannelAxis(channel_names=["channel1", "channel2"]),
            v0_5.SpaceInputAxis(id="x", size=10),
            v0_5.SpaceInputAxis(id="y", size=10),
        ]
    )
    with patched_prediction_pipeline(mocked_prediction_pipeline):
        yield mocked_prediction_pipeline, mocked_output_sample


@pytest.fixture
def bioimage_model_param_siso() -> MockedPredictionPipeline:
    mocked_prediction_pipeline, mocked_output_sample = _bioimage_model_siso(
        [
            v0_5.BatchAxis(),
            v0_5.ChannelAxis(channel_names=["channel1", "channel2"]),
            v0_5.SpaceInputAxis(id="x", size=v0_5.ParameterizedSize(min=10, step=2)),
            v0_5.SpaceInputAxis(id="y", size=v0_5.ParameterizedSize(min=20, step=3)),
        ]
    )
    with patched_prediction_pipeline(mocked_prediction_pipeline):
        yield mocked_prediction_pipeline, mocked_output_sample


def _bioimage_model_siso(input_axes: List[v0_5.InputAxis]) -> Tuple[PredictionPipeline, Sample]:
    """
    Mocked bioimageio prediction pipeline with single input single output
    """

    mocked_input = create_autospec(v0_5.InputTensorDescr)
    mocked_input.id = "input"
    mocked_input.axes = input_axes
    return _bioimage_model([mocked_input])


@pytest.fixture
def bioimage_model_miso() -> MockedPredictionPipeline:
    """
    Mocked bioimageio prediction pipeline with three inputs single output
    """

    mocked_input1 = create_autospec(v0_5.InputTensorDescr)
    mocked_input1.id = "input1"
    mocked_input1.axes = [
        v0_5.BatchAxis(),
        v0_5.ChannelAxis(channel_names=["channel1", "channel2"]),
        v0_5.SpaceInputAxis(id=AxisId("x"), size=10),
        v0_5.SpaceInputAxis(id=AxisId("y"), size=v0_5.SizeReference(tensor_id="input3", axis_id="x")),
    ]

    mocked_input2 = create_autospec(v0_5.InputTensorDescr)
    mocked_input2.id = "input2"
    mocked_input2.axes = [
        v0_5.BatchAxis(),
        v0_5.ChannelAxis(channel_names=["channel1", "channel2"]),
        v0_5.SpaceInputAxis(id=AxisId("x"), size=v0_5.ParameterizedSize(min=10, step=2)),
        v0_5.SpaceInputAxis(id=AxisId("y"), size=v0_5.ParameterizedSize(min=10, step=5)),
    ]

    mocked_input3 = create_autospec(v0_5.InputTensorDescr)
    mocked_input3.id = "input3"
    mocked_input3.axes = [
        v0_5.BatchAxis(),
        v0_5.ChannelAxis(channel_names=["channel1", "channel2"]),
        v0_5.SpaceInputAxis(id="x", size=v0_5.SizeReference(tensor_id="input2", axis_id="x")),
        v0_5.SpaceInputAxis(id="y", size=10),
    ]

    mocked_prediction_pipeline, mocked_output_sample = _bioimage_model([mocked_input1, mocked_input2, mocked_input3])
    with patched_prediction_pipeline(mocked_prediction_pipeline):
        yield mocked_prediction_pipeline, mocked_output_sample


def _bioimage_model(inputs: List[v0_5.InputTensorDescr]) -> Tuple[PredictionPipeline, Sample]:
    mocked_descr = create_autospec(v0_5.ModelDescr)

    mocked_output = create_autospec(v0_5.OutputTensorDescr)
    mocked_output.id = "output"
    mocked_output.axes = [
        v0_5.BatchAxis(),
        v0_5.ChannelAxis(channel_names=["channel1", "channel2"]),
        v0_5.SpaceInputAxis(id=AxisId("x"), size=20),
        v0_5.SpaceInputAxis(id=AxisId("y"), size=20),
    ]
    mocked_descr.inputs = inputs
    mocked_descr.outputs = [mocked_output]

    mocked_output_sample = Sample(
        members={
            TensorId("output"): Tensor.from_xarray(
                xr.DataArray(np.arange(2 * 20 * 20).reshape((1, 2, 20, 20)), dims=["batch", "channel", "x", "y"])
            )
        },
        id=None,
        stat={},
    )

    mocked_prediction_pipeline = create_autospec(PredictionPipeline)
    mocked_prediction_pipeline.model_description = mocked_descr
    mocked_prediction_pipeline.predict_sample_without_blocking.return_value = mocked_output_sample
    return mocked_prediction_pipeline, mocked_output_sample
