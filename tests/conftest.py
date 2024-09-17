import faulthandler
import io
import logging.handlers
import multiprocessing as mp
import signal
import sys
import tempfile
import threading
from os import getenv
from pathlib import Path
from random import randint
from typing import List, Tuple

import numpy as np
import pytest
import torch
import xarray as xr
from bioimageio.core import AxisId
from bioimageio.spec import save_bioimageio_package_to_stream
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromLibraryDescr,
    Author,
    BatchAxis,
    ChannelAxis,
    CiteEntry,
    Doi,
    FileDescr,
    HttpUrl,
    InputAxis,
    InputTensorDescr,
    LicenseId,
    ModelDescr,
    OutputTensorDescr,
    ParameterizedSize,
    PytorchStateDictWeightsDescr,
    SizeReference,
    SpaceInputAxis,
    SpaceOutputAxis,
    Version,
    WeightsDescr,
)
from torch import nn


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


@pytest.fixture
def bioimage_model_explicit_siso() -> Tuple[io.BytesIO, xr.DataArray]:
    test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    model_descr, expected_output = _bioimage_model_siso(
        [
            BatchAxis(),
            ChannelAxis(channel_names=["channel1", "channel2"]),
            SpaceInputAxis(id="x", size=10),
            SpaceInputAxis(id="y", size=10),
        ],
        test_tensor,
    )
    model_bytes = io.BytesIO()
    save_bioimageio_package_to_stream(model_descr, output_stream=model_bytes)
    return model_bytes, expected_output


@pytest.fixture
def bioimage_model_param_siso() -> Tuple[io.BytesIO, xr.DataArray]:
    test_tensor = np.arange(1 * 2 * 10 * 20, dtype="float32").reshape(1, 2, 10, 20)
    model_descr, expected_output = _bioimage_model_siso(
        [
            BatchAxis(),
            ChannelAxis(channel_names=["channel1", "channel2"]),
            SpaceInputAxis(id="x", size=ParameterizedSize(min=10, step=2)),
            SpaceInputAxis(id="y", size=ParameterizedSize(min=20, step=3)),
        ],
        test_tensor,
    )
    model_bytes = io.BytesIO()
    save_bioimageio_package_to_stream(model_descr, output_stream=model_bytes)
    return model_bytes, expected_output


def _bioimage_model_siso(input_axes: List[InputAxis], test_tensor: np.array) -> Tuple[ModelDescr, xr.DataArray]:
    """
    Mocked bioimageio prediction pipeline with single input single output
    """
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as test_tensor_file:
        np.save(test_tensor_file.name, test_tensor)

    input_tensor = InputTensorDescr(
        id="input", axes=input_axes, description="", test_tensor=FileDescr(source=Path(test_tensor_file.name))
    )
    return _bioimage_model([input_tensor])


@pytest.fixture
def bioimage_model_miso() -> Tuple[io.BytesIO, xr.DataArray]:
    """
    Mocked bioimageio prediction pipeline with three inputs single output
    """
    test_tensor1 = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    test_tensor2 = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    test_tensor3 = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as test_tensor1_file:
        np.save(test_tensor1_file.name, test_tensor1)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as test_tensor2_file:
        np.save(test_tensor2_file.name, test_tensor2)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as test_tensor3_file:
        np.save(test_tensor3_file.name, test_tensor3)

    input1 = InputTensorDescr(
        id="input1",
        axes=[
            BatchAxis(),
            ChannelAxis(channel_names=["channel1", "channel2"]),
            SpaceInputAxis(id=AxisId("x"), size=10),
            SpaceInputAxis(id=AxisId("y"), size=SizeReference(tensor_id="input3", axis_id="y")),
        ],
        description="",
        test_tensor=FileDescr(source=Path(test_tensor1_file.name)),
    )

    input2 = InputTensorDescr(
        id="input2",
        axes=[
            BatchAxis(),
            ChannelAxis(channel_names=["channel1", "channel2"]),
            SpaceInputAxis(id=AxisId("x"), size=ParameterizedSize(min=10, step=2)),
            SpaceInputAxis(id=AxisId("y"), size=ParameterizedSize(min=10, step=5)),
        ],
        description="",
        test_tensor=FileDescr(source=Path(test_tensor1_file.name)),
    )

    input3 = InputTensorDescr(
        id="input3",
        axes=[
            BatchAxis(),
            ChannelAxis(channel_names=["channel1", "channel2"]),
            SpaceInputAxis(id="x", size=SizeReference(tensor_id="input2", axis_id="x")),
            SpaceInputAxis(id="y", size=10),
        ],
        description="",
        test_tensor=FileDescr(source=Path(test_tensor1_file.name)),
    )

    model_descr, expected_output = _bioimage_model([input1, input2, input3])
    model_bytes = io.BytesIO()
    save_bioimageio_package_to_stream(model_descr, output_stream=model_bytes)
    return model_bytes, expected_output


def _bioimage_model(inputs: List[InputTensorDescr]) -> Tuple[ModelDescr, xr.DataArray]:
    test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as test_tensor_file:
        np.save(test_tensor_file.name, test_tensor)

    dummy_model = _DummyNetwork()
    with tempfile.NamedTemporaryFile(suffix=".pts", delete=False) as weights_file:
        torch.save(dummy_model.state_dict(), weights_file.name)

    output_tensor = OutputTensorDescr(
        id="output",
        axes=[
            BatchAxis(),
            ChannelAxis(channel_names=["channel1", "channel2"]),
            SpaceOutputAxis(id=AxisId("x"), size=10),
            SpaceOutputAxis(id=AxisId("y"), size=10),
        ],
        description="",
        test_tensor=FileDescr(source=Path(test_tensor_file.name)),
    )

    mocked_descr = ModelDescr(
        name="mocked model",
        description="A test model for demonstration purposes only",
        authors=[Author(name="me", affiliation="my institute", github_user="bioimageiobot")],
        # change github_user to your GitHub account name
        cite=[CiteEntry(text="for model training see my paper", doi=Doi("10.1234something"))],
        license=LicenseId("MIT"),
        documentation=HttpUrl("https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/README.md"),
        git_repo=HttpUrl("https://github.com/bioimage-io/spec-bioimage-io"),
        inputs=inputs,
        outputs=[output_tensor],
        weights=WeightsDescr(
            pytorch_state_dict=PytorchStateDictWeightsDescr(
                source=weights_file.name,
                architecture=ArchitectureFromLibraryDescr(
                    import_from="tests.conftest", callable=_DummyNetwork.__name__
                ),
                pytorch_version=Version("1.1.1"),
            )
        ),
    )
    return mocked_descr, _dummy_network_output


_dummy_network_output = xr.DataArray(np.arange(2 * 10 * 10).reshape(1, 2, 10, 10), dims=["batch", "channel", "x", "y"])


class _DummyNetwork(nn.Module):
    def forward(self, *args):
        return _dummy_network_output
