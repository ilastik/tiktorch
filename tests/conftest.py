from __future__ import annotations

import faulthandler
import io
import logging.handlers
import multiprocessing as mp
import signal
import sys
import tempfile
import threading
from datetime import datetime
from enum import Enum
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
from bioimageio.spec.model import v0_4
from bioimageio.spec.model.v0_5 import (
    ArchitectureFromLibraryDescr,
    Author,
    BatchAxis,
    ChannelAxis,
    CiteEntry,
    Doi,
    FileDescr,
    HttpUrl,
    Identifier,
    InputAxis,
    InputTensorDescr,
    LicenseId,
    ModelDescr,
    OutputAxis,
    OutputTensorDescr,
    ParameterizedSize,
    PytorchStateDictWeightsDescr,
    SizeReference,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    TorchscriptWeightsDescr,
    Version,
    WeightsDescr,
)
from torch import nn


class WeightsFormat(Enum):
    PYTORCH = ("pytorch",)
    TORCHSCRIPT = "torchscript"


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


@pytest.fixture(params=[WeightsFormat.PYTORCH, WeightsFormat.TORCHSCRIPT])
def bioimage_model_explicit_siso(request) -> Tuple[io.BytesIO, xr.DataArray]:
    input_axes = [
        BatchAxis(),
        ChannelAxis(channel_names=[Identifier("channel1"), Identifier("channel2")]),
        SpaceInputAxis(id=AxisId("x"), size=10),
        SpaceInputAxis(id=AxisId("y"), size=10),
    ]
    input_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    if request.param == WeightsFormat.PYTORCH:
        return _bioimage_model_dummy_v5_siso_pytorch(input_axes, input_test_tensor)
    elif request.param == WeightsFormat.TORCHSCRIPT:
        return _bioimage_model_dummy_v5_siso_torchscript(input_axes, input_test_tensor)
    else:
        raise NotImplementedError(f"{request.param}")


@pytest.fixture(params=[WeightsFormat.PYTORCH, WeightsFormat.TORCHSCRIPT])
def bioimage_model_param_siso(request) -> Tuple[io.BytesIO, xr.DataArray]:
    input_test_tensor = np.arange(1 * 2 * 10 * 20, dtype="float32").reshape(1, 2, 10, 20)
    input_axes = [
        BatchAxis(),
        ChannelAxis(channel_names=[Identifier("channel1"), Identifier("channel2")]),
        SpaceInputAxis(id=AxisId("x"), size=ParameterizedSize(min=10, step=2)),
        SpaceInputAxis(id=AxisId("y"), size=ParameterizedSize(min=20, step=3)),
    ]
    if request.param == WeightsFormat.PYTORCH:
        return _bioimage_model_dummy_v5_siso_pytorch(input_axes, input_test_tensor)
    elif request.param == WeightsFormat.TORCHSCRIPT:
        return _bioimage_model_dummy_v5_siso_torchscript(input_axes, input_test_tensor)
    else:
        raise NotImplementedError(f"{request.param}")


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
        id=TensorId("input1"),
        axes=[
            BatchAxis(),
            ChannelAxis(channel_names=[Identifier("channel1"), Identifier("channel2")]),
            SpaceInputAxis(id=AxisId("x"), size=10),
            SpaceInputAxis(id=AxisId("y"), size=SizeReference(tensor_id=TensorId("input3"), axis_id=AxisId("y"))),
        ],
        description="",
        test_tensor=FileDescr(source=Path(test_tensor1_file.name)),
    )

    input2 = InputTensorDescr(
        id=TensorId("input2"),
        axes=[
            BatchAxis(),
            ChannelAxis(channel_names=[Identifier("channel1"), Identifier("channel2")]),
            SpaceInputAxis(id=AxisId("x"), size=ParameterizedSize(min=10, step=2)),
            SpaceInputAxis(id=AxisId("y"), size=ParameterizedSize(min=10, step=5)),
        ],
        description="",
        test_tensor=FileDescr(source=Path(test_tensor1_file.name)),
    )

    input3 = InputTensorDescr(
        id=TensorId("input3"),
        axes=[
            BatchAxis(),
            ChannelAxis(channel_names=[Identifier("channel1"), Identifier("channel2")]),
            SpaceInputAxis(id=AxisId("x"), size=SizeReference(tensor_id=TensorId("input2"), axis_id=AxisId("x"))),
            SpaceInputAxis(id=AxisId("y"), size=10),
        ],
        description="",
        test_tensor=FileDescr(source=Path(test_tensor1_file.name)),
    )

    dummy_model = _DummyNetwork()
    expected_output = _dummy_network_output
    with tempfile.NamedTemporaryFile(suffix=".pts", delete=False) as weights_file:
        torch.save(dummy_model.state_dict(), weights_file.name)
    weights = WeightsDescr(
        pytorch_state_dict=PytorchStateDictWeightsDescr(
            source=Path(weights_file.name),
            architecture=ArchitectureFromLibraryDescr(
                import_from="tests.conftest",
                callable=Identifier(f"{_DummyNetwork.__name__}"),
            ),
            pytorch_version=Version("1.1.1"),
        )
    )

    output_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    output_axes = [
        BatchAxis(),
        ChannelAxis(channel_names=[Identifier("channel1"), Identifier("channel2")]),
        SpaceOutputAxis(id=AxisId("x"), size=10),
        SpaceOutputAxis(id=AxisId("y"), size=10),
    ]

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as output_test_tensor_file:
        np.save(output_test_tensor_file.name, output_test_tensor)

    output_tensor = OutputTensorDescr(
        id=TensorId("output"),
        axes=output_axes,
        description="",
        test_tensor=FileDescr(source=Path(output_test_tensor_file.name)),
    )

    model_bytes = _bioimage_model_v5(weights=weights, inputs=[input1, input2, input3], outputs=[output_tensor])
    return model_bytes, expected_output


def _bioimage_model_dummy_v5_siso_torchscript(
    input_axes: List[InputAxis], input_test_tensor: np.ndarray
) -> Tuple[io.BytesIO, xr.DataArray]:
    dummy_model = _DummyNetwork()
    expected_output = _dummy_network_output
    traced_model = torch.jit.trace(dummy_model, example_inputs=torch.from_numpy(input_test_tensor))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as model_file:
        traced_model.save(model_file.name)
    weights = WeightsDescr(
        torchscript=TorchscriptWeightsDescr(source=Path(model_file.name), pytorch_version=Version("1.1.1"))
    )

    output_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    output_axes = [
        BatchAxis(),
        ChannelAxis(channel_names=[Identifier("channel1"), Identifier("channel2")]),
        SpaceOutputAxis(id=AxisId("x"), size=10),
        SpaceOutputAxis(id=AxisId("y"), size=10),
    ]

    return (
        _bioimage_model_v5_siso(
            weights=weights,
            input_axes=input_axes,
            output_axes=output_axes,
            input_test_tensor=input_test_tensor,
            output_test_tensor=output_test_tensor,
        ),
        expected_output,
    )


def _bioimage_model_dummy_v5_siso_pytorch(
    input_axes: List[InputAxis], input_test_tensor: np.ndarray
) -> Tuple[io.BytesIO, xr.DataArray]:
    dummy_model = _DummyNetwork()
    expected_output = _dummy_network_output
    with tempfile.NamedTemporaryFile(suffix=".pts", delete=False) as weights_file:
        torch.save(dummy_model.state_dict(), weights_file.name)
    weights = WeightsDescr(
        pytorch_state_dict=PytorchStateDictWeightsDescr(
            source=Path(weights_file.name),
            architecture=ArchitectureFromLibraryDescr(
                import_from="tests.conftest",
                callable=Identifier(f"{_DummyNetwork.__name__}"),
            ),
            pytorch_version=Version("1.1.1"),
        )
    )

    output_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    output_axes = [
        BatchAxis(),
        ChannelAxis(channel_names=[Identifier("channel1"), Identifier("channel2")]),
        SpaceOutputAxis(id=AxisId("x"), size=10),
        SpaceOutputAxis(id=AxisId("y"), size=10),
    ]

    return (
        _bioimage_model_v5_siso(
            weights=weights,
            input_axes=input_axes,
            output_axes=output_axes,
            input_test_tensor=input_test_tensor,
            output_test_tensor=output_test_tensor,
        ),
        expected_output,
    )


def _bioimage_model_v5_siso(
    weights: WeightsDescr,
    input_axes: List[InputAxis],
    output_axes: List[OutputAxis],
    input_test_tensor: np.ndarray,
    output_test_tensor: np.ndarray,
) -> io.BytesIO:
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as input_test_tensor_file:
        np.save(input_test_tensor_file.name, input_test_tensor)

    input_tensor = InputTensorDescr(
        id=TensorId("input"),
        axes=input_axes,
        description="",
        test_tensor=FileDescr(source=Path(input_test_tensor_file.name)),
    )

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as output_test_tensor_file:
        np.save(output_test_tensor_file.name, output_test_tensor)

    output_tensor = OutputTensorDescr(
        id=TensorId("output"),
        axes=output_axes,
        description="",
        test_tensor=FileDescr(source=Path(output_test_tensor_file.name)),
    )
    return _bioimage_model_v5(weights=weights, inputs=[input_tensor], outputs=[output_tensor])


def _bioimage_model_v5(
    weights: WeightsDescr, inputs: List[InputTensorDescr], outputs: List[OutputTensorDescr]
) -> io.BytesIO:
    mocked_descr = ModelDescr(
        name="mocked v5 model",
        description="A test model for demonstration purposes only",
        authors=[Author(name="me", affiliation="my institute", github_user="bioimageiobot")],
        # change github_user to your GitHub account name
        cite=[CiteEntry(text="for model training see my paper", doi=Doi("10.1234something"))],
        license=LicenseId("MIT"),
        documentation=HttpUrl("https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/README.md"),
        git_repo=HttpUrl("https://github.com/bioimage-io/spec-bioimage-io"),
        inputs=inputs,
        outputs=outputs,
        weights=weights,
    )
    model_bytes = io.BytesIO()
    save_bioimageio_package_to_stream(mocked_descr, output_stream=model_bytes)
    return model_bytes


@pytest.fixture(params=[WeightsFormat.PYTORCH, WeightsFormat.TORCHSCRIPT])
def bioimage_model_v4(request) -> Tuple[io.BytesIO, xr.DataArray]:
    if request.param == WeightsFormat.PYTORCH:
        return _bioimage_model_dummy_v4_siso_pytorch()
    elif request.param == WeightsFormat.TORCHSCRIPT:
        return _bioimage_model_dummy_v4_siso_torchscript()
    else:
        raise NotImplementedError(f"{request.param}")


def _bioimage_model_dummy_v4_siso_pytorch() -> Tuple[io.BytesIO, xr.DataArray]:
    dummy_model = _DummyNetwork()
    dummy_model_expected_output = _dummy_network_output
    input_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    output_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    traced_model = torch.jit.trace(dummy_model, example_inputs=torch.from_numpy(input_test_tensor))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as weights_file:
        traced_model.save(weights_file.name)
    weights = v0_4.WeightsDescr(torchscript=v0_4.TorchscriptWeightsDescr(source=Path(weights_file.name)))
    model_bytes = _bioimage_model_v4_siso(
        weights=weights, input_test_tensor=input_test_tensor, output_test_tensor=output_test_tensor
    )
    return model_bytes, dummy_model_expected_output


def _bioimage_model_dummy_v4_siso_torchscript() -> Tuple[io.BytesIO, xr.DataArray]:
    dummy_model = _DummyNetwork()
    dummy_model_expected_output = _dummy_network_output
    input_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    output_test_tensor = np.arange(1 * 2 * 10 * 10, dtype="float32").reshape(1, 2, 10, 10)
    traced_model = torch.jit.trace(dummy_model, example_inputs=torch.from_numpy(input_test_tensor))
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as model_file:
        traced_model.save(model_file.name)
    weights = v0_4.WeightsDescr(torchscript=v0_4.TorchscriptWeightsDescr(source=Path(model_file.name)))
    model_bytes = _bioimage_model_v4_siso(
        weights=weights, input_test_tensor=input_test_tensor, output_test_tensor=output_test_tensor
    )
    return model_bytes, dummy_model_expected_output


def _bioimage_model_v4_siso(
    weights: v0_4.WeightsDescr, input_test_tensor: np.ndarray, output_test_tensor: np.ndarray
) -> io.BytesIO:
    input_tensor = v0_4.InputTensorDescr(
        name=v0_4.TensorName("input"), description="", axes="bcxy", shape=input_test_tensor.shape, data_type="float32"
    )

    output_tensor = v0_4.OutputTensorDescr(
        name=v0_4.TensorName("output"), description="", axes="bcxy", shape=output_test_tensor.shape, data_type="float32"
    )

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as output_test_tensor_file:
        np.save(output_test_tensor_file.name, output_test_tensor)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as input_test_tensor_file:
        np.save(input_test_tensor_file.name, input_test_tensor)

    model_descr = v0_4.ModelDescr(
        name="mocked v4 model",
        authors=[v0_4.Author(name="me")],
        cite=[v0_4.CiteEntry(text="for model training see my paper", url=HttpUrl("https://doi.org/10.1234something"))],
        description="",
        inputs=[input_tensor],
        outputs=[output_tensor],
        documentation=HttpUrl("https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/README.md"),
        license="MIT",
        test_inputs=[Path(input_test_tensor_file.name)],
        test_outputs=[Path(output_test_tensor_file.name)],
        timestamp=v0_4.Datetime(root=datetime.now()),
        weights=weights,
    )

    model_bytes = io.BytesIO()
    save_bioimageio_package_to_stream(model_descr, output_stream=model_bytes)
    return model_bytes


_dummy_network_output = xr.DataArray(np.arange(2 * 10 * 10).reshape(1, 2, 10, 10), dims=["batch", "channel", "x", "y"])


class _DummyNetwork(nn.Module):
    def forward(self, *args):
        return torch.from_numpy(_dummy_network_output.values)
