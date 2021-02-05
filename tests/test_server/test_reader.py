from itertools import permutations
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from tiktorch.server.model_adapter import ModelAdapter
from tiktorch.server.model_adapter._torchscript_model_adapter import TorchscriptModelAdapter
from tiktorch.server.model_adapter._onnx_model_adapter import ONNXModelAdapter
from tiktorch.server.reader import eval_model_zip, guess_model_path


@pytest.mark.parametrize("paths", permutations(["mymodel.model.yml", "file3.yml", "file.model", "model.yml"]))
def test_guess_model_path_with_model_file(paths):
    assert "mymodel.model.yml" == guess_model_path(paths)


@pytest.mark.parametrize("paths", permutations(["file3.yml", "file.model", "model.yml"]))
def test_guess_model_path_without_model_file(paths):
    assert guess_model_path(paths) is None


def test_eval_model_zip(pybio_model_bytes, cache_path):
    with ZipFile(pybio_model_bytes) as zf:
        exemplum = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        assert isinstance(exemplum, ModelAdapter)


def test_eval_tensorflow_model_zip(pybio_dummy_tensorflow_model_bytes, cache_path):
    with ZipFile(pybio_dummy_tensorflow_model_bytes) as zf:
        exemplum = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        assert isinstance(exemplum, ModelAdapter)


def test_eval_torchscript_model_zip(pybio_unet2d_torchscript_bytes, cache_path):
    with ZipFile(pybio_unet2d_torchscript_bytes) as zf:
        adapter = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        assert isinstance(adapter, ModelAdapter)
        assert isinstance(adapter, TorchscriptModelAdapter)


@pytest.mark.xfail
def test_eval_torchscript_model_zip_predict(pybio_unet2d_torchscript_bytes,
                                            pybio_unet2d_torchscript_test_data,
                                            cache_path):
    with ZipFile(pybio_unet2d_torchscript_bytes) as zf:
        adapter = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        test_input = np.load(pybio_unet2d_torchscript_test_data["test_input"]).astype(np.float32)
        test_output = np.load(pybio_unet2d_torchscript_test_data["test_output"])
        result = adapter.forward(test_input)
        assert_array_almost_equal(result, test_output, decimal=3)


def test_eval_onnx_model_zip(pybio_unet2d_onnx_bytes, cache_path):
    with ZipFile(pybio_unet2d_onnx_bytes) as zf:
        adapter = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        assert isinstance(adapter, ModelAdapter)
        assert isinstance(adapter, ONNXModelAdapter)


@pytest.mark.xfail
def test_eval_onnx_model_zip_predict(pybio_unet2d_onnx_bytes, pybio_unet2d_onnx_test_data, cache_path):
    with ZipFile(pybio_unet2d_onnx_bytes) as zf:
        adapter = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        test_input = np.load(pybio_unet2d_onnx_test_data["test_input"]).astype(np.float32)
        test_output = np.load(pybio_unet2d_onnx_test_data["test_output"])
        result = adapter.forward(test_input)
        assert_array_almost_equal(result, test_output, decimal=3)
