from itertools import permutations
from zipfile import ZipFile

import numpy as np
import pytest
import xarray
from numpy.testing import assert_array_almost_equal
from xarray.testing import assert_equal

from tiktorch.server.prediction_pipeline import PredictionPipeline
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
        assert isinstance(exemplum, PredictionPipeline)


def test_eval_tensorflow_model_zip(pybio_dummy_tensorflow_model_bytes, cache_path):
    with ZipFile(pybio_dummy_tensorflow_model_bytes) as zf:
        pipeline = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        assert isinstance(pipeline, PredictionPipeline)

        test_input = xarray.DataArray(np.zeros(shape=(1, 128, 128)), dims=("c", "y", "x"))
        out_arr = np.ones(shape=(1, 128, 128))
        out_arr.fill(42)
        test_output = xarray.DataArray(out_arr, dims=("c", "y", "x"))
        result = pipeline.forward(test_input)
        assert_equal(result, test_output)


def test_eval_torchscript_model_zip(pybio_unet2d_torchscript_bytes, cache_path):
    with ZipFile(pybio_unet2d_torchscript_bytes) as zf:
        adapter = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        assert isinstance(adapter, PredictionPipeline)


def test_eval_onnx_model_zip(pybio_unet2d_onnx_bytes, cache_path):
    with ZipFile(pybio_unet2d_onnx_bytes) as zf:
        adapter = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        assert isinstance(adapter, PredictionPipeline)
