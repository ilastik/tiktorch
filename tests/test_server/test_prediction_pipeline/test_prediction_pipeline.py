from zipfile import ZipFile

import numpy as np
import xarray
from numpy.testing import assert_array_almost_equal

from tiktorch.server.reader import eval_model_zip


def test_eval_onnx_model_zip_predict(pybio_unet2d_onnx_bytes, pybio_unet2d_onnx_test_data, cache_path):
    with ZipFile(pybio_unet2d_onnx_bytes) as zf:
        adapter = eval_model_zip(zf, devices=["cpu"], preserve_batch_dim=True)
        test_input = xarray.DataArray(np.load(pybio_unet2d_onnx_test_data["test_input"]), dims=("b", "c", "x", "y"))
        # TODO: Figure out why test output doesn't match result
        adapter.forward(test_input)


def test_eval_torchscript_model_zip_predict(
    pybio_unet2d_torchscript_bytes, pybio_unet2d_torchscript_test_data, cache_path
):
    with ZipFile(pybio_unet2d_torchscript_bytes) as zf:
        pipeline = eval_model_zip(zf, devices=["cpu"], preserve_batch_dim=True)
        test_input = xarray.DataArray(
            np.load(pybio_unet2d_torchscript_test_data["test_input"]).astype(np.float32), dims=("b", "c", "x", "y")
        )
        test_output = np.load(pybio_unet2d_torchscript_test_data["test_output"])
        result = pipeline.forward(test_input)
        assert_array_almost_equal(result.data, test_output, decimal=4)


def test_eval_model_zip_metadata_no_batch_dim(
    pybio_unet2d_torchscript_bytes, pybio_unet2d_torchscript_test_data, cache_path
):
    with ZipFile(pybio_unet2d_torchscript_bytes) as zf:
        pipeline = eval_model_zip(zf, devices=["cpu"], preserve_batch_dim=False)
        assert pipeline.name == "UNet 2D Nuclei Broad"
        assert pipeline.input_axes == "cyx"
        assert pipeline.output_axes == "cyx"
        assert pipeline.input_shape == [("c", 1), ("y", 512), ("x", 512)]
        assert pipeline.halo == [("c", 0), ("y", 32), ("x", 32)]


def test_eval_model_zip(pybio_model_bytes, cache_path):
    with ZipFile(pybio_model_bytes) as zf:
        pipeline = eval_model_zip(zf, devices=["cpu"], preserve_batch_dim=True)

    assert pipeline.input_axes == "bcyx"
    assert pipeline.output_axes == "bcyx"
    assert pipeline.input_shape == [("b", 1), ("c", 1), ("y", 512), ("x", 512)]
    assert pipeline.halo == [("b", 0), ("c", 0), ("y", 32), ("x", 32)]


def test_eval_model_zip_metadata_with_batch_dim(
    pybio_unet2d_torchscript_bytes, pybio_unet2d_torchscript_test_data, cache_path
):
    with ZipFile(pybio_unet2d_torchscript_bytes) as zf:
        pipeline = eval_model_zip(zf, devices=["cpu"], preserve_batch_dim=True)
        assert pipeline.input_axes == "bcyx"
        assert pipeline.output_axes == "bcyx"
        assert pipeline.input_shape == [("b", 1), ("c", 1), ("y", 512), ("x", 512)]
        assert pipeline.halo == [("b", 0), ("c", 0), ("y", 32), ("x", 32)]
