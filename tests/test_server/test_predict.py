import pytest
import sys


pytest_plugins = ["pytester"]


@pytest.fixture
def output_path(tmpdir):
    output = tmpdir / "output.zip"
    return str(output)


def test_running_predict_with_valid_arguments(testdir, pybio_dummy_model_filepath, npy_zeros_file, output_path):
    result = testdir.run("python", "-m", "tiktorch.server.predict", "--model", pybio_dummy_model_filepath, npy_zeros_file, "--output", output_path)
    assert result.ret == 0


def test_running_predict_fails_when_model_unspecified(testdir, npy_zeros_file, output_path):
    result = testdir.run("python", "-m", "tiktorch.server.predict", npy_zeros_file, "--output", output_path)
    assert result.ret != 0


def test_running_predict_fails_when_no_images_specified(testdir, pybio_dummy_model_filepath, output_path):
    result = testdir.run("python", "-m", "tiktorch.server.predict", "--model", pybio_dummy_model_filepath, "--output", output_path)
    assert result.ret != 0


def test_running_predict_fails_when_invalid_image_specified(testdir, pybio_dummy_model_filepath, output_path):
    result = testdir.run("python", "-m", "tiktorch.server.predict", "--model", pybio_dummy_model_filepath, "nonexisting", "--output", output_path)
    assert result.ret != 0


def test_running_predict_fails_without_when_model_file_does_not_exist(testdir, tmpdir, npy_zeros_file, output_path):
    result = testdir.run("python", "-m", "tiktorch.server.predict", "--model", str(tmpdir / "randomfile.zip"), npy_zeros_file, "--output", output_path)
    assert result.ret != 0


def test_running_predict_failes_when_output_is_unspecified(testdir, pybio_dummy_model_filepath, npy_zeros_file, output_path):
    result = testdir.run("python", "-m", "tiktorch.server.predict", "--model", pybio_dummy_model_filepath, npy_zeros_file)
    assert result.ret != 0
