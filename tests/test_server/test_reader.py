from itertools import permutations
from zipfile import ZipFile

import pytest

from tiktorch.server.exemplum import Exemplum
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
        assert isinstance(exemplum, Exemplum)

@pytest.mark.xfail
def test_eval_tensorflow_model_zip(pybio_dummy_tensorflow_model_bytes, cache_path):
    with ZipFile(pybio_dummy_tensorflow_model_bytes) as zf:
        exemplum = eval_model_zip(zf, devices=["cpu"], cache_path=cache_path)
        assert isinstance(exemplum, Exemplum)
