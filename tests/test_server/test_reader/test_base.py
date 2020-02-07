import pathlib
import yaml
import zipfile
import pytest
from itertools import permutations

from tiktorch.server.exemplum import Exemplum
from tiktorch.server.reader._base import guess_model_path, eval_model


@pytest.mark.parametrize("paths", permutations(["mymodel.model.yml", "file3.yml", "file.model", "model.yml"]))
def test_guess_model_path_with_model_file(paths):
    assert "mymodel.model.yml" == guess_model_path(paths)


@pytest.mark.parametrize("paths", permutations(["file3.yml", "file.model", "model.yml"]))
def test_guess_model_path_without_model_file(paths):
    assert guess_model_path(paths) is None


def test_read_config(pybio_unet_zip):
    with zipfile.ZipFile(pybio_unet_zip) as zip_file:
        exemplum =  eval_model(zip_file)
        assert isinstance(exemplum, Exemplum)
