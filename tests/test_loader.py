import pathlib
import yaml
import zipfile
import pytest
from itertools import permutations
from tiktorch.server.reader._base import guess_model_path, eval_model


@pytest.mark.parametrize("paths", permutations(["mymodel.model.yml", "file3.yml", "file.model", "model.yml"]))
def test_guess_model_path_with_model_file(paths):
    assert "mymodel.model.yml" == guess_model_path(paths)


@pytest.mark.parametrize("paths", permutations(["file3.yml", "file.model", "model.yml"]))
def test_guess_model_path_without_model_file(paths):
    assert guess_model_path(paths) is None


p = pathlib.Path(__file__)


def test_read_config():
    pzip = p.parent.parent / "unet_sample.zip"
    with zipfile.ZipFile(pzip) as zip_file:
        print(eval_model(zip_file))
