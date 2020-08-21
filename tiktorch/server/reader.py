import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence
from zipfile import ZipFile

import torch

from pybio import spec
from pybio.spec.utils import train
from tiktorch.server.exemplum import Exemplum

MODEL_EXTENSIONS = (".model.yaml", ".model.yml")
logger = logging.getLogger(__name__)


def guess_model_path(file_names: List[str]) -> Optional[str]:
    for file_name in file_names:
        if file_name.endswith(MODEL_EXTENSIONS):
            return file_name

    return None


def eval_model_zip(model_zip: ZipFile, devices: Sequence[str], cache_path: Optional[Path] = None):
    temp_path = Path(tempfile.mkdtemp(prefix="tiktorch"))
    if cache_path is None:
        cache_path = temp_path / "cache"

    model_zip.extractall(temp_path)

    spec_file_str = guess_model_path([str(file_name) for file_name in temp_path.glob("*")])
    if not spec_file_str:
        raise Exception(
            "Model config file not found, make sure that .model.yaml file in the root of your model archive"
        )

    pybio_model = spec.utils.load_model(spec_file_str, root_path=temp_path, cache_path=cache_path)

    devices = [torch.device(d) for d in devices]
    if pybio_model.spec.training is None:
        return Exemplum(pybio_model=pybio_model, _devices=devices)
    else:
        ret = train(pybio_model, _devices=devices)
        assert isinstance(ret, Exemplum)

    def _on_errror(function, path, exc_info):
        logger.warning("Failed to delete temp directory %s", path)

    shutil.rmtree(temp_path, on_error=_on_errror)

    return ret
