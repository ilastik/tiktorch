import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence
from zipfile import ZipFile

from bioimageio import spec

from tiktorch.server.prediction_pipeline import PredictionPipeline, create_prediction_pipeline

MODEL_FILENAMES = ("rdf.yaml", "rdf.yml")
MODEL_EXTENSIONS = (".yaml", ".yml")
logger = logging.getLogger(__name__)


def guess_model_path(file_names: List[str]) -> Optional[str]:
    for file_name in file_names:
        if file_name in MODEL_FILENAMES:
            return file_name

    for file_name in file_names:
        if file_name.endswith(MODEL_EXTENSIONS):
            return file_name

    return None


def eval_model_zip(model_zip: ZipFile, devices: Sequence[str], *, preserve_batch_dim=False) -> PredictionPipeline:
    temp_path = Path(tempfile.mkdtemp(prefix="tiktorch_"))

    model_zip.extractall(temp_path)

    spec_file_str = guess_model_path([str(file_name) for file_name in temp_path.glob("*")])
    if not spec_file_str:
        raise Exception(
            "Model config file not found, make sure that .model.yaml file in the root of your model archive"
        )

    bioimageio_model = spec.load_model(Path(spec_file_str))
    ret = create_prediction_pipeline(
        bioimageio_model=bioimageio_model, devices=devices, preserve_batch_dim=preserve_batch_dim
    )

    def _on_error(function, path, exc_info):
        logger.warning("Failed to delete temp directory %s", path)

    shutil.rmtree(temp_path, onerror=_on_error)

    return ret
