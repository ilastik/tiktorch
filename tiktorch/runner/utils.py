import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence
from zipfile import ZipFile

from bioimageio.spec import load_node, nodes
from marshmallow import missing
from tiktorch.runner.prediction_pipeline import PredictionPipeline, create_prediction_pipeline

MODEL_EXTENSIONS = (".yaml", ".yml")
logger = logging.getLogger(__name__)


def guess_model_path(file_names: List[str]) -> Optional[str]:
    for file_name in file_names:
        if file_name.endswith(MODEL_EXTENSIONS):
            return file_name

    return None


def eval_model_zip(model_zip: ZipFile, devices: Sequence[str], *, preserve_batch_dim=False) -> PredictionPipeline:
    temp_path = Path(tempfile.mkdtemp(prefix="tiktorch_"))

    model_zip.extractall(temp_path)

    spec_file_str = guess_model_path([str(file_name) for file_name in temp_path.glob("*")])
    if not spec_file_str:
        raise Exception("Model config file not found, make sure that rdf.yaml file in the root of your model archive")

    bioimageio_model = load_node(Path(spec_file_str))
    ret = create_prediction_pipeline(
        bioimageio_model=bioimageio_model, devices=devices, preserve_batch_dim=preserve_batch_dim
    )

    def _on_error(function, path, exc_info):
        logger.warning("Failed to delete temp directory %s", path)

    shutil.rmtree(temp_path, onerror=_on_error)

    return ret


def get_nn_instance_from_source(
    node: nodes.Model,  # type: ignore  # Name "nodes.Model" is not defined ???
    **kwargs,
):
    if not isinstance(node, nodes.Model):  # type: ignore
        raise TypeError(node)

    if not isinstance(node.source, nodes.ImportedSource):  # type: ignore
        raise ValueError(
            f"Encountered unexpected node.source type {type(node.source)}. "  # type: ignore
            f"`get_nn_instance_from_source` requires _UriNodeTransformer and _SourceNodeTransformer to be applied beforehand."
        )

    joined_kwargs = {} if node.kwargs is missing else dict(node.kwargs)  # type: ignore
    joined_kwargs.update(kwargs)
    return node.source(**joined_kwargs)
