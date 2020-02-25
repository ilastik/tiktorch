import dataclasses
import io
import multiprocessing as mp
import os
import uuid
import zipfile
from concurrent.futures import Future
from multiprocessing.connection import Connection
from pathlib import Path
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy
from inferno.io.transform import Compose

from tiktorch import log
from tiktorch.configkeys import (
    TRAINING,
    TRANSFORMS,
    VALIDATION,
)
from tiktorch.rpc import Shutdown
from tiktorch.rpc.mp import MPServer
from tiktorch.server.datasets import DynamicDataset
from tiktorch.server.reader import eval_model_zip
from tiktorch.server.utils import get_transform
from . import worker
from .interface import ISession


def make_datasets(config):
    DEFAULT_TRANSFORM = {"Normalize": {"apply_to": [0]}}

    def composed_transforms(transforms):
        transforms = transforms or DEFAULT_TRANSFORM
        return Compose(*[get_transform(name, **kwargs) for name, kwargs in transforms.items()])

    training_transform = composed_transforms(config[TRAINING].get(TRANSFORMS))
    validation_transform = composed_transforms(config[VALIDATION].get(TRANSFORMS))

    return {
        TRAINING: DynamicDataset(transform=training_transform),
        VALIDATION: DynamicDataset(transform=validation_transform),
    }


# class SparseOneHot(Transform):
#     """Mask out the zero label """
#
#     def __init__(self, **super_kwargs):
#         super().__init__(**super_kwargs)
#
#     def batch_function(self, tensors):
#         prediction, target = tensors
#         mask = torch.zeros_like(prediction)
#         mask[target > 0] = 1
#         mask.requires_grad = False
#         one_hot_target = torch.zeros_like(prediction)
#         for c in range(one_hot_target.shape[1]):
#             label = c + 1
#             one_hot_target[:, c] = target == label
#
#         # mask prediction with mask
#         masked_prediction = prediction * mask
#         return masked_prediction, one_hot_target


@dataclasses.dataclass
class ModelInfo:
    # TODO: Test for model info
    name: str
    input_axes: str
    output_axes: str
    valid_shapes: List[List[Tuple[str, int]]]
    halo: List[Tuple[str, int]]


class SessionProcess(ISession):
    def __init__(self, model_zip: bytes, devices: List[str]) -> None:
        cache_path = os.getenv("PYBIO_CACHE_PATH", None)
        if cache_path is not None:
            cache_path = Path(cache_path)

        with zipfile.ZipFile(io.BytesIO(model_zip)) as model_file:
            self._model = eval_model_zip(model_file, devices, cache_path=cache_path)

        self._datasets = {}
        self._worker = worker.SessionWorker(self._model)

    def forward(self, input_tensor: numpy.ndarray) -> Future:
        res = self._worker.forward(input_tensor)
        return res

    def create_dataset(self, mean, stddev):
        id_ = uuid.uuid4().hex
        self._datasets[id_] = {"mean": mean, "stddev": stddev}
        return id_

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            self._model.name,
            self._model.input_axes,
            self._model.output_axes,
            valid_shapes=[self._model.input_shape],
            halo=self._model.halo,
        )

    def shutdown(self) -> Shutdown:
        self._worker.shutdown()
        return Shutdown()


def run_model_process(conn: Connection, model_zip: bytes, devices: List[str], log_queue: Optional[mp.Queue] = None):
    try:
        # from: https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    except ModuleNotFoundError:
        pass  # probably running on windows

    if log_queue:
        log.configure(log_queue)
    model_proc = SessionProcess(model_zip, devices)
    srv = MPServer(model_proc, conn)
    srv.listen()
