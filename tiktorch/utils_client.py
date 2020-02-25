import contextlib
import os
import pathlib
import shutil
import tempfile
import zipfile
from collections import namedtuple
from typing import Optional

import yaml

from .types import Model, ModelState

ZIP_EXT = ".zip"

NNModel = namedtuple("NNModel", ["model", "state"])


class ModelError(Exception):
    pass


def _read_bytes(filename):
    with open(filename, "rb") as file:
        return file.read()


@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


# def _tuple_to_list(dct):
#     for key, val in dct.items():
#         if isinstance(val, dict):
#             _tuple_to_list(val)
#         elif isinstance(val, tuple):
#             dct[key] = list(val)
#
#
# def _guess_model_path(path: str) -> Optional[str]:
#     path = pathlib.Path(path)
#     possible_paths = path.glob("**/tiktorch_config.yml")
#     config_path = next(possible_paths, None)
#
#     if not config_path:
#         return None
#
#     return config_path.parent
#
#
# def read_model(path: str) -> NNModel:
#     if path.endswith(ZIP_EXT):
#         with tempdir() as tmppath:
#             with zipfile.ZipFile(path, "r") as model_zip:
#                 model_zip.extractall(tmppath)
#                 path = _guess_model_path(tmppath)
#                 if not path:
#                     raise ModelError(f"Couldn't find config in {path}")
#
#                 return read_model(str(path))
#
#     try:
#         with open(os.path.join(path, "tiktorch_config.yml")) as file:
#             conf = yaml.load(file)
#             if "name" not in conf:
#                 conf["name"] = os.path.basename(os.path.normpath(path))
#             _tuple_to_list(conf)
#
#         code = _read_bytes(os.path.join(path, "model.py"))
#         state = _read_bytes(os.path.join(path, "state.nn"))
#     except Exception as e:
#         raise ModelError(f"Failed to load model {path}") from e
#
#     return NNModel(Model(code=code, config=conf), ModelState(model_state=state))
