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
