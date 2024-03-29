import contextlib
import shutil
import tempfile
from collections import namedtuple

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
