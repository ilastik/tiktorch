import faulthandler
import io
import logging.handlers
import multiprocessing as mp
import os
import signal
import sys
import threading
from collections import namedtuple
from os import getenv
from pathlib import Path
from random import randint
from zipfile import ZipFile

import numpy as np
import pytest

TEST_DATA = "data"
TEST_PYBIO_ZIPFOLDER = "unet2d"
TEST_PYBIO_DUMMY = "dummy"
TEST_PYBIO_TENSORFLOW_DUMMY = "dummy_tensorflow"
TEST_PYBIO_TORCHSCRIPT = "unet2d_torchscript"

NNModel = namedtuple("NNModel", ["model", "state"])


@pytest.fixture
def data_path():
    conf_path = Path(__file__).parent
    return conf_path / TEST_DATA


def read_bytes(filename):
    with open(filename, "rb") as file:
        return file.read()


@pytest.fixture
def srv_port():
    return getenv("TEST_TIKTORCH_PORT", randint(5500, 8000))


@pytest.fixture
def pub_port():
    return getenv("TEST_TIKTORCH_PUB_PORT", randint(8000, 9999))


@pytest.fixture(scope="session", autouse=True)
def register_faulthandler():
    if not sys.platform.startswith("win"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)


class QueueListener(logging.handlers.QueueListener):
    def start(self):
        # Redefine to provide meaningful thread name
        self._thread = t = threading.Thread(target=self._monitor, name="QueueListener")
        t.daemon = True
        t.start()


@pytest.fixture(scope="module")
def log_queue():
    q = mp.Queue()

    logger = logging.getLogger()

    listener = QueueListener(q, *logger.handlers)
    listener.start()

    yield q

    listener.stop()


@pytest.fixture(scope="session")
def assert_threads_cleanup():
    yield
    running_threads = [str(t) for t in threading.enumerate() if t != threading.current_thread() and not t.daemon]
    if len(running_threads):
        pytest.fail("Threads still running:\n\t%s" % "\n\t".join(running_threads))


@pytest.fixture
def pybio_model_bytes(data_path):
    zip_folder = data_path / TEST_PYBIO_ZIPFOLDER
    data = io.BytesIO()
    with ZipFile(data, mode="w") as zip_model:
        for f_path in zip_folder.iterdir():
            if str(f_path.name).startswith("__"):
                continue

            with f_path.open(mode="rb") as f:
                zip_model.writestr(f_path.name, f.read())

    return data


@pytest.fixture
def pybio_model_zipfile(pybio_model_bytes):
    with ZipFile(pybio_model_bytes, mode="r") as zf:
        yield zf


@pytest.fixture
def pybio_dummy_model_filepath(data_path, tmpdir):
    pybio_net_dir = Path(data_path) / TEST_PYBIO_DUMMY
    path = tmpdir / "dummy_model.zip"

    with ZipFile(path, mode="w") as zip_model:
        for f_path in pybio_net_dir.iterdir():
            if str(f_path.name).startswith("__"):
                continue

            with f_path.open(mode="rb") as f:
                zip_model.writestr(f_path.name, f.read())

    return path


@pytest.fixture
def pybio_dummy_model_bytes(data_path):
    pybio_net_dir = Path(data_path) / TEST_PYBIO_DUMMY
    data = io.BytesIO()
    with ZipFile(data, mode="w") as zip_model:
        for f_path in pybio_net_dir.iterdir():
            if str(f_path.name).startswith("__"):
                continue

            with f_path.open(mode="rb") as f:
                zip_model.writestr(f_path.name, f.read())

    return data


def archive(directory):
    result = io.BytesIO()

    with ZipFile(result, mode="w") as zip_model:

        def _archive(path_to_archive):
            for path in path_to_archive.iterdir():
                if str(path.name).startswith("__"):
                    continue

                if path.is_dir():
                    _archive(path)

                else:
                    with path.open(mode="rb") as f:
                        zip_model.writestr(str(path).replace(str(directory), ""), f.read())

        _archive(directory)

    return result


@pytest.fixture
def pybio_dummy_tensorflow_model_bytes(data_path):
    pybio_net_dir = Path(data_path) / TEST_PYBIO_TENSORFLOW_DUMMY
    return archive(pybio_net_dir)


@pytest.fixture
def npy_zeros_file(tmpdir):
    path = str(tmpdir / "zeros.npy")
    zeros = np.zeros(shape=(64, 64))
    np.save(path, zeros)
    return path


@pytest.fixture
def pybio_unet2d_torchscript_bytes(data_path):
    pybio_net_dir = Path(data_path) / TEST_PYBIO_TORCHSCRIPT
    return archive(pybio_net_dir)


@pytest.fixture
def cache_path(tmp_path):
    return Path(getenv("PYBIO_CACHE_PATH", tmp_path))
