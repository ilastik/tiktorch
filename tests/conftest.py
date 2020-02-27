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

import pytest

TEST_DATA = "data"
TEST_PYBIO_ZIPFOLDER = "unet2d"
TEST_PYBIO_DUMMY = "dummy"

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


@pytest.fixture
def cache_path(tmp_path):
    return Path(getenv("PYBIO_CACHE_PATH", tmp_path))
