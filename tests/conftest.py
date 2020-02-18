import faulthandler
import logging.handlers
import multiprocessing as mp
import os
import pathlib
import pickle
import io
import signal
import sys
import threading
import zipfile
from collections import namedtuple
from os import getenv, path
from random import randint

import pytest
import yaml

from tiktorch.types import Model, ModelState

TEST_DATA = "data"
TEST_NET = "CREMI_DUNet_pretrained_new"
TEST_PYBIO_UNET = "unet2d"
TEST_PYBIO_DUMMY = "dummy"

NNModel = namedtuple("NNModel", ["model", "state"])


@pytest.fixture
def datadir(tmpdir):
    confdir = path.dirname(__file__)
    return path.join(confdir, TEST_DATA)


def _tuple_to_list(dct):
    for key, val in dct.items():
        if isinstance(val, dict):
            _tuple_to_list(val)
        elif isinstance(val, tuple):
            dct[key] = list(val)


def read_bytes(filename):
    with open(filename, "rb") as file:
        return file.read()


@pytest.fixture
def nn_zip(datadir):
    model_zip_fn = path.join(datadir, f"{TEST_NET}.zip")
    return model_zip_fn


@pytest.fixture
def pybio_nn_zip(datadir):
    return path.join(datadir, f"pybio_unet.zip")


@pytest.fixture
def nn_dir(tmpdir, nn_zip):
    tmp_model_dir = tmpdir / "models"
    tmp_model_dir.mkdir()

    with zipfile.ZipFile(nn_zip, "r") as model_zip:
        model_zip.extractall(tmp_model_dir)
        nn_dir = path.join(tmp_model_dir, TEST_NET)

    return nn_dir


@pytest.fixture
def nn_sample(nn_dir):
    with open(path.join(nn_dir, "tiktorch_config.yml")) as file:
        conf = yaml.load(file)
        _tuple_to_list(conf)

    code = read_bytes(path.join(nn_dir, "model.py"))
    state = read_bytes(path.join(nn_dir, "state.nn"))

    return NNModel(Model(code=code, config=conf), ModelState(model_state=state))


@pytest.fixture
def srv_port():
    return getenv("TEST_TIKTORCH_PORT", randint(5500, 8000))


@pytest.fixture
def pub_port():
    return getenv("TEST_TIKTORCH_PUB_PORT", randint(8000, 9999))


@pytest.fixture
def base_config():
    return {
        "model_state": b"",
        "optimizer_state": b"",
        "config": {
            "model_init_kwargs": {},
            "input_channels": 1,
            "training": {
                "batch_size": 2,
                "loss_criterion_config": {"method": "MSELoss"},
                "optimizer_config": {"method": "Adam"},
            },
            "validation": {},
        },
    }


@pytest.fixture
def tiny_model(datadir, base_config):
    with open(path.join(datadir, "tiny_models.py"), "rb") as f:
        base_config["model_file"] = f.read()

    base_config["config"]["model_class_name"] = "TestModel0"
    base_config["config"]["training"]["training_shape_upper_bound"] = (1, 15)
    return base_config


@pytest.fixture
def tiny_model_2d(datadir, base_config):
    with open(path.join(datadir, "tiny_models.py"), "rb") as f:
        base_config["model_file"] = f.read()

    base_config["config"]["model_class_name"] = "TinyConvNet2d"
    base_config["config"]["training"]["training_shape_upper_bound"] = (1, 15, 15)
    return base_config


@pytest.fixture
def tiny_model_3d(datadir, base_config):
    with open(path.join(datadir, "tiny_models.py"), "rb") as f:
        base_config["model_file"] = f.read()

    base_config["config"]["model_class_name"] = "TinyConvNet3d"
    base_config["config"]["training"]["training_shape_upper_bound"] = (1, 15, 15, 15)
    return base_config


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
def pybio_unet_zip(datadir):
    pybio_net_dir = pathlib.Path(datadir) / TEST_PYBIO_UNET
    data = io.BytesIO()
    with zipfile.ZipFile(data, mode="w") as zip_model:
        for f_path in pybio_net_dir.iterdir():
            with f_path.open(mode="rb") as f:
                zip_model.writestr(f_path.name, f.read())

    return data


@pytest.fixture
def pybio_dummy_zip(datadir):
    pybio_net_dir = pathlib.Path(datadir) / TEST_PYBIO_DUMMY
    data = io.BytesIO()
    with zipfile.ZipFile(data, mode="w") as zip_model:
        for f_path in pybio_net_dir.iterdir():
            with f_path.open(mode="rb") as f:
                zip_model.writestr(f_path.name, f.read())

    return data


@pytest.fixture
def cache_path(tmp_path):
    return pathlib.Path(os.getenv("PYBIO_CACHE_PATH", tmp_path))
