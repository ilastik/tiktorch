import torch
import logging.config
from importlib import import_module
from functools import partial

from tests.handler.dummyserver import DummyServer
from tiktorch.handler.dryrun import DryRunProcess, in_subproc

# from tests.data.tiny_models import TinyConvNet2d
# model = TinyConvNet2d()

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {"default": {"level": "DEBUG", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"}},
        "loggers": {"": {"handlers": ["default"], "level": "DEBUG", "propagate": True}},
    }
)


def test_minimal_device_test():
    assert DryRunProcess.minimal_device_test(torch.device("cpu"))


def test_minimal_device_test_in_subproc():
    ret = in_subproc(DryRunProcess.minimal_device_test, torch.device("cpu"))
    assert ret.recv()


def test_confirm_training_shape(tiny_model_2d):
    tiny_model_2d['config'].update({"training_shape": (15, 15)})
    ts = DummyServer(**tiny_model_2d)
    try:
        ts.active_children()
        assert ts.listen(timeout=10) is not None
        ts.handler_conn.send(('set_devices', {"device_names": ["cpu"]}))
        answer = ts.listen(timeout=10)
        # todo: fix this test! Where is the answer?
        assert answer is not None
    except Exception:
        raise
    finally:
        ts.shutdown()
