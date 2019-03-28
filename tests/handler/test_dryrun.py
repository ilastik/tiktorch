import torch
import logging.config

from tiktorch.handler.dryrun import DryRunProcess
from tiktorch.rpc import Shutdown

from tiktorch.configkeys import TRAINING_SHAPE, INFERENCE_SHAPE_UPPER_BOUND

from tests.data.tiny_models import TinyConvNet2d


def test_minimal_device_test(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    dr = DryRunProcess(config=config, model=model)
    try:
        assert dr.minimal_device_test(torch.device("cpu"))
    finally:
        try:
            dr.shutdown()
        except Shutdown:
            pass


def test_with_given_training_shape(tiny_model_2d):
    config = tiny_model_2d["config"]
    config.update({TRAINING_SHAPE: (15, 15)})
    config.update({INFERENCE_SHAPE_UPPER_BOUND: (20, 20)})

    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    dr = DryRunProcess(config=config, model=model)
    try:
        fut = dr.dry_run(devices=[torch.device("cpu")])
        print(fut.result())
    finally:
        try:
            dr.shutdown()
        except Shutdown:
            pass


def test_with_given_malicious_training_shape(tiny_model_2d):
    config = tiny_model_2d["config"]
    config.update({TRAINING_SHAPE: (2, 2)})
    config.update({INFERENCE_SHAPE_UPPER_BOUND: (20, 20)})

    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    dr = DryRunProcess(config=config, model=model)
    try:
        fut = dr.dry_run(devices=[torch.device("cpu")])
        assert isinstance(fut.exception(), ValueError)
    finally:
        try:
            dr.shutdown()
        except Shutdown:
            pass
