import torch
import logging.config

from tiktorch.server.handler.dryrun import DryRunProcess

from tiktorch.configkeys import TRAINING, TRAINING_SHAPE, TRAINING_SHAPE_UPPER_BOUND, TRAINING_SHAPE_LOWER_BOUND

from tests.data.tiny_models import TinyConvNet2d


def test_initialization(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    dr = DryRunProcess(config=config, model=model)
    dr.shutdown()


def test_minimal_device_test(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    dr = DryRunProcess(config=config, model=model)
    try:
        assert dr.minimal_device_test([torch.device("cpu")])
    finally:
        dr.shutdown()


def test_given_training_shape(tiny_model_2d):
    config = tiny_model_2d["config"]
    config[TRAINING][TRAINING_SHAPE] = (1, 15, 15)
    config[TRAINING][TRAINING_SHAPE_UPPER_BOUND] = (1, 15, 15)

    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    dr = DryRunProcess(config=config, model=model)
    try:
        dr.dry_run(devices=[torch.device("cpu")]).result(timeout=30)
    finally:
        dr.shutdown()


def test_given_training_shape_intervall(tiny_model_2d):
    config = tiny_model_2d["config"]
    if TRAINING_SHAPE in config[TRAINING]:
        del config[TRAINING][TRAINING_SHAPE]

    config[TRAINING][TRAINING_SHAPE_LOWER_BOUND] = (1, 20, 20)
    config[TRAINING][TRAINING_SHAPE_UPPER_BOUND] = (1, 25, 25)

    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    dr = DryRunProcess(config=config, model=model)
    try:
        dr.dry_run(devices=[torch.device("cpu")]).result(timeout=30)
    finally:
        dr.shutdown()


def test_malicious_training_shape(tiny_model_2d):
    config = tiny_model_2d["config"]
    config[TRAINING].update({TRAINING_SHAPE: (1, 0, 20), TRAINING_SHAPE_UPPER_BOUND: (1, 2, 2)})

    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    dr = DryRunProcess(config=config, model=model)
    try:
        fut = dr.dry_run(devices=[torch.device("cpu")])
        assert isinstance(fut.exception(timeout=20), ValueError)
    finally:
        dr.shutdown()


class PickyModel(TinyConvNet2d):
    def forward(self, x):
        raise NotImplementedError()


def test_invalid_training_shape(tiny_model_2d):
    config = tiny_model_2d["config"]
    config[TRAINING][TRAINING_SHAPE] = (1, 15, 15)

    in_channels = config["input_channels"]

    model = PickyModel(in_channels=in_channels)

    dr = DryRunProcess(config=config, model=model)
    try:
        fut = dr.dry_run(devices=[torch.device("cpu")])
        assert isinstance(fut.exception(timeout=20), ValueError)
    finally:
        dr.shutdown()
