import numpy
import pytest

from tiktorch.utils_client import read_model


def test_read_model(nn_dir):
    read_model(nn_dir)


def test_read_model_zip(nn_zip):
    read_model(nn_zip)
