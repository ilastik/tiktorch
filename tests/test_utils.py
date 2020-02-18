import numpy
import pytest

from tiktorch.utils import generate_tile_rois
from tiktorch.utils_client import read_model


def test_read_model(nn_dir):
    read_model(nn_dir)


def test_read_model_zip(nn_zip):
    read_model(nn_zip)


generate_tile_rois_testdata = [
    ([10], [3], 4),
    ([3, 3], [2, 2], 4),
    ([2, 2], [2, 2], 1),
    ([2, 3], [2, 2], 2),
    ([3, 3, 3], [2, 2, 2], 8),
]


@pytest.mark.parametrize("array_shape,tile_shape,ntiles", generate_tile_rois_testdata)
def test_nr_rois_of_generate_tile_rois(array_shape, tile_shape, ntiles):
    out = list(generate_tile_rois(array_shape, tile_shape))
    assert len(out) == ntiles
