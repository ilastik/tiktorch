import numpy as np
import pytest
from pybio.spec.nodes import Preprocessing
from xarray import DataArray
from xarray.testing import assert_allclose

from tiktorch.server.prediction_pipeline._preprocessing import ADD_BATCH_DIM, make_preprocessing


def test_zero_mean_unit_variance_preprocessing():
    zero_mean_spec = Preprocessing(name="zero_mean_unit_variance", kwargs={})
    data = DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    expected = DataArray(
        np.array(
            [
                [-1.54919274, -1.16189455, -0.77459637],
                [-0.38729818, 0.0, 0.38729818],
                [0.77459637, 1.16189455, 1.54919274],
            ]
        ),
        dims=("x", "y"),
    )
    preprocessing = make_preprocessing([zero_mean_spec])
    result = preprocessing(data)
    assert_allclose(expected, result)


def test_zero_mean_unit_across_axes():
    zero_mean_spec = Preprocessing(name="zero_mean_unit_variance", kwargs={"axes": ("x", "y")})
    data = DataArray(np.arange(18).reshape(2, 3, 3), dims=("c", "x", "y"))
    expected = DataArray(
        np.array(
            [
                [-1.54919274, -1.16189455, -0.77459637],
                [-0.38729818, 0.0, 0.38729818],
                [0.77459637, 1.16189455, 1.54919274],
            ]
        ),
        dims=("x", "y"),
    )
    preprocessing = make_preprocessing([zero_mean_spec])
    result = preprocessing(data)
    assert_allclose(expected, result[0])


def test_unknown_preprocessing_should_raise():
    mypreprocessing = Preprocessing(name="mycoolpreprocessing", kwargs={"axes": ("x", "y")})
    with pytest.raises(NotImplementedError):
        make_preprocessing([mypreprocessing])


def test_add_batch_dim():
    add_batch = make_preprocessing([ADD_BATCH_DIM])

    data = DataArray(np.arange(18).reshape(2, 3, 3), dims=("c", "x", "y"))
    result = add_batch(data)
    assert result.shape == (1, 2, 3, 3)
    assert result.dims == ("b", "c", "x", "y")


def test_combination_of_preprocessing_steps_with_dims_specified():
    zero_mean_spec = Preprocessing(name="zero_mean_unit_variance", kwargs={"axes": ("x", "y")})
    data = DataArray(np.arange(18).reshape(2, 3, 3), dims=("c", "x", "y"))

    expected = DataArray(
        np.array(
            [
                [-1.54919274, -1.16189455, -0.77459637],
                [-0.38729818, 0.0, 0.38729818],
                [0.77459637, 1.16189455, 1.54919274],
            ]
        ),
        dims=("x", "y"),
    )

    preprocessing = make_preprocessing([ADD_BATCH_DIM, zero_mean_spec])
    result = preprocessing(data)
    assert_allclose(expected, result[0][0])
