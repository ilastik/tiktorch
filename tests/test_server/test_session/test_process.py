import pytest

from tiktorch.server.session.process import AxisWithValue, ParameterizedShape


@pytest.mark.parametrize(
    "min_shape, step, axes, expected",
    [
        ((512, 512), (10, 10), "yx", (512, 512)),
        ((256, 512), (10, 10), "yx", (256, 512)),
        ((256, 256), (2, 2), "yx", (512, 512)),
        ((128, 256), (2, 2), "yx", (384, 512)),
        ((64, 64, 64), (1, 1, 1), "zyx", (64, 64, 64)),
        ((2, 64, 64), (1, 1, 1), "zyx", (2, 64, 64)),
        ((2, 2, 64), (1, 1, 1), "zyx", (2, 2, 64)),
        ((2, 2, 32), (1, 1, 1), "zyx", (34, 34, 64)),
        ((42, 10, 512, 512), (0, 0, 10, 10), "tcyx", (42, 10, 512, 512)),
    ],
)
def test_enforce_min_shape(min_shape, step, axes, expected):
    shape = ParameterizedShape.from_values(min_shape, step, axes)
    assert shape.get_total_shape().values == expected


def test_param_shape_set_custom_multiplier():
    min_shape = (512, 512, 256)
    step = (2, 2, 2)
    axes = "zyx"

    shape = ParameterizedShape.from_values(min_shape, step, axes)
    shape.multiplier = 2
    assert shape.get_total_shape().values == (516, 516, 260)

    assert shape.get_total_shape(4).values == (520, 520, 264)
    assert shape.multiplier == 4

    with pytest.raises(ValueError):
        shape.multiplier = -1


@pytest.mark.parametrize(
    "sizes, axes, spatial_axes, spatial_sizes",
    [
        ((512, 512), "yx", "yx", (512, 512)),
        ((1, 256, 512), "tyx", "yx", (256, 512)),
        ((256, 1, 512), "ytx", "yx", (256, 512)),
        ((128, 256, 1), "yxt", "yx", (128, 256)),
        ((64, 64, 64), "zyx", "zyx", (64, 64, 64)),
        ((1, 2, 64, 64), "bzyx", "zyx", (2, 64, 64)),
        ((1, 2, 3, 64), "zbyx", "zyx", (1, 3, 64)),
        ((1, 2, 3, 4), "zybx", "zyx", (1, 2, 4)),
        ((1, 2, 3, 4, 5), "tczyx", "zyx", (3, 4, 5)),
    ],
)
def test_spatial_axes(sizes, axes, spatial_axes, spatial_sizes):
    shape = AxisWithValue(axes, sizes)
    assert shape.spatial_values == spatial_sizes
    assert shape.spatial_axes == spatial_axes
