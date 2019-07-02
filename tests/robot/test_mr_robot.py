import pytest
from mr_robot.mr_robot import MrRobot
from mr_robot.utils import tile_image
from tiktorch.server import TikTorchServer


def test_tile_image():
    # when image dim are multiple of patch size
    tiled_indices = tile_image((2, 2, 2), (2, 2, 1))
    assert tiled_indices == [
        (slice(0, 2, None), slice(0, 2, None), slice(0, 1, None)),
        (slice(0, 2, None), slice(0, 2, None), slice(1, 2, None)),
    ]

    tiled_indices = tile_image((2, 2, 2), (1, 1, 1))
    assert tiled_indices == [
        (slice(0, 1, None), slice(0, 1, None), slice(0, 1, None)),
        (slice(0, 1, None), slice(0, 1, None), slice(1, 2, None)),
        (slice(0, 1, None), slice(1, 2, None), slice(0, 1, None)),
        (slice(0, 1, None), slice(1, 2, None), slice(1, 2, None)),
        (slice(1, 2, None), slice(0, 1, None), slice(0, 1, None)),
        (slice(1, 2, None), slice(0, 1, None), slice(1, 2, None)),
        (slice(1, 2, None), slice(1, 2, None), slice(0, 1, None)),
        (slice(1, 2, None), slice(1, 2, None), slice(1, 2, None)),
    ]

    # when image dimension are not multiple of patch size
    tiled_indices = tile_image((5, 5), (3, 3))
    assert tiled_indices == [
        (slice(0, 3, None), slice(0, 3, None)),
        (slice(0, 3, None), slice(2, 5, None)),
        (slice(2, 5, None), slice(0, 3, None)),
        (slice(2, 5, None), slice(2, 5, None)),
    ]

    tiled_indices = tile_image((10, 2, 2, 2), (5, 2, 1, 1))
    assert tiled_indices == [
        (slice(0, 5, None), slice(0, 2, None), slice(0, 1, None), slice(0, 1, None)),
        (slice(0, 5, None), slice(0, 2, None), slice(0, 1, None), slice(1, 2, None)),
        (slice(0, 5, None), slice(0, 2, None), slice(1, 2, None), slice(0, 1, None)),
        (slice(0, 5, None), slice(0, 2, None), slice(1, 2, None), slice(1, 2, None)),
        (slice(5, 10, None), slice(0, 2, None), slice(0, 1, None), slice(0, 1, None)),
        (slice(5, 10, None), slice(0, 2, None), slice(0, 1, None), slice(1, 2, None)),
        (slice(5, 10, None), slice(0, 2, None), slice(1, 2, None), slice(0, 1, None)),
        (slice(5, 10, None), slice(0, 2, None), slice(1, 2, None), slice(1, 2, None)),
    ]

    # when image too small for the patch
    with pytest.raises(AssertionError):
        tiled_indices = tile_image((1, 48, 48), (1, 64, 32))
        tiled_indices = tile_image((64, 64), (2, 1, 1))


def test_MrRobot():

    robo = MrRobot("D:/Machine Learning/tiktorch/mr_robot/robot_config.yml", "strategy1")
    assert isinstance(robo, MrRobot)
    assert isinstance(robo.new_server, TikTorchServer)

    # assert isinstance(robo.slicer, list)
    robo._load_model()
    # robo._resume()
    # robo._predict()
    # assert len(strategy.patched_data) == 4
    robo._run()
    # robo.terminate()
    # print(robo.)
