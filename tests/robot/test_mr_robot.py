import os
import pytest

from mr_robot.mr_robot import MrRobot
from mr_robot.strategies.strategy import HighestLoss, ClassWiseLoss, StrategyRandom, StrategyAbstract
from mr_robot.utils import tile_image
from tiktorch.server import TikTorchServer

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    #paths = {path_to_raw_data: "raw", path_to_labelled: "labels"}

    #strat = StrategyAbstract()
    robo = MrRobot("/home/psharma/psharma/repos/tiktorch/mr_robot/robot_config.yml", "strategyabstract")
    assert isinstance(robo, MrRobot)
    assert isinstance(robo.new_server, TikTorchServer)

    robo._load_model()
    robo._run()

