import pytest
from mr_robot.mr_robot import MrRobot
from mr_robot.utils import tile_image
from tiktorch.server import TikTorchServer


def test_tile_image():
    # when image dim are not multiple of patch size
    tiled_indices = tile_image((1, 48, 48), (1, 32, 32))
    assert len(tiled_indices) == 4
    tiled_indices = tile_image((3, 71, 71), (1, 23, 23))
    assert len(tiled_indices) == 48

    # when image too small for the patch
    with pytest.raises(AssertionError):
        tiled_indices = tile_image((1, 48, 48), (1, 64, 32))


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
