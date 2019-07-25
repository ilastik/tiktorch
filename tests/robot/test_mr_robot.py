import os
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt

from mr_robot.mr_robot import MrRobot
from mr_robot.strategies.strategy import HighestLoss, ClassWiseLoss, StrategyRandom, StrategyAbstract
from mr_robot.utils import tile_image, get_confusion_matrix, make_plot
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

    robo = MrRobot("/home/psharma/psharma/repos/tiktorch/mr_robot/robot_config.yml", "strategyabstract", "gpu:4")
    assert isinstance(robo, MrRobot)
    assert isinstance(robo.new_server, TikTorchServer)

    robo._load_model()
    robo._run()


def test_get_confusion_matrix():
    predicted = np.array([1, 2, 3])
    actual = np.array([2, 1, 3])
    classes = [0, 1, 2, 3]

    expected = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]

    res = get_confusion_matrix(predicted, actual, classes)
    assert expected == res.tolist()


def test_make_plot():
    stats = {
        "training_loss": [],
        "robo_predict_accuracy": [],
        "robo_predict_f1_score": [],
        "robo_predict_loss": [],
        "confusion_matrix": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "training_iterations": [],
        "number_of_patches": [],
    }

    for key in stats.keys():
        stats[key] = random.sample(range(0, 100), 10)
    stats["training_iterations"] = [i for i in range(0, 100, 10)]
    stats["number_of_patches"] = [i for i in range(0, 80, 8)]

    make_plot(stats)
    # plt.savefig()
    # plt.show()
