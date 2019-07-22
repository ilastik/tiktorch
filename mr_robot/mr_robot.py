import logging
import zipfile
import os

import concurrent.futures as cf
import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
import z5py
import random

from scipy import sparse
from io import BytesIO
from mr_robot.utils import get_confusion_matrix, integer_to_onehot, plot_confusion_matrix, tile_image, get_coordinate
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter

from tiktorch.models.dunet import DUNet
from tiktorch.rpc.utils import BatchedExecutor
from tiktorch.server import TikTorchServer
from tiktorch.types import NDArray, NDArrayBatch
from mr_robot.strategies.strategy import *

img_dim = 32
batch_size = 1


class MrRobot:
    """ The robot class runs predictins on the model, and feeds the worst performing patch back for training. 
    The order in which patches are feed back is determined by the 'strategy'. The robot applies a given strategy, 
    adds new patches to the training data and logs the metrics to tensorboard

    Args:
    path_to_config_file (string): path to the robot configuration file to
                                  load necessary variables
    strategy (string): strategy to follow (atleast intially)
    """

    def __init__(self, path_to_config_file, strategy):

        assert torch.cuda.device_count() == 1, f"Device count is {torch.cuda.device_count()}"
        # start the server
        self.new_server = TikTorchServer()

        with open(path_to_config_file, mode="r") as f:
            self.base_config = yaml.load(f)

        if self.base_config["data_dir"]["raw_data_base_folder"].endswith(".h5"):
            self.raw_data_file = h5py.File(self.base_config["data_dir"]["raw_data_base_folder"])
            self.labelled_data_file = h5py.File(self.base_config["data_dir"]["labelled_data_base_folder"])
        else:
            self.raw_data_file = z5py.File(self.base_config["data_dir"]["raw_data_base_folder"])
            self.labelled_data_file = z5py.File(self.base_config["data_dir"]["labelled_data_base_folder"])

        image_shape = self.raw_data_file[self.base_config["data_dir"]["path_to_raw_data"]].shape
        print(image_shape)
        self.block_list = tile_image(image_shape, self.base_config["training"]["training_shape"])
        print("number of patches: %s" % len(self.block_list))
        print()

        strategy_class = strategies[strategy]
        # self.strategy = strategy_class(
        #    self.base_config["training"]["loss_criterion_config"]["method"], self.base_config["class_dict"]
        # )
        # TO BE REMOVED ##
        paths = {"path_to_raw_data": "raw", "path_to_labelled": "labels"}
        strat0 = StrategyRandom(
            "MSELoss",
            {0: "background", 1: "cell"},
            self.raw_data_file,
            self.labelled_data_file,
            paths,
            "random_blob",
            0.6,
        )
        strat1 = HighestLoss(
            "MSELoss",
            {0: "background", 1: "cell"},
            self.raw_data_file,
            self.labelled_data_file,
            paths,
            "random_blob",
            0.6,
        )
        strat2 = ClassWiseLoss(
            "MSELoss",
            {0: "background", 1: "cell"},
            self.raw_data_file,
            self.labelled_data_file,
            paths,
            "random_blob",
            0.6,
        )
        strat3 = VideoLabelling(
            "MSELoss",
            {0: "background", 1: "cell"},
            self.raw_data_file,
            self.labelled_data_file,
            paths,
            "random_blob",
            0.6,
            (1, 512, 512),
        )
        self.strategy = StrategyAbstract(self.new_server, (strat3, 2))

        self.iterations_max = self.base_config.pop("max_robo_iterations")
        self.iterations_done = 0
        mr_robot_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tensorboard_writer = SummaryWriter(logdir=os.path.join(mr_robot_folder, "tests", "robot", "robo_logs"))
        self.logger = logging.getLogger(__name__)

    def _load_model(self):

        if self.base_config["model_dir"]["path_to_folder"].endswith(".zip"):
            archive = zipfile.ZipFile(self.base_config["model_dir"]["path_to_folder"], "r")
            model = archive.read(self.base_config["model_dir"]["path_in_folder_to_model"])
            binary_state = archive.read(self.base_config["model_dir"]["path_in_folder_to_state"])

        else:
            model = open(
                os.path.join(
                    self.base_config["model_dir"]["path_to_folder"],
                    self.base_config["model_dir"]["path_in_folder_to_model"],
                )
            )
            binary_state = open(
                os.path.join(
                    self.base_config["model_dir"]["path_to_folder"],
                    self.base_config["model_dir"]["path_in_folder_to_state"],
                )
            )

        # cleaning dictionary before passing to tiktorch
        # self.base_config.pop("model_dir")

        self.new_server.load_model(self.base_config, model, binary_state, b"", ["gpu:4"])
        # self.tensorboard_writer.add_graph(DUNet(1,1),torch.from_numpy(self.raw_data_file[self.base_config["data_dir"]["path_to_raw_data"]][0]) )
        self.logger.info("model loaded")

    def _resume(self):

        self.new_server.resume_training()
        # self.binary_state = self.new_server.get_model_state()
        self.logger.info("training resumed")

    def _predict(self):
        """ run prediction on the whole set of patches
        """
        # self.strategy.patched_data.clear()
        self.patch_id = dict()
        x = 0
        prediction_list = []
        path_to_input = self.base_config["data_dir"]["path_to_raw_data"]
        path_to_label = self.base_config["data_dir"]["path_to_labelled"]

        batch_maker = BatchedExecutor(batch_size=5)
        for block in self.block_list[:10]:
            # map each slicer with its corresponding index
            self.assign_id(block, x)
            # self.patch_id[block[0].start] = x
            x += 1
            # pred_output = self.new_server.forward(NDArray(self.raw_data_file[path_to_input][block]))
            prediction_list.append(
                batch_maker.submit(self.new_server.forward, NDArray(self.raw_data_file[path_to_input][block]))
            )
            # self.pred_output = pred_output.result().as_numpy()
            # print("hello")
            # self.strategy.update_state(self.pred_output, self.labelled_data_file[path_to_label][block], block)

        i = 0
        for prediction in cf.as_completed(prediction_list):
            block = self.block_list[i]
            i += 1
            self.strategy.update_state(
                prediction.result().as_numpy(), self.labelled_data_file[path_to_label][block], block
            )

        # self.logger.info("prediction run for iteration {}", self.iterations_done)

    def stop(self):
        """ function which determines when the robot should stop

        currently, it stops after robot has completed 'iterations_max' number of iterations
        """

        if self.iterations_done > self.iterations_max:
            return True
        else:
            self.iterations_done += 1
            return False

    def _run(self):
        """ Feed patches to tiktorch (add to the training data)

        The function fetches the patches in order decided by the strategy,
        removes it from the list of indices and feeds it to tiktorch
        """

        while not self.stop():
            print("robot running for iteration: %s" % self.iterations_done)

            self._predict()

            # log average loss and accuracy for all patches per iteration to tensorboard
            self.write_to_tensorboard()

            # model = torch.load(BytesIO(self.new_server.get_model_state()))
            # self.tensorboard_writer.add_graph(model)
            # file_writer = self.tensorboard_writer._get_file_writer()
            # file_writer.flush()

            data_batch = self.strategy.get_next_batch(batch_size)

            self._add(data_batch)
            self.remove_key([id for image, label, id in data_batch])

            self._resume()

        self.terminate()

    def _add(self, new_data_batch):
        """ add a new batch of images to training data

        Args:
        new_data_batch (list): list of tuples, where each tuple contains an image, its label and their block id
        """
        assert new_data_batch is not None, "No data provided!"

        new_inputs, new_labels = [], []
        for image, label, block_id in new_data_batch:
            new_inputs.append(NDArray(image.astype(np.float), block_id))
            new_labels.append(NDArray(label.astype(np.float), block_id))

        self.new_server.update_training_data(NDArrayBatch(new_inputs), NDArrayBatch(new_labels))

    def write_to_tensorboard(self):
        metric_data = self.strategy.get_metrics()
        # print("average loss: %s   average accuracy: %s" % (metric_data["avg_loss"], metric_data["avg_accuracy"] * 100))
        # print()
        self.tensorboard_writer.add_scalar("avg_loss", metric_data["avg_loss"], self.iterations_done)
        self.tensorboard_writer.add_scalar("avg_accuracy", metric_data["avg_accuracy"] * 100, self.iterations_done)
        self.tensorboard_writer.add_scalar("F1_score", metric_data["avg_f1_score"], self.iterations_done)
        self.tensorboard_writer.add_figure(
            "confusion_matrix", metric_data["confusion_matrix"], global_step=self.iterations_done
        )

    def assign_id(self, block, index):
        self.patch_id[get_coordinate(block)] = index

    def remove_key(self, ids):
        for id in ids:
            self.patch_id.pop(id, None)

    # annotate worst patch
    def dense_annotate(self, x, y, label, image):
        raise NotImplementedError()

    def terminate(self):
        self.tensorboard_writer.close()
        self.new_server.shutdown()


strategies = {
    "highestloss": HighestLoss,
    "strategyrandom": StrategyRandom,
    "randomsparseannotate": RandomSparseAnnotate,
    "densesparseannotate": DenseSparseAnnotate,
    "classwiseloss": ClassWiseLoss,
    "videolabelling": VideoLabelling,
    "strategyabstract": StrategyAbstract,
}
