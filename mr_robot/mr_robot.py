import logging
import os
import zipfile

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
import z5py

from tensorboardX import SummaryWriter

from tiktorch.server import TikTorchServer
from tiktorch.types import NDArray, NDArrayBatch
from mr_robot.utils import tile_image

img_dim = 32
batch_size = 1


class MrRobot:
    """ The robot class runs predictins on the model, and feeds the
    worst performing patch back for training. The order in which patches
    are feed back is determined by the 'strategy'. The robot can change
    strategies as training progresses.

    Args:
    path_to_config_file (string): path to the robot configuration file to
                                  load necessary variables
    strategy (string): strategy to follow (atleast intially)
    """

    def __init__(self, path_to_config_file, strategy):
        # start the server
        self.new_server = TikTorchServer()

        with open(path_to_config_file, mode="r") as f:
            self.base_config = yaml.load(f)

        strategy_class = strategies[strategy]
        self.strategy = strategy_class(self.base_config["training"]["loss_criterion_config"]["method"])
        self.data_file = z5py.File(self.base_config["data_dir"]["base_folder"])

        image_shape = self.data_file[self.base_config["data_dir"]["path_to_raw_data"]].shape
        self.block_list = tile_image(image_shape, self.base_config["training"]["training_shape"])

        self.iterations_max = self.base_config.pop("max_robo_iterations")
        self.iterations_done = 0
        self.tensorboard_writer = SummaryWriter()
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
        self.base_config.pop("model_dir")

        self.new_server.load_model(self.base_config, model, binary_state, b"", ["cpu"])
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

        for block in self.block_list:
            # map each slicer with its corresponding index
            self.assign_id(block, x)
            # self.patch_id[block[0].start] = x
            x += 1

            path_to_input = self.base_config["data_dir"]["path_to_raw_data"]
            path_to_label = self.base_config["data_dir"]["path_to_labelled"]

            pred_output = self.new_server.forward(NDArray(self.data_file[path_to_input][block]))
            self.pred_output = pred_output.result()

            self.strategy.update_state(pred_output, self.data_file[path_to_label][block], block)

        self.logger.info("prediction run for iteration {}", self.iterations_done)

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
            self._predict()

            # log average loss for all patches per iteration to tensorboard
            avg = self.strategy.get_mean()
            self.tensorboard_writer.add_scalar("avg_loss", avg, self.iterations_done)
            # self.tensorboard_writer.add_image("confusion_matrix", getConfusionMatrixFromlables(self.pred_output))

            block_batch = self.strategy.get_next_batch(batch_size)
            self._add(block_batch)
            self.remove_key(block_batch)
            for block in block_batch:
                self.block_list.pop(self.patch_id[block[0].start])

            self._resume()

        self.terminate()

    def _add(self, new_block_batch):
        """ add a new batch of images to training data

        Args:
        new_block_batch (list): list of tuples, where each tuple is 
        a slicer corresponding to a block
        """

        new_label_batch = []
        for i in range(len(new_block_batch)):

            new_block = self.data_file[self.base_config["data_dir"]["path_to_raw_data"]][new_block_batch[i]]
            new_label = self.data_file[self.base_config["data_dir"]["path_to_labelled"]][new_block_batch[i]]

            new_block_batch[i] = NDArray(new_block.astype(float), (new_block_batch[i][0].start))
            new_label_batch.append(NDArray(new_label.astype(float), (new_block_batch[i][0].start)))

        # new_input = NDArray(self.data_file["path_to_raw_data"][block].astype(float), (block[0].start))
        # new_label = NDArray(self.data_file[self.base_config["path_to_labelled"]][block].astype(float), (block[0].start))
        self.new_server.update_training_data(NDArrayBatch(new_block_batch), NDArrayBatch(new_label_batch))

    def get_coordinate(self, block):
        """ return the starting co-ordinate of a block

        Args:
        block(tuple): tuple of slice objects, one per dimension
        """

        coordinate = []
        for slice_ in block:
            coordinate.append(slice_.start)

        return tuple(coordinate)

    def assign_id(self, block, index):
        coordinate = self.get_coordinate(block)
        self.patch_id[coordinate] = index

    def remove_key(self, block_batch):
        for block in block_batch:
            coordinate = self.get_coordinate(block)
            self.patch_id.pop(coordinate)

    # annotate worst patch
    def dense_annotate(self, x, y, label, image):
        raise NotImplementedError()

    def terminate(self):
        self.tensorboard_writer.close()
        self.new_server.shutdown()


class BaseStrategy:
    def __init__(self, loss_fn):
        # with open(path_to_config_file, mode="r") as f:
        #    self.base_config=yaml.load(f)

        self.patched_data = []
        self.loss_fn = loss_fn
        self.logger = logging.getLogger(__name__)

    def update_state(self, pred_output, target, block):
        """  computes loss corresponding to the output and target according to
        the given loss function and update patch data

        Args:
        predicted_output(np.ndarray) : output predicted by the model
        target(np.ndarray): ground truth
        loss_fn(string): loss metric
        block(tuple): tuple of slice objects, one per dimension, specifying the corresponding block
        """

        criterion_class = getattr(nn, self.loss_fn, None)
        assert criterion_class is not None, "Criterion {} not found.".format(method)
        criterion_class_obj = criterion_class()
        curr_loss = criterion_class_obj(
            torch.from_numpy(pred_output.result().as_numpy().astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
        self.patched_data.append((curr_loss, block))

    def get_next_batch(self):
        raise NotImplementedError()

    def rearrange(self):
        raise NotImplementedError()


class Strategy1(BaseStrategy):
    """ This strategy sorts the patches in descending order of their loss

    Args:
    path_to_config_file (string): path to the configuration file for the robot
    """

    def __init__(self, loss_fn):
        super().__init__(loss_fn)
        # self.patch_counter = -1

    def rearrange(self):
        """ rearranges the patches in descending order of their loss
        """
        self.patched_data.sort(reverse=True)

    def get_next_batch(self, batch_size=1):
        """ Feeds a batch of patches at a time to the robot in descending order of their loss

        Args:
        batch_size (int): number of patches to be returned, defaults to 1
        """
        # self.patch_counter += 1
        assert len(self.patched_data) >= batch_size, "batch_size too big for current dataset"

        self.rearrange()
        return_patch_set = [blocks for loss, blocks in self.patched_data[:batch_size]]
        self.patched_data.clear()
        return return_patch_set

    def get_mean(self):
        return np.mean([loss for loss, slice_ in self.patched_data])


class StrategyRandom(BaseStrategy):
    """ randomly selected a patch, or batch of patches
    and returns them to the robot

    Args:
    loss_fn (string): loss metric
    """

    def __init__(self, loss_fn):
        super().__init__(loss_fn)

    def rearrange(self):
        pass

    def get_next_batch(self, batch_size=1):
        assert len(self.patched_data) >= batch_size, "batch_size too big for current dataset"

        # rand_indices = np.random.randint(len(self.patched_data), size = )


class Strategy3(BaseStrategy):
    def __init__():
        super().__init__()


strategies = {"strategy1": Strategy1, "strategyrandom": StrategyRandom, "strategy3": Strategy3}
