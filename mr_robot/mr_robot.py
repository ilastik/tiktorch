import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.metrics import mean_squared_error
import zipfile
import h5py
import z5py
from z5py.converter import convert_from_h5
from torch.autograd import Variable
from collections import OrderedDict
import yaml
import logging
from tensorboardX import SummaryWriter
from tiktorch.server import TikTorchServer
from tiktorch.rpc import Client, Server, InprocConnConf
from tiktorch.rpc_interface import INeuralNetworkAPI
from tiktorch.types import NDArray, NDArrayBatch
from mr_robot.utils import tile_image

patch_size = 16
img_dim = 32


class MrRobot:
    """ The robot class runs predictins on the model, and feeds the 
    worst performing patch back for training. The order in which patches 
    are feed back is determined by the 'strategy'. The robot can change 
    strategies as training progresses. 

    Args:
    path_to_config_file (string): path to the robot configuration file to
                                  load necessary variables
    strategy (Strategy object): strategy to follow (atleast intially)  
    """

    def __init__(self, path_to_config_file, strategy):
        # start the server
        self.new_server = TikTorchServer()
        self.strategy = strategy

        with open(path_to_config_file, mode="r") as f:
            self.base_config = yaml.load(f)

        self.data_file = z5py.File(self.base_config.pop("raw_data_path"))

        self.tile_indices = tile_image(self.base_config["training"]["training_shape"], patch_size)
        self.input_shape = list((self.base_config["training"]["training_shape"]))
        self.slicer = [slice(0, i) for i in self.input_shape]

        self.iterations_max = self.base_config.pop("max_robo_iterations")
        self.iterations_done = 0
        self.tensorboard_writer = SummaryWriter()
        self.logger = logging.getLogger(__name__)
        plt.ion()

    # def load_data(self):
    #    self.f = z5py.File(self.base_config["cremi_data_dir"])
    #    self.logger("data file loaded")

    def _load_model(self):

        archive = zipfile.ZipFile(self.base_config["data_dir"]["path_to_zip"], "r")
        model = archive.read(self.base_config["data_dir"]["path_in_zip_to_model"])
        binary_state = archive.read(self.base_config["data_dir"]["path_in_zip_to_state"])

        # cleaning dictionary before passing to tiktorch
        self.base_config.pop("data_dir")

        self.new_server.load_model(self.base_config, model, binary_state, b"", ["cpu"])
        self.logger.info("model loaded")

    def _resume(self):

        self.new_server.resume_training()
        # self.binary_state = self.new_server.get_model_state()
        self.logger.info("training resumed")

    def _predict(self):
        """ run prediction on the whole set of patches 
        """
        self.strategy.patched_data.clear()
        self.patch_id = dict()
        x = 0

        for i in self.tile_indices:
            self.slicer[-1] = slice(i[0][1], i[1][1])
            self.slicer[-2] = slice(i[0][0], i[1][0])
            new_slicer = tuple(self.slicer)
            self.patch_id[i[0]] = x  # map each slicer with its corresponding index
            x += 1
            pred_output = self.new_server.forward(NDArray(self.data_file["volume"][new_slicer]))
            pred_output = pred_output.result()
            self.strategy._loss(
                pred_output, self.data_file[self.base_config["labelled_data_path"]][new_slicer], new_slicer
            )

        self.logger.info("prediction run")

    def stop(self):
        """ function which determines when the robot should stop

        currently, it stops after robot has completed 'iterations_max' number of iterations
        """

        if self.iterations_done > self.iterations_max:
            return False
        else:
            self.iterations_done += 1
            return True

    def _run(self):
        """ Feed patches to tiktorch (add to the training data)

        The function fetches the patches in order decided by the strategy, 
        removes it from the list of indices and feeds it to tiktorch 
        """
        while self.stop():
            self._predict()

            # log average loss for all patches per iteration to tensorboard
            total_loss = sum([pair[0] for pair in self.strategy.patched_data])
            avg = total_loss / float(len(self.strategy.patched_data))
            self.tensorboard_writer.add_scalar("avg_loss", avg, self.iterations_done)

            self.strategy.rearrange()
            slicer = self.strategy.get_next_patch()
            self.tile_indices.pop(self.patch_id[(slicer[-2].start, slicer[-1].start)])
            self._add(slicer)
            self._resume()

        self.terminate()

    def _add(self, slicer):
        new_input = NDArray(self.data_file["volume"][slicer].astype(float), (slicer[-2].start, slicer[-1].start))
        new_label = NDArray(
            self.data_file[self.base_config["labelled_data_path"]][slicer].astype(float),
            (slicer[-2].start, slicer[-1].start),
        )
        self.new_server.update_training_data(NDArrayBatch([new_input]), NDArrayBatch([new_label]))

    # annotate worst patch
    def dense_annotate(self, x, y, label, image):
        raise NotImplementedError()

    def terminate(self):
        self.tensorboard_writer.close()
        self.new_server.shutdown()


class BaseStrategy:
    def __init__(self, path_to_config_file):
        with open(path_to_config_file, mode="r") as f:
            self.base_config = yaml.load(f)

        self.patched_data = []
        self.loss_fn = self.base_config["training"]["loss_criterion_config"]["method"]
        self.logger = logging.getLogger(__name__)

    def _loss(self, pred_output, target, slicer):
        """  computes loss corresponding to the output and target according to 
        the given loss function

        Args:
        predicted_output(np.ndarray) : output predicted by the model
        target(np.ndarray): ground truth
        loss_fn(string): loss metric
        slicer(tuple): tuple of slice objects, one per dimension
        """

        criterion_class = getattr(nn, self.loss_fn, None)
        assert criterion_class is not None, "Criterion {} not found.".format(method)
        criterion_class_obj = criterion_class()
        curr_loss = criterion_class_obj(
            torch.from_numpy(pred_output.as_numpy().astype(np.float32)), torch.from_numpy(target.astype(np.float32))
        )
        self.patched_data.append((curr_loss, slicer))

    def get_next_patch(self):
        raise NotImplementedError()

    def rearrange(self):
        raise NotADirectoryError()


class Strategy1(BaseStrategy):
    """ This strategy sorts the patches in descending order of their loss

    Args:
    path_to_config_file (string): path to the configuration file for the robot
    """

    def __init__(self, path_to_config_file):
        super().__init__(path_to_config_file)
        # self.patch_counter = -1

    def rearrange(self):
        """ rearranges the patches in descending order of their loss
        """
        self.patched_data.sort(reverse=True)

    def get_next_patch(self):
        """ Feeds patches to the robot in descending order of their loss
        """

        # self.patch_counter += 1
        return self.patched_data[0][1]


class Strategy2(BaseStrategy):
    def __init__():
        super().__init__()


class Strategy3(BaseStrategy):
    def __init__():
        super().__init__()
