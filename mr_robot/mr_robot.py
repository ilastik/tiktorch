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
from typing import List


from scipy import sparse
from io import BytesIO
from mr_robot.utils import (
    get_confusion_matrix,
    integer_to_onehot,
    plot_confusion_matrix,
    tile_image,
    get_coordinate,
    make_plot,
)
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter

from tiktorch.models.dunet import DUNet
from tiktorch.rpc.utils import BatchedExecutor
from tiktorch.server.base import TikTorchServer
from tiktorch.types import NDArray, NDArrayBatch, Model, ModelState
from mr_robot.strategies.strategy import *
from mr_robot.utils import get_confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score

img_dim = 32
batch_size = 48


class MrRobot:
    """ The robot class runs predictins on the model, and feeds the worst performing patch back for training.
    The order in which patches are feed back is determined by the 'strategy'. The robot applies a given strategy,
    adds new patches to the training data and logs the metrics to tensorboard

    Args:
    path_to_config_file (string): path to the robot configuration file to
                                  load necessary variables
    strategy (string): strategy to follow (atleast intially)
    """

    def __init__(self, path_to_config_file, strategy, devices: List[str]) -> None:

        assert torch.cuda.device_count() == 1, f"Device count is {torch.cuda.device_count()}"
        # start the server
        self.new_server = TikTorchServer()
        self._devices = devices

        with open(path_to_config_file, mode="r") as f:
            self.base_config = yaml.load(f)

        if self.base_config["data_dir"]["raw_data_base_folder"].endswith(".h5"):
            self.raw_data_file = h5py.File(self.base_config["data_dir"]["raw_data_base_folder"], 'r')
            self.labelled_data_file = h5py.File(self.base_config["data_dir"]["labelled_data_base_folder"], 'r')
            self.validation_raw_file = h5py.File(self.base_config["data_dir"]["validation_raw_base"], 'r')
            self.validation_label_file = h5py.File(self.base_config["data_dir"]["validation_label_base"], 'r')
        else:
            self.raw_data_file = z5py.File(self.base_config["data_dir"]["raw_data_base_folder"], 'r')
            self.labelled_data_file = z5py.File(self.base_config["data_dir"]["labelled_data_base_folder"], 'r')
            self.validation_raw_file = h5py.File(self.base_config["data_dir"]["validation_raw_base"], 'r')
            self.validation_label_file = h5py.File(self.base_config["data_dir"]["validation_label_base"], 'r')

        image_shape = self.raw_data_file[self.base_config["data_dir"]["path_to_raw_data"]].shape
        validation_data_shape = self.validation_raw_file[self.base_config["data_dir"]["validation_raw"]].shape
        print(image_shape)
        self.block_list = tile_image(image_shape, self.base_config["training"]["training_shape"])
        self.validation_block_list = tile_image(validation_data_shape, self.base_config["training"]["training_shape"])
        print("number of patches: %s" % len(self.block_list))
        print()

        # strategy_class = strategies[strategy]
        # self.strategy = strategy_class(
        #    self.base_config["training"]["loss_criterion_config"]["method"], self.base_config["class_dict"], self.raw_data_file, self.labelled_data_file, self.base_config["data_dir"], annotation_strat,
        # )
        strat0 = VideoLabelling(
            "BCELoss",
            self.base_config["class_dict"],
            self.raw_data_file,
            self.labelled_data_file,
            self.base_config["data_dir"],
            "dense",
            0.6,
            (1,256,256)
        )
        strat1 = ClassWiseLoss(
            "BCELoss",
            self.base_config["class_dict"],
            self.raw_data_file,
            self.labelled_data_file,
            self.base_config["data_dir"],
            "dense",
            0.6
        )

        strat2 = HighestLoss(
            "BCELoss",
            self.base_config["class_dict"],
            self.raw_data_file,
            self.labelled_data_file,
            self.base_config["data_dir"],
            "dense",
            0.6,
        )
        self.strategy = StrategyAbstract(self.new_server, (strat1, batch_size * 2))

        self.iterations_max = self.base_config.pop("max_robo_iterations")
        self.iterations_done = 0
        self.stats = {
            "training_loss": [],
            "training_accuracy": [],
            "robo_predict_accuracy": [],
            "f1_score": [],
            "robo_predict_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "training_iterations": [],
            "number_of_patches": [],
            "training_confusion" : 0,

        }
        mr_robot_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tensorboard_writer = SummaryWriter(logdir=os.path.join(mr_robot_folder, "tests", "robot", "robo_logs", "strat_classwise"))
        self.patch_id = dict()
        self.logger = logging.getLogger(__name__)

    def _load_model(self):

        if self.base_config["model_dir"]["path_to_folder"].endswith(".zip"):
            archive = zipfile.ZipFile(self.base_config["model_dir"]["path_to_folder"], "r")
            model_file = archive.read(self.base_config["model_dir"]["path_in_folder_to_model"])
            binary_state = archive.read(self.base_config["model_dir"]["path_in_folder_to_state"])

        else:
            model_file = open(
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

        model = Model(code=model_file, config=self.base_config)
        binary_state = ModelState(b"")
        # binary_state.model_state = b""

        self.new_server.load_model(model, binary_state, self._devices)
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
        
        x = 0
        prediction_list = []
        path_to_input = self.base_config["data_dir"]["path_to_raw_data"]
        path_to_label = self.base_config["data_dir"]["path_to_labelled"]

        batch_maker = BatchedExecutor(batch_size=8)
        for block in self.block_list:
            # map each slicer with its corresponding index
            signal = self.assign_id(block, x)
            print("signal", signal)
            if( signal == -1):
                continue
            # self.patch_id[block[0].start] = x
            # pred_output = self.new_server.forward(NDArray(self.raw_data_file[path_to_input][block]))
            prediction_list.append(
                batch_maker.submit(self.new_server.forward, NDArray(self.raw_data_file[path_to_input][block], x))
            )
            x += 1
            # self.pred_output = pred_output.result().as_numpy()
            # print("hello")
            # self.strategy.update_state(self.pred_output, self.labelled_data_file[path_to_label][block], block)

        for prediction in cf.as_completed(prediction_list):
            block = self.block_list[prediction.result().id]
            self.strategy.update_state(
                prediction.result().as_numpy(), self.labelled_data_file[path_to_label][block], block, False
            )

        # self.logger.info("prediction run for iteration {}", self.iterations_done)

    def validate(self):
        validation_list = []
        x = 0  # CHECK: same id as prediction, does that work??
        path_to_input = self.base_config["data_dir"]["validation_raw"]
        path_to_label = self.base_config["data_dir"]["validation_label"]

        batch_maker = BatchedExecutor(batch_size=8)
        for block in self.validation_block_list:
            validation_list.append(
                batch_maker.submit(self.new_server.forward, NDArray(self.validation_raw_file[path_to_input][block], x))
            )
            x += 1
        for prediction in cf.as_completed(validation_list):
            block = self.validation_block_list[prediction.result().id]
            self.strategy.update_state(
                prediction.result().as_numpy(), self.validation_label_file[path_to_label][block], block, True
            )

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
            self.validate()
            curr_model_state = self.new_server.get_model_state()
            self.stats["training_iterations"].append(curr_model_state.num_iterations_done)
            print("before: ", self.stats["training_loss"], self.stats["number_of_patches"])
            #self.stats["training_loss"].append(curr_model_state.loss)
            self.stats["number_of_patches"].append( (self.iterations_done-1) * batch_size)
            print("after: ", self.stats["training_loss"], self.stats["number_of_patches"])
            self.write_to_tensorboard()

            data_batch = self.strategy.get_next_batch(batch_size)

            self._add(data_batch)
            #self.remove_key([id for image, label, id in data_batch])
            print("waiting for training")
            self.new_server.train_for(10).result()
            print("training done!!")

        self.terminate()

    def _add(self, new_data_batch):
        """ add a new batch of images to training data

        Args:
        new_data_batch (list): list of tuples, where each tuple contains an image, its label and their block id
        """
        assert new_data_batch is not None, "No data provided!"
        print("adding new data batch:", len(new_data_batch))

        new_inputs, new_labels = [], []
        for image, label, block_id in new_data_batch:
            print("values in label:", np.unique(label), label.shape)
            print("values in image:", np.unique(image), image.shape)
            new_inputs.append(NDArray(image.astype(np.int32), block_id))
            new_labels.append(NDArray(label.astype(np.int32), block_id))
            print("label", block_id, label.any())
        self.new_server.update_training_data(NDArrayBatch(new_inputs), NDArrayBatch(new_labels))
        print("addition done!!")

    def write_to_tensorboard(self):
        metric_data = self.strategy.get_metrics()

        for key, value in metric_data.items():
            if "confusion_matrix" not in key and key != "robo_predict_loss":
                self.stats[key].append(value)
            if key == "robo_predict_loss":
                print(value.item())
                self.stats[key].append(value.item())

        if(self.iterations_done == 1):
            self.stats["training_accuracy"].append(0)
            self.stats["training_loss"].append(0)
            training_confusion_matrix = np.zeros((2,2))

        else:
            path_to_input = self.base_config["data_dir"]["path_to_raw_data"]
            path_to_label = self.base_config["data_dir"]["path_to_labelled"]

            training_block = []
            for key,value in self.patch_id.items():
                if value == -1:
                    training_shape = self.base_config["training"]["training_shape"]
                    block = tuple([slice(key[i],key[i]+training_shape[i]) for i in range(len(training_shape))])
                    print(block)
                    training_block.append(block)

            x=0
            print("training data size:", len(training_block))
            train_prediction_list = []
            batch_maker = BatchedExecutor(batch_size=8)
            for block in training_block:
                image = self.raw_data_file[path_to_input][block]
                print("training accuracy calc:", image.shape)
                train_prediction_list.append(
                    batch_maker.submit(self.new_server.forward, NDArray(image, x))
                )
                x += 1
            print()
            train_accuracy,conf_matrix,train_loss = 0.0,0,0
            for prediction in cf.as_completed(train_prediction_list):
                block = training_block[prediction.result().id]
                label = self.labelled_data_file[path_to_label][block]
                label[label==2] = 0

                pred_output = prediction.result().as_numpy()

                criterion_class = getattr(nn, "BCELoss", None)
                criterion_class_obj = criterion_class(reduction="mean")
                train_loss += criterion_class_obj(
                torch.from_numpy(pred_output.astype(np.float32)), torch.from_numpy(label.astype(np.float32))
                )   

                pred_output = pred_output.flatten().round().astype(np.int32)
                target = label.flatten().round().astype(np.int32)

                train_accuracy+= accuracy_score(target,pred_output)
                conf_matrix += get_confusion_matrix(pred_output, target, list(self.base_config["class_dict"].keys()))
                
            train_accuracy /= ((self.iterations_done-1) * batch_size)
            train_loss /= ((self.iterations_done-1) * batch_size)
            training_confusion_matrix = conf_matrix/((self.iterations_done-1) * batch_size)
            self.stats["training_accuracy"].append(train_accuracy)
            self.stats["training_loss"].append(train_loss)

        training_confusion_matrix =  plot_confusion_matrix( training_confusion_matrix, self.base_config["class_dict"])
        loss_plot, accuracy_plot = make_plot(self.stats)
        self.tensorboard_writer.add_figure("loss_plot", loss_plot)
        self.tensorboard_writer.add_figure("accuracy_plot", accuracy_plot)
        # self.tensorboard_writer.add_scalar("avg_loss", metric_data["avg_loss"], self.iterations_done)
        # self.tensorboard_writer.add_scalar("avg_accuracy", metric_data["avg_accuracy"] * 100, self.iterations_done)
        # self.tensorboard_writer.add_scalar("F1_score", metric_data["avg_f1_score"], self.iterations_done)
        self.tensorboard_writer.add_figure(
            "robo_confusion_matrix", metric_data["robo_confusion_matrix"], global_step=self.iterations_done
        )
        self.tensorboard_writer.add_figure(
            "validation_confusion_matrix", metric_data["validation_confusion_matrix"], global_step=self.iterations_done
        )
        self.tensorboard_writer.add_figure(
            "training_confusion_matrix", training_confusion_matrix, global_step=self.iterations_done
        )
    def assign_id(self, block, index):
        id = get_coordinate(block)
        if ( id in self.patch_id and self.patch_id[id] == -1):
            return -1
        
        self.patch_id[id] = index
        return 0

    def remove_key(self, ids):
        for id in ids:
            self.patch_id[id] = -1

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
