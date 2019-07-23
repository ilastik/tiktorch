import logging
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import z5py
import random

from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter

from tiktorch.server import TikTorchServer
from mr_robot.annotator.annotate import *
from mr_robot.utils import (
    get_confusion_matrix,
    integer_to_onehot,
    plot_confusion_matrix,
    tile_image,
    get_random_patch,
    get_random_index,
)

batch_size = 1


def randomize(label, num_of_classes):
    """ perform a random action on the label
    Action:
    -1: erase label
    0: retain current state (ignore)
    x: update label to x, where x is class number

    Args:
    label (np.ndarray): actual ground truth
    num_of_classes (int): number of classes in the dataset
    """

    actions = [-1, 0] + [i for i in range(1, num_of_classes + 1)]
    volume = np.product(label.shape)
    print(volume, label.shape)
    x = np.random.randint(0, volume)
    while x:
        index = get_random_index(label.shape)
        label[index] = random.choice(actions)
        x -= 1

    return label


def user_simulator(raw_data_file, label_data_file, internal_paths, canvas_shape, num_of_classes):
    """ mimic user annotation process by randomly taking a patch from the dataset and labelling it
    labels can be added, updated or deleted

    Args:
    raw_data_file(file pointer): pointer to folder containing raw data
    lab_data_file(file pointer): pointer to folder containing labelled data
    internal_paths (dictionary): paths inside base folders to raw and labelled data file
    canvas_shape (tuple): shape of canvas
    num_of_classes (int): number of classes in the dataset
    """
    print(canvas_shape)
    timesteps = np.random.randint(5, 10)
    video = []
    for i in range(timesteps):
        random_patch = get_random_patch(canvas_shape)
        image, label = (
            raw_data_file[internal_paths["path_to_raw_data"]][random_patch],
            label_data_file[internal_paths["path_to_labelled"]][random_patch],
        )
        label = randomize(label, num_of_classes)
        video.append((image, label, random_patch))
    return video


class BaseStrategy:
    def __init__(
        self, loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
    ):

        self.patched_data = []
        self.loss_fn = loss_fn
        self.strategy_metric = {"avg_loss": 0, "avg_accuracy": 0, "avg_f1_score": 0, "confusion_matrix": 0}
        self.class_dict = class_dict
        self.raw_data_file = raw_data_file
        self.labelled_data_file = labelled_data_file
        self.paths = paths
        self.annotater = Annotater(annotation_percent)
        self.labelling_strategy = labelling_strategy
        # self.tikserver = tikserver
        self.logger = logging.getLogger(__name__)

    def update_state(self, pred_output, target, block):
        """  computes loss and accuracy corresponding to the output and target according to
        the given loss function and update patch data

        Args:
        predicted_output(np.ndarray) : output predicted by the model
        target(np.ndarray): ground truth
        block(tuple): tuple of slice objects, one per dimension, specifying the corresponding block
        in the actual image
        """

        criterion_class = getattr(nn, self.loss_fn, None)
        assert criterion_class is not None, "Criterion {} not found.".format(method)
        criterion_class_obj = criterion_class(reduction="sum")
        if pred_output.shape != target.shape:
            target = integer_to_onehot(target)
            pred_output = np.expand_dims(pred_output, axis=0)

        curr_loss = criterion_class_obj(
            torch.from_numpy(pred_output.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
        )
        self.patched_data.append((curr_loss, block))
        # print("data added", len(self.patched_data))
        pred_output = pred_output.flatten().round().astype(np.int32)
        target = target.flatten().round().astype(np.int32)

        self.write_metric(pred_output, target, curr_loss)

    def write_metric(self, pred_output, target, curr_loss):
        self.strategy_metric["confusion_matrix"] += get_confusion_matrix(
            pred_output, target, list(self.class_dict.keys())
        )
        self.strategy_metric["avg_accuracy"] += accuracy_score(pred_output, target)
        self.strategy_metric["avg_f1_score"] += f1_score(target, pred_output, average="weighted")
        self.strategy_metric["avg_loss"] += curr_loss
        # print(self.strategy_metric)

    def get_annotated_data(self, return_block_set):
        return_data_set = []
        for block in return_block_set:
            print(self.raw_data_file[self.paths["path_to_raw_data"]])
            image, label = (
                self.raw_data_file[self.paths["path_to_raw_data"]][block],
                self.labelled_data_file[self.paths["path_to_labelled"]][block],
            )
            return_data_set.append(
                (image, getattr(self.annotater, self.labelling_strategy)(label), get_coordinate(block))
            )

        self.patched_data.clear()
        return return_data_set

    def get_metrics(self):
        for key, value in self.strategy_metric.items():
            self.strategy_metric[key] /= len(self.patched_data)
        # print(type(self.strategy_metric), self.strategy_metric)
        import copy

        strategy_metric = copy.deepcopy(self.strategy_metric)

        # FIXME confision_matrix -> plotted_confusion_matrix
        strategy_metric["confusion_matrix"] = plot_confusion_matrix(
            strategy_metric["confusion_matrix"], self.class_dict
        )
        self.strategy_metric = self.strategy_metric.fromkeys(self.strategy_metric, 0)
        return strategy_metric

    def get_next_batch(self):
        raise NotImplementedError()

    def rearrange(self):
        raise NotImplementedError()


class HighestLoss(BaseStrategy):
    """ This strategy sorts the patches in descending order of their loss

    Args:
    loss_fn (string): loss metric to be used
    class_dict (dictionary): dictionary indicating the mapping between classes and their labels
    """

    def __init__(
        self, loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
    ):
        super().__init__(
            loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
        )
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
        return_block_set = [block for loss, block in np.array(self.patched_data)[:batch_size]]
        return super().get_annotated_data(return_block_set)


class StrategyRandom(BaseStrategy):
    """ randomly selected a patch, or batch of patches
    and returns them to the robot

    Args:
    loss_fn (string): loss metric
    class_dict (dictionary): dictionary indicating the mapping between classes and their labels
    """

    def __init__(
        self, loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
    ):
        super().__init__(
            loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
        )

    def rearrange(self):
        pass

    def get_next_batch(self, batch_size=1):
        """ returns a random set of patches

        Args:
        batch_size (int): number of patches to return
        """
        print("get next batch called!!")
        print("dataset size:", len(self.patched_data))
        assert len(self.patched_data) >= batch_size, "batch_size too big for current dataset"

        rand_indices = np.random.randint(0, len(self.patched_data), size=batch_size)
        # print(rand_indices, self.patched_data[rand_indices])
        return_block_set = [block for loss, block in np.array(self.patched_data)[rand_indices]]

        return super().get_annotated_data(return_block_set)


class RandomSparseAnnotate(HighestLoss):
    """ randomly annotate pixels in the labels.
    This emulates a user who randomly annotates pixels evenly spread across the entire image

    Args:
    loss_fn (string): loss metric
    class_dict (dictionanry): dictionary indicating the mapping between classes and their labels
    raw_data_file (h5py/z5py.File): pointer to base folder containing raw images
    labelled_data_file (h5py/z5py.File): pointer to base folder containing labelled images
    paths (dictionary): path inside base folders to raw images and their labels
    """

    def __init__(
        self, loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
    ):
        super().__init__(
            loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
        )

    def update_state(self, pred_output, target, block):
        super().update_state(pred_output, target, block)
        self.block_shape = target.shape

    def get_random_index(self):
        random_index = []
        for i in range(len(self.block_shape)):
            random_index.append(np.random.randint(0, self.block_shape[i]))

        return tuple(random_index)

    def get_next_batch(self, batch_size=1):
        assert len(self.patched_data) >= batch_size, "batch_size too big for current dataset"

        return_block_set = [block for loss, block in np.array(self.patched_data)[:batch_size]]
        return super().get_annotated_data(return_block_set)
        return_data_set = []
        for block in return_block_set:
            image, label = (
                self.raw_data_file[self.paths["path_to_raw_data"]][block],
                self.labelled_data_file[self.paths["path_to_labelled"]][block],
            )
            x = np.random.randint(0, np.product(self.block_shape))
            for i in range(x):
                label[self.get_random_index()] = -1
            return_data_set.append((image, label, get_coordinate(block)))

        return return_data_set


class DenseSparseAnnotate(RandomSparseAnnotate):
    """ sparsely annotate dense patches of labels.
    This emulates a user who randomly annotates small patches sparsely spread across the entire image

    Args:
    loss_fn (string): loss metric
    class_dict (dictionanry): dictionary indicating the mapping between classes and their labels
    raw_data_file (file pointer): pointer to base folder containing raw images
    labelled_data_file (): pointer to base folder containing labelled images
    paths (dictionary): path inside base folders to raw images and their labels
    """

    def __init__(
        self, loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
    ):
        super().__init__(
            loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
        )

    def get_random_patch(self):
        rand_index = super().get_random_index()
        patch_dimension = []
        for i in range(len(self.block_shape)):
            patch_dimension.append(np.random.randint(0, self.block_shape[i] - rand_index[i]))

        block = []
        for i in range(len(patch_dimension)):
            block.append(slice(rand_index[i], rand_index[i] + patch_dimension[i]))
        return tuple(block)

    def get_next_batch(self, batch_size=1):
        assert len(self.patched_data) >= batch_size, "batch_size too big for current dataset"

        return_block_set = [block for loss, block in np.array(self.patched_data)[:batch_size]]
        return_data_set = []
        for block in return_block_set:
            image, label = (
                self.raw_data_file[self.paths["path_to_raw_data"]][block],
                self.labelled_data_file[self.paths["path_to_labelled"]][block],
            )
            x, sparse_label = np.random.randint(0, np.product(self.block_shape)), np.full(label.shape, -1)
            for i in range(x):
                block = get_random_patch()
                sparse_label[block] = label[block]

            return_data_set.append((image, sparse_label, get_coordinate(block)))

        return return_data_set


class ClassWiseLoss(BaseStrategy):
    """ sorts patches according to classes with highest loss, patches with maximum
    instances of this class are fed first
    Assumptions:
    1. model output for multiclass claissfication will always be one hot encoded
    2. class labels are annotated using 1 based indexing

    Args:
    loss_fn (string): loss function to use
    class_dict (dictionary): dictionary indicating the mapping between classes and their labels
    """

    def __init__(
        self, loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
    ):
        super().__init__(
            loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
        )
        self.num_classes = len(self.class_dict)
        self.class_loss = [0] * self.num_classes
        self.image_class_count = np.zeros((1, self.num_classes + 1)).astype(np.int32)
        self.image_counter = 1
        self.image_id = dict()

    def update_state(self, pred_output, target, block):
        """
        1. calculate loss for given prediction and label
        2. map each image with a corresponding ID

        Args:
        pred_output (numpy.ndarray): prediction
        target (numpy.ndarray): actual label
        block (tuple[slice]): tuple of slice objects, one per dimension, specifying the patch in the actual image
        """

        criterion_class = getattr(nn, self.loss_fn, None)
        assert criterion_class is not None, "Criterion {} not found.".format(method)
        criterion_class_obj = criterion_class(reduction="none")

        if len(self.class_dict) > 2:
            one_hot_target = integer_to_onehot(target)
        else:
            one_hot_target = np.expand_dims(target, axis=0)

        pred_output = np.expand_dims(pred_output, axis=0)

        np.vstack([self.image_class_count, np.zeros((1, self.num_classes + 1))])
        self.image_class_count[-1][0] = self.image_counter
        self.image_id[self.image_counter] = block
        self.image_counter += 1

        indices = [0] * (len(target.shape))
        self.record_classes(0, target, indices)

        self.loss_matrix = criterion_class_obj(
            torch.from_numpy(one_hot_target.astype(np.float32)), torch.from_numpy(pred_output.astype(np.float32))
        )

        indices = [0] * (len(self.loss_matrix.shape))
        self.record_class_loss(2, indices, self.loss_matrix.shape)

        curr_total_loss = torch.sum(self.loss_matrix)

        if len(self.class_dict) > 2:
            target = one_hot_target
        super().write_metric(
            pred_output.flatten().round().astype(np.int32), target.flatten().round().astype(np.int32), curr_total_loss
        )

    def record_classes(self, curr_dim, label, indices):
        """ record the number of occurences of each class in a patch

        Args:
        curr_dim (int): current dimension to index
        label (numpy.ndarray): target label
        indices (list): list of variables each representing the current state of index for the n dimension
        """

        if curr_dim + 1 == len(label.shape):
            for i in range(label.shape[curr_dim]):
                indices[curr_dim] = i
                self.image_class_count[-1][label[tuple(indices)] + 1] += 1
            return

        for i in range(label.shape[curr_dim]):
            indices[curr_dim] = i
            self.record_classes(curr_dim + 1, label, indices)

    def record_class_loss(self, curr_dim, indices, output_shape):
        """ record the loss class wise for the given image

        Args:
        curr_dim (int): current dimension to index
        indices (list): list of variables each representing the current state of index for the n dimension
        output_shape (tuple): shape of loss matrix
        """

        if curr_dim + 1 == len(output_shape):
            for i in range(output_shape[curr_dim]):
                indices[curr_dim] = i
                index = indices
                index[1] = slice(0, output_shape[1])
                index = tuple(index)
                # print("record_class_loss", index)
                curr_losses = self.loss_matrix[index].numpy().tolist()
                # print(curr_losses)
                for j in range(len(curr_losses)):
                    self.class_loss[j] += curr_losses[j]

            return

        for i in range(output_shape[curr_dim]):
            indices[curr_dim] = i
            self.record_class_loss(curr_dim + 1, indices, output_shape)

    def rearrange(self):
        """ rearrange the rows of the image_class_count matrix wrt to the class (column) with the highest loss
        """
        self.image_class_count[self.image_class_count[:, np.argmax(self.class_loss) + 1].argsort()[::-1]]

    def get_next_batch(self, batch_size=1):
        self.rearrange()
        return_block_set = [
            self.image_id[image_number]
            for image_number in [image_id for image_id in self.image_class_count[:batch_size, 0]]
        ]
        self.class_loss = [0] * self.num_classes
        self.image_class_count = np.zeros((1, self.num_classes + 1)).astype(np.int32)
        self.image_counter = 1
        self.image_id.clear()
        return super().get_annotated_data(return_block_set)


class VideoLabelling(BaseStrategy):
    """emulates user who randomly annotates/de-annotates/updates various patches at different timestamps
    The strategy expects a series of operations, which are then performed and added to the canvas (sparse matrix)
    The canvas state at each time step is added to the training data
    """

    def __init__(
        self,
        loss_fn,
        class_dict,
        raw_data_file,
        labelled_data_file,
        paths,
        labelling_strategy,
        annotation_percent,
        training_shape,
    ):
        super().__init__(
            loss_fn, class_dict, raw_data_file, labelled_data_file, paths, labelling_strategy, annotation_percent
        )
        dataset_shape = list(self.raw_data_file[self.paths["path_to_raw_data"]].shape)
        dataset_shape[0] = 1
        self.canvas_shape = tuple(dataset_shape)
        self.video = []
        self.training_shape = training_shape
        self.process_video()

    def rearrange(self):
        pass

    def get_next_batch(self, batch_size=None):
        if not self.video:
            raise ValueError("no more annotations in the video available!")
        return self.video.pop(0)

    def update_canvas(self, label, block, curr_dim, index_list):
        """ update the canvas from the received labels

        UPDATE RULE:
        if label is:
        -1: erase current label from canvas
        0: retain previous label on canvas
        x: update label on canvas, where x is new class label

        Args:
        label (np.ndarray): label received from user
        block (tuple): tuple specifying the block in the canvas
        curr_dim (int): current dimension in the recursion step
        index_list (list): list of indices specifying the current index of each dimension
        """

        if curr_dim + 1 == len(label.shape):
            for i in range(label.shape[curr_dim]):
                index_list[curr_dim] = i
                index = tuple(index_list)
                if label[index] == -1:
                    self.canvas[block][index] = 0
                elif label[index] != 0:
                    self.canvas[block][index] = label[index]

            return

        for i in range(label.shape[curr_dim]):
            index_list[curr_dim] = i
            self.update_canvas(label, block, curr_dim + 1, index_list)

    def paint(self, label, block):
        """ paints the newly received label onto the canvas
        """

        index_list = [0] * len(label.shape)
        self.update_canvas(label, block, 0, index_list)

    def process_video(self):
        """ take the series of annotations performed by the user and resize them to trainable shape
        """

        self.canvas = np.zeros((self.canvas_shape))
        user_annotations = user_simulator(
            self.raw_data_file, self.labelled_data_file, self.paths, self.canvas_shape, len(self.class_dict)
        )

        for image, label, block in user_annotations:
            self.paint(label, block)
            base_coordinate = get_coordinate(block)

            # resize image and label by zero padding if image size is less than training shape
            image_shape = tuple([max(image.shape[i], self.training_shape[i]) for i in range(len(image.shape))])
            image.resize(image_shape, refcheck=False)
            label.resize(image_shape, refcheck=False)
            canvas_tiles = tile_image(image.shape, self.training_shape)

            # iterate over the tiled image and add the data to the list 'video'.
            # Each timestep is added as a list
            curr_timestep = []
            for tile in canvas_tiles:
                local_id = get_coordinate(tile)
                global_id = tuple([base_coordinate[i] + local_id[i] for i in range(len(base_coordinate))])
                curr_timestep.append((image[tile], label[tile], global_id))

            self.video.append(curr_timestep)


class StrategyAbstract:
    """ abstract strategy which is a combination of one or more basic strategies

    Args:
    *args [(Any, iterations)]: list of strategy objects to applied in given order, for {iterations} number of times
    """

    def __init__(self, tikserver, *args):
        self.strategies = args
        self.num_iterations = 0
        self.index = 0
        self.tikserver = tikserver
        self.tiktorch_config = {"training": {"num_iterations_done": 0}}

    def update_strategy(self):
        if len(self.strategies) > self.index + 1:
            self.index += 1
        # self.num_iterations += 1
        self.tiktorch_config["training"]["num_iterations_done"] = self.strategies[self.index][1]
        self.tikserver.update_config(self.tiktorch_config)
        print("curr strategy:", str(self.strategies[self.index][0]))

    def update_state(self, pred_output, target, loss):
        self.strategies[self.index][0].update_state(pred_output, target, loss)

    def rearrange(self):
        self.strategies[self.index][0].rearrange()

    def get_next_batch(self, batch_size=1):
        new_batch = self.strategies[self.index][0].get_next_batch(batch_size)
        self.update_strategy()
        return new_batch

    def get_metrics(self):
        return self.strategies[self.index][0].get_metrics()
