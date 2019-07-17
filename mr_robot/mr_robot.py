import logging
import zipfile

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
from mr_robot.utils import get_confusion_matrix, integer_to_onehot, plot_confusion_matrix, tile_image
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter

from tiktorch.models.dunet import DUNet
from tiktorch.rpc.utils import BatchedExecutor
from tiktorch.server import TikTorchServer
from tiktorch.types import NDArray, NDArrayBatch

img_dim = 32
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

    timesteps = np.random.randint(0, 100)
    video = []
    for i in range(timestep):
        random_patch = get_random_patch(canvas_shape)
        image, label = (
            raw_data_file[internal_paths["path_to_raw_data"]][random_patch],
            label_data_file[internal_paths["path_to_labelled"]][random_patch],
        )
        label = randomize(label, num_of_classes)
        video.append((image, label, random_patch))
    return video


def get_random_index(canvas_shape):
    random_index = []
    for i in range(len(canvas_shape)):
        random_index.append(np.random.randint(0, canvas_shape[i]))

    return tuple(random_index)


def get_random_patch(canvas_shape):
    rand_index = get_random_index(canvas_shape)
    patch_dimension = []
    for i in range(len(canvas_shape)):
        patch_dimension.append(np.random.randint(0, canvas_shape[i] - rand_index[i]))

    block = []
    for i in range(len(patch_dimension)):
        block.append(slice(rand_index[i], rand_index[i] + patch_dimension[i]))
    return tuple(block)


def get_coordinate(block):
    """ return the starting co-ordinate of a block

    Args:
    block(tuple): tuple of slice objects, one per dimension
    """

    coordinate = []
    for slice_ in block:
        coordinate.append(slice_.start)

    return tuple(coordinate)


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
        self.strategy = strategy_class(
            self.base_config["training"]["loss_criterion_config"]["method"], self.base_config["class_dict"]
        )

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
        self.base_config.pop("model_dir")

        self.new_server.load_model(self.base_config, model, binary_state, b"", [os.environ["CUDA_VISIBLE_DEVICES"]])
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
        for block in self.block_list:
            # map each slicer with its corresponding index
            self.assign_id(block, x)
            # self.patch_id[block[0].start] = x
            x += 1
            # pred_output = self.new_server.forward(NDArray(self.raw_data_file[path_to_input][block]))
            prediction_list.append(
                batch_maker.submit(self.new_server.forward, NDArray(self.raw_data_file[path_to_input][block]))
            )
            # self.pred_output = pred_output.result().as_numpy()
            print("hello")
            # self.strategy.update_state(self.pred_output, self.labelled_data_file[path_to_label][block], block)

        # print("prediction run for robo iteration %s" % self.iterations_done)
        # print()

        i = 0
        for prediction in cf.as_completed(prediction_list):
            # print(i, type(prediction))
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

            block_batch = self.strategy.get_next_batch(batch_size)

            if len(block_batch[0]) == 3:
                self._add(None, block_batch)
                self.remove_key(None, block_batch)

            else:
                self._add(block_batch)
                self.remove_key(block_batch)

            self._resume()

        self.terminate()

    def _add(self, new_block_batch=None, new_data_batch=None):
        """ add a new batch of images to training data

        Args:
        new_block_batch (list): list of tuples, where each tuple is 
        a slicer corresponding to a block
        """
        assert new_block_batch is not None or new_data_batch is not None, "No data provided!"

        new_inputs, new_labels = [], []
        if new_data_batch is not None:
            for image, label, block_id in new_data_batch:
                new_inputs.append(NDArray(image.astype(np.float), block_id))
                new_labels.append(NDArray(label.astype(np.float), block_id))

        else:
            for i in range(len(new_block_batch)):

                new_block = self.raw_data_file[self.base_config["data_dir"]["path_to_raw_data"]][new_block_batch[i]]
                new_label = self.labelled_data_file[self.base_config["data_dir"]["path_to_labelled"]][
                    new_block_batch[i]
                ]

                new_inputs.append(NDArray(new_block.astype(float), get_coordinate(new_block_batch[i])))
                new_labels.append(NDArray(new_label.astype(float), get_coordinate(new_block_batch[i])))

        self.new_server.update_training_data(NDArrayBatch(new_inputs), NDArrayBatch(new_labels))

    def write_to_tensorboard(self):
        metric_data = self.strategy.get_metrics()
        print("average loss: %s   average accuracy: %s" % (metric_data["avg_loss"], metric_data["avg_accuracy"] * 100))
        print()
        self.tensorboard_writer.add_scalar("avg_loss", metric_data["avg_loss"], self.iterations_done)
        self.tensorboard_writer.add_scalar("avg_accuracy", metric_data["avg_accuracy"] * 100, self.iterations_done)
        self.tensorboard_writer.add_scalar("F1_score", metric_data["avg_f1_score"], self.iterations_done)
        self.tensorboard_writer.add_figure(
            "confusion_matrix", metric_data["confusion_matrix"], global_step=self.iterations_done
        )

    def assign_id(self, block, index):
        coordinate = get_coordinate(block)
        self.patch_id[coordinate] = index

    def remove_key(self, block_batch=None, data_batch=None):

        if data_batch is not None:
            for image, label, block_id in data_batch:
                self.patch_id.pop(block_id)
        else:
            for block in block_batch:
                coordinate = get_coordinate(block)
                self.patch_id.pop(coordinate)

    # annotate worst patch
    def dense_annotate(self, x, y, label, image):
        raise NotImplementedError()

    def terminate(self):
        self.tensorboard_writer.close()
        self.new_server.shutdown()


class BaseStrategy:
    def __init__(self, loss_fn, class_dict):

        self.patched_data = []
        self.loss_fn = loss_fn
        self.strategy_metric = {"avg_loss": 0, "avg_accuracy": 0, "avg_f1_score": 0, "confusion_matrix": 0}
        self.class_dict = class_dict
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
        curr_loss = criterion_class_obj(
            torch.from_numpy(pred_output.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
        )
        self.patched_data.append((curr_loss, block))

        pred_output = pred_output.flatten().round().astype(np.int32)
        target = target.flatten().round().astype(np.int32)

        self.write_metric(pred_output, target, curr_loss)

    def write_metric(self, pred_output, target, curr_loss):
        self.strategy_metric["confusion_matrix"] += get_confusion_matrix(pred_output, target, self.class_dict)
        self.strategy_metric["avg_accuracy"] += accuracy_score(pred_output, target)
        self.strategy_metric["avg_f1_score"] += f1_score(target, pred_output, average="weighted")
        self.strategy_metric["avg_loss"] += curr_loss

    def get_next_batch(self):
        raise NotImplementedError()

    def rearrange(self):
        raise NotImplementedError()

    def get_metrics(self):
        for key, value in self.strategy_metric.iteritems():
            self.strategy_metric[key] /= len(self.patched_data)

        strategy_metric = self.strategy_metric
        strategy_metric["confusion_matrix"] = plot_confusion_matrix(
            strategy_metric["confusion_matrix"], self.class_dict
        )
        self.strategy_metric = self.strategy_metric.fromkeys(self.strategy_metric, 0)
        return strategy_metric


class HighestLoss(BaseStrategy):
    """ This strategy sorts the patches in descending order of their loss

    Args:
    loss_fn (string): loss metric to be used
    class_dict (dictionary): dictionary indicating the mapping between classes and their labels
    """

    def __init__(self, loss_fn, class_dict):
        super().__init__(loss_fn, class_dict)
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


class StrategyRandom(BaseStrategy):
    """ randomly selected a patch, or batch of patches
    and returns them to the robot

    Args:
    loss_fn (string): loss metric
    class_dict (dictionary): dictionary indicating the mapping between classes and their labels
    """

    def __init__(self, loss_fn, class_dict):
        super().__init__(loss_fn, class_dict)

    def rearrange(self):
        pass

    def get_next_batch(self, batch_size=1):
        """ returns a random set of patches

        Args:
        batch_size (int): number of patches to return
        """

        assert len(self.patched_data) >= batch_size, "batch_size too big for current dataset"

        rand_indices = np.random.randint(len(self.patched_data), size=batch_size)
        return_patch_set = [block for loss, block in self.patched_data[rand_indices]]
        self.patched_data.clear()
        return return_patch_set


class RandomSparseAnnotate(Strategy1):
    """ randomly annotate pixels in the labels.
    This emulates a user who randomly annotates pixels evenly spread across the entire image

    Args:
    loss_fn (string): loss metric
    class_dict (dictionanry): dictionary indicating the mapping between classes and their labels
    raw_data_file (h5py/z5py.File): pointer to base folder containing raw images
    labelled_data_file (h5py/z5py.File): pointer to base folder containing labelled images
    paths (dictionary): path inside base folders to raw images and their labels
    """

    def __init__(self, loss_fn, class_dict, raw_data_file, labelled_data_file, paths):
        super().__init__(loss_fn, class_dict)
        self.raw_data_file = raw_data_file
        self.labelled_data_file = labelled_data_file
        self.paths = paths

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

        return_block_set = [block for loss, block in self.patched_data[:batch_size]]
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

    def __init__(self, loss_fn, class_dict, raw_data_file, labelled_data_file, paths):
        super().__init__(loss_fn, class_dict, raw_data_file, labelled_data_file, paths)

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

        return_block_set = [block for loss, block in self.patched_data[:batch_size]]
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

    def __init__(self, loss_fn, class_dict):
        super().__init__(loss_fn, class_dict)
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
        criterion_class_obj = criterion_class(reduction=None)

        # pred_output = pred_output.flatten().round().astype(np.int32)
        # target = target.flatten().round().astype(np.int32)
        if len(self.class_dict) > 2:
            one_hot_target = np.expand_dims(integer_to_onehot(target), axis=0)
        else:
            one_hot_target = np.expand_dims(target, axis=0)

        pred_output = np.expand_dims(pred_output, axis=0)

        np.vstack([self.image_class_count, np.zeros((1, self.num_classes + 1))])
        self.image_class_count[-1][0] = self.image_counter
        self.image_id[self.image_counter] = block
        self.image_counter += 1
        indices = [0] * (len(self.pred_output.shape))
        self.record_classes(0, target)
        self.loss_matrix = criterion_class_obj(torch.from_numpy(one_hot_target), torch.from_numpy(pred_output))
        record_class_loss(2, indices, len(self.loss_matrix.shape))
        curr_total_loss = torch.sum(self.loss_matrix)
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
                self.image_class_count[-1][label[tuple(indices)]] += 1
            return

        for i in range(label.shape[curr_dim]):
            indices[curr_dim] = i
            record_classes(curr_dim + 1, label, indices)

    def record_class_loss(self, curr_dim, indices, n):
        """ record the loss class wise for the given image

        Args:
        curr_dim (int): current dimension to index
        indices (list): list of variables each representing the current state of index for the n dimension
        n (int): number of dimensions in the loss matrix
        """

        if curr_dim + 1 == n:
            for i in range(self.pred_output.shape[curr_dim]):
                indices[curr_dim] = i
                index = indices
                index[1] = slice(self.image_class_count)
                index = tuple(index)
                curr_losses = self.loss_matrix[index]
                for i in len(curr_losses):
                    self.class_loss[i] += curr_losses[i]

            return

        for i in range(self.pred_output.shape[curr_dim]):
            indices[curr_dim] = i
            record_class_loss(curr_dim + 1, indices, n)

    def rearrange(self):
        """ rearrange the rows of the image_class_count matrix wrt to the class (column) with the highest loss
        """
        self.image_class_count[self.image_class_count[:, np.argmax(self.class_loss)].argsort()[::-1]]

    def get_next_batch(self, batch_size=1):
        return_block_set = [
            self.image_id[image_number]
            for image_number in [image_number for image_number in self.image_class_count[:batch_size, 0]]
        ]
        self.class_loss = [0] * self.num_classes
        self.image_class_count = np.zeros((1, self.num_classes + 1)).astype(np.int32)
        self.image_counter = 1
        self.image_id.clear()
        return return_block_set


class VideoLabelling(BaseStrategy):
    """emulates user who randomly annotates/de-annotates/updates various patches at different timestamps
    The strategy expects a series of operations, which are then performed and added to the canvas (sparse matrix)
    The canvas state at each time step is added to the training data 
    """

    def __init__(self, loss_fn, class_dict, raw_data_file, labelled_data_file, paths, training_shape):
        super().__init__(loss_fn, class_dict)
        self.raw_data_file = raw_data_file
        self.labelled_data_file = labelled_data_file
        self.paths = paths
        self.canvas_shape = self.raw_data_file[self.paths["path_to_raw_data"]].shape
        self.video = []
        self.training_shape = training_shape

    def rearrange(self):
        pass

    def get_next_batch(self):
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
            for i in range(len(label.shape[curr_dim])):
                index_list[curr_dim] = i
                index = tuple(index_list)
                if label[index] == -1:
                    self.canvas[block][index] = 0
                elif label[index] != 0:
                    self.canvas[block][index] = label[index]

            return

        for i in range(len(label.shape[curr_dim])):
            index_list[curr_dim] = i
            self.update_canvas(label, block, curr_dim + 1, index_list)

    def paint(self, label, block):
        """ paints the newly received label onto the canvas
        """

        index_list = [0] * len(label.shape)
        self.update_canvas(label, block, curr_dim, index_list)

    def process_video(self):
        """ take the series of annotations performed by the user and resize them to trainable shape
        """

        self.canvas = np.zeros((self.canvas_shape))
        user_annotations = user_simulator(self.labelled_data_file, self.paths, self.canvas_shape)

        for image, label, block in user_annotations:
            self.paint(label, block)
            base_coordinate = get_coordinate(block)

            # resize image and label by zero padding if image size is less than training shape
            image_shape = tuple([max(image.shape[i], self.training_shape[i]) for i in range(len(image.shape))])
            image.resize(image_shape)
            label.resize(image_shape)
            canvas_tiles = tile_image(image.shape, self.training_shape)

            # iterate over the tiled image and add the data to the list 'video'.
            # Each timestep is added as a list
            curr_timestep = []
            for tile in canvas_tiles:
                local_id = get_coordinate(tile)
                global_id = tuple([base_coordinate[i] + local_id[i] for i in range(len(base_coordinate))])
                curr_timestep.append((image[tile], label[tile], global_id))

            self.video.append(curr_timestep)


strategies = {
    "highestloss": HighestLoss,
    "strategyrandom": StrategyRandom,
    "randomsparseannotate": RandomSparseAnnotate,
    "densesparseannotate": DenseSparseAnnotate,
    "classwiseloss": ClassWiseLoss,
    "videolabelling": VideoLabelling,
}
