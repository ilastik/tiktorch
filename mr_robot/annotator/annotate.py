import numpy as np
import random

from mr_robot.utils import get_coordinate


class Annotater:
    """ The annotater class prvides methods for different labelling strategies, emulating a user
    in some way

    Args:
    annotation_percent (int): percentage of pixels in the patch to annotate
    """

    def __init__(self, annotation_percent):
        self.annotation_percent = annotation_percent
        # self.block_shape = block_shape

    def get_random_index(self, block_shape):
        random_index = []
        for i in range(len(block_shape)):
            random_index.append(np.random.randint(0, block_shape[i]))

        return tuple(random_index)

    def get_random_patch(self, block_shape):
        rand_index = self.get_random_index(block_shape)
        patch_dimension = []
        for i in range(len(block_shape)):
            patch_dimension.append(np.random.randint(0, block_shape[i] - rand_index[i]))

        block = []
        for i in range(len(patch_dimension)):
            block.append(slice(rand_index[i], rand_index[i] + patch_dimension[i]))
        return tuple(block)

    def dense(self, label):
        return label

    def random_sparse(self, label):
        ret_label = np.zeros(label.shape)
        for i in range(int(self.annotation_percent) * np.product(label.shape)):
            index = self.get_random_index(label.shape)
            ret_label[index] = label[index]
        return ret_label

    def random_blob(self, label):
        ret_label = np.ones(label.shape)
        for i in range(int(self.annotation_percent) * np.product(label.shape)):
            random_block = self.get_random_patch(label.shape)
            ret_label[random_block] = label[random_block]
            i += np.product(label[random_block].shape)
        print("random blob annotated label:", np.unique(ret_label), "labels of actual return patch:", np.unique(label))
        return ret_label
