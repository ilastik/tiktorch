# import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.metrics import mean_squared_error
from model import DUNet2D
import h5py
from scipy.ndimage import convolve
from torch.autograd import Variable
from collections import OrderedDict
import yaml
from tiktorch.server import TikTorchServer
from tiktorch.rpc import Client, Server, InprocConnConf
from tiktorch.rpc_interface import INeuralNetworkAPI
from tiktorch.types import NDArray, NDArrayBatch
from utils import *

patch_size = 16


class MrRobot:
    def __init__(self):
        # start the server
        self.new_server = TikTorchServer()

    def load_data(self):
        with h5py.File("train.hdf", "r") as f:
            x = np.array(f.get("volumes/labels/neuron_ids"))
            y = np.array(f.get("volumes/raw"))

        self.labels = []
        self.ip = []

        for i in range(0, 1):
            self.labels.append(make_edges3d(np.expand_dims(x[i], axis=0)))
            self.ip.append(make_edges3d(np.expand_dims(y[i], axis=0)))

        self.labels = np.asarray(self.labels)[:, :, 0:patch_size, 0:patch_size]
        self.ip = NDArray(np.asarray(self.ip)[:, :, 0:patch_size, 0:patch_size])
        print("data loaded")
        return (ip, labels)

    def load_model(self):
        # load the model
        with open("state.nn", mode="rb") as f:
            binary_state = f.read()
        with open("model.py", mode="rb") as f:
            model_file = f.read()

        with open("robo_config.yml", mode="r") as f:
            base_config = yaml.load(f)

        fut = self.new_server.load_model(base_config, model_file, binary_state, b"", ["cpu"])
        print("model loaded")
        # print(fut.result())

    def resume(self):
        self.new_server.resume_training()
        print("training resumed")

    def predict(self):
        self.op = new_server.forward(self.ip)
        self.op = op.result().as_numpy()
        print("prediction run")
        return (self.op, self.labels)

    def add(self, row, column):
        self.ip = self.ip.as_numpy()[
            0, :, patch_size * row : patch_size * (row + 1), patch_size * column : patch_size * (column + 1)
        ].astype(float)
        self.label = self.labels[
            0, :, patch_size * row : patch_size * (row + 1), patch_size * column : patch_size * (column + 1)
        ].astype(float)
        # print(ip.dtype, label.dtype)
        self.new_server.update_training_data(NDArrayBatch([NDArray(self.ip)]), NDArrayBatch([self.label]))

    # annotate worst patch
    def dense_annotate(self, x, y, label, image):
        raise NotImplementedError

    def terminate():
        new_server.shutdown()


class BaseStrategy:
    def __init__():
        raise NotImplementedError

    # compute loss for a given patch
    def base_loss(self, patch, label):
        result = mean_squared_error(label, patch)  # CHECK THIS
        return result


class Strategy1(BaseStrategy):
    def __init__(self, op, labels):
        super().__init__()
        pred_idx = tile_image2D(op[0, 0].shape, 16)
        actual_idx = tile_image2D(labels[0, 0].shape, 16)
        w, h, self.row, self.column = 32, 32, -1, -1
        error = 1e7
        for i in range(len(pred_patches)):
            # print(pred_patches[i].shape, actual_patches[i].shape)
            curr_loss = self.loss(
                op[0, 0, pred_idx[i][0] : pred_idx[i][1], pred_idx[i][2] : pred_idx[i][3]],
                labels[0, 0, actual_idx[i][0] : actual_idx[i][1], actual_idx[i][2] : actual_idx[i][3]],
            )
            print(curr_loss)
            if error > curr_loss:
                error = curr_loss
                self.row, self.column = int(i / (w / patch_size)), int(i % (w / patch_size))

    def get_patch(self):
        return (self.row, self.column)


class Strategy2(BaseStrategy):
    def __init__():
        super().__init__()
        raise NotImplementedError


class Strategy3(BaseStrategy):
    def __init__():
        super().__init__()
        raise NotImplementedError


if __name__ == "__main__":

    robo = MrRobot()
    robo.load_data()
    robo.load_model()
    robo.resume()  # resume training

    # run prediction
    op, label = robo.predict()

    metric = Strategy1(op, label)
    row, column = metric.get_patch()
    robo.add(row, column)

    # shut down server
    robo.terminate()
