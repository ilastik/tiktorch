# import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.metrics import mean_squared_error
#from model import DUNet2D
import h5py
import z5py
from z5py.converter import convert_from_h5 
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
img_dim = 32

class MrRobot:
    def __init__(self):
        # start the server
        self.new_server = TikTorchServer()


    def load_data(self):
        self.f = z5py.File('train.n5')
        return self.f
        """
        #with h5py.File("train.hdf", "r") as f:
        #    x = np.array(f.get("volumes/labels/neuron_ids"))
        #    y = np.array(f.get("volumes/raw"))

        self.labels = []
        self.ip = []

        for i in range(0, 1):
            self.labels.append(make_edges3d(np.expand_dims(x[i], axis=0)))
            self.ip.append(make_edges3d(np.expand_dims(y[i], axis=0)))

        self.labels = np.asarray(self.labels)[:, :, 0:patch_size, 0:patch_size]
        self.ip = NDArray(np.asarray(self.ip)[:, :, 0:patch_size, 0:patch_size])
        print("data loaded")
        return (ip, labels)
        """

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
        self.ip = np.expand_dims(self.f['volume'][0,0:img_dim, 0:img_dim], axis = 0)
        #self.label = np.expand_dims(self.f['volumes/labels/neuron_ids'][0,0:img_dim,0:img_dim], axis=0)
        self.op = self.new_server.forward(self.ip)
        self.op = op.result().as_numpy()
        print("prediction run")
        return self.op

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

    def terminate(self):
        self.new_server.shutdown()


class BaseStrategy:
    def __init__(self, file, op):
        self.f = file
        self.op = op

    # compute loss for a given patch
    def base_loss(self, patch, label):
        label = label[0][0]
        patch = patch[0][0]
        result = mean_squared_error(label, patch)  # CHECK THIS
        return result


class Strategy1(BaseStrategy):
    def __init__(self, file, op):
        super().__init__(file,op)

    def run(self):
        idx = tile_image(self.op.shape, patch_size)
        label = np.expand_dims(self.f['volumes/labels/neuron_ids'][0,0:img_dim,0:img_dim], axis=0)
        #idx = tile_image(label.shape, patch_size)
        w, h, self.row, self.column = img_dim, img_dim, -1, -1
        error = 1e7
        for i in range(len(idx)):
            # print(pred_patches[i].shape, actual_patches[i].shape)
            curr_loss = super().base_loss(
                self.op[idx[i][0]: idx[i][1], idx[i][2]:idx[i][3], idx[i][4] : idx[i][5], idx[i][6] : idx[i][7]],
                labels[idx[i][0]: idx[i][1], idx[i][2]:idx[i][3], idx[i][4] : idx[i][5], idx[i][6] : idx[i][7]],
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
    file = robo.load_data()
    robo.load_model()
    robo.resume()  # resume training

    # run prediction
    op = robo.predict()

    metric = Strategy1(file, op)
    metric.run()
    row, column = metric.get_patch()
    robo.add(row, column)

    # shut down server
    robo.terminate()
    