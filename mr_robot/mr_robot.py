import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.metrics import mean_squared_error
import zipfile
import h5py
import z5py
from z5py.converter import convert_from_h5
from scipy.ndimage import convolve
from torch.autograd import Variable
from collections import OrderedDict
import yaml
import logging
from abc import ABC, abstractmethod
from tiktorch.server import TikTorchServer
from tiktorch.rpc import Client, Server, InprocConnConf
from tiktorch.rpc_interface import INeuralNetworkAPI
from tiktorch.types import NDArray, NDArrayBatch
from tests.conftest import nn_sample
from mr_robot.utils import *

patch_size = 16
img_dim = 32


class MrRobot:
    def __init__(self, path_to_config_file, strategy):
        # start the server
        self.new_server = TikTorchServer()
        self.strategy =strategy

        with open(path_to_config_file, mode="r") as f:
            self.base_config = yaml.load(f)

        self.max_robo_iterations = self.base_config['max_robo_iterations']
        self.counter = 0
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.f = z5py.File(self.base_config["cremi_data_dir"])
        self.logger('data file loaded')


    def load_model(self):
        # load the model

        
        #with open(base_config['cremi_data_dir'], mode="rb") as f:
        #    binary_state = f.read()

        archive = zipfile.ZipFile(self.base_config['cremi_dir']['path_to_zip'], 'r')
        model = archive.read(self.base_config['cremi_dir']['path_in_zip_to_model'])
        binary_state = archive.read(self.base_config['cremi_dir']['path_in_zip_to_state'])

        #cleaning dictionary before passing to tiktorch
        self.base_config.pop('cremi_dir')
        self.base_config.pop('cremi_data')
        self.base_config.pop('cremi_path_to_labelled')

        #with open("model.py", mode="rb") as f:
        #    model_file = f.read()
        
        fut = self.new_server.load_model(base_config, model, binary_state, b"", ["cpu"])
        self.logger.info("model loaded")

    def resume(self):
        self.new_server.resume_training()
        self.logger.info("training resumed")

    def predict(self):
        self.ip = self.f["volume"][0:1, 0:img_dim, 0:img_dim]
        # self.label = np.expand_dims(self.f['volumes/labels/neuron_ids'][0,0:img_dim,0:img_dim], axis=0)
        self.op = self.new_server.forward(self.ip)
        self.op = self.op.result().as_numpy()
        #self.logger.info("prediction run")
    
    def stop(self):
        if(self.counter > self.max_robo_iterations):
            return False
        else:
            self.counter+=1
            return True

    def run(self):
        self.strategy.patch('MSE', self.op)
        while(self.stop()):
            idx = self.strategy.get_next_patch(self.op)
            self.add(idx)

    def add(self, idx):
        file = z5py.File(self.base_config["cremi_data"])
        labels = file["cremi_path_to_labelled"][0:1, 0:img_dim, 0:img_dim]

        new_ip = self.ip.as_numpy()[idx[0]:idx[1], idx[2]:idx[3], idx[4]:idx[5]].astype(float)
        new_label = labels[ idx[0]:idx[1], idx[2]:idx[3], idx[4]:idx[5] ].astype(float)
        # print(ip.dtype, label.dtype)
        self.new_server.update_training_data(NDArrayBatch([NDArray(new_ip)]), NDArrayBatch([new_label]))

    # annotate worst patch
    def dense_annotate(self, x, y, label, image):
        raise NotImplementedError

    def terminate(self):
        self.new_server.shutdown()


class BaseStrategy(ABC):

    def __init__(self, path_to_config_file):
        with open(path_to_config_file, mode="r") as f:
            self.base_config = yaml.load(f)
        #self.op = op
        self.logger = logging.getLogger(__name__)
   
    def loss(self,tile,label, loss_fn):
        label = label[0]
        tile = tile[0]
        result = mean_squared_error(label, tile)  # CHECK THIS
        return result

    def base_patch(self, loss_fn, op):
        idx = tile_image(op.shape, patch_size)
        file = z5py.File(self.base_config["cremi_data"])
        labels = file["cremi_path_to_labelled"][0:1, 0:img_dim, 0:img_dim]

        self.patch_data = []
        for i in range(len(idx)):
            curr_loss = self.loss(
                op[idx[i][0] : idx[i][1], idx[i][2] : idx[i][3], idx[i][4] : idx[i][5]],
                labels[idx[i][0] : idx[i][1], idx[i][2] : idx[i][3], idx[i][4] : idx[i][5]],
                loss_fn
            )

            self.patch_data.append((curr_loss,idx[i]))
            self.logger.info("loss for patch %d: %d" % (i,curr_loss) )


    @abstractmethod
    def get_next_patch(self):
        pass


class Strategy1(BaseStrategy):

    def __init__(self, path_to_config_file):
        super().__init__(path_to_config_file)
        self.counter=-1

    def patch(self, loss_fn, op):
        super().base_patch(loss_fn,op)
        self.patch_data.sort(reverse = True)

    def get_next_patch(self):
        self.counter+=1
        return self.patch_data[self.counter][1]


class Strategy2(BaseStrategy):
    def __init__():
        super().__init__()


class Strategy3(BaseStrategy):
    def __init__():
        super().__init__()


