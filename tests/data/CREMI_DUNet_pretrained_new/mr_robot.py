# simulating user initial draft
#import sys
#sys.path.append('D:/Machine Learning/tiktorch/tests/data/CREMI_DUNet_pretrained_new')
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
from tiktorch.server import TikTorchServer
from tiktorch.rpc import Client, Server, InprocConnConf
from tiktorch.rpc_interface import INeuralNetworkAPI
from tiktorch.types import NDArray, NDArrayBatch


def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) *
                           batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print(
        "Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")


class Mr_Robot():
        # def __init__(self, y_pred ):
    #    self.y_pred = y_pred

        # create patches
    def tile_image(self, image, tile_size):

        tiles = []
        w, h, = 32, 32
        for wsi in range(0, w - tile_size + 1, int(tile_size)):
            for hsi in range(0, h - tile_size + 1, int(tile_size)):
                img = image[wsi:wsi + tile_size, hsi:hsi + tile_size]
                tiles.append(img)

        if h % tile_size != 0:
            for wsi in range(0, w - tile_size + 1, int(tile_size)):
                img = image[wsi:wsi + tile_size, h - tile_size:]

                tiles.append(img)

        if w % tile_size != 0:
            for hsi in range(0, h - tile_size + 1, int(tile_size)):
                img = image[w - tile_size:, hsi:hsi + tile_size]
                tiles.append(img)

        if w % tile_size != 0 and h % tile_size != 0:
            img = image[w - tile_size:, h - tile_size:]
            tiles.append(img)

        return tiles

    # compute loss for a given patch
    def loss(self, patch, label):
        #patch, label = patch[0][0], label[0][0]
        #print(patch.shape, label.shape)
        result = mean_squared_error(label, patch)  # CHECK THIS
        return result

    # annotate worst patch
    def dense_annotate(x, y, label, image):
        # for i in range(x*patch_size,(x+1)*patch_size):
        #   for j in range(y*patch_size, (y+1)*patch_size):
        #       img = image[]
        img = label[x * patch_size:(x + 1) * patch_size,
                    y * patch_size:(y + 1) * patch_size]
        # add to labeled data ??
        #


patch_size = 16


def start_robot(model, label):
    robot = mr_robot()

    y_pred = model(ip)
    pred_patches = robot.tile_image(y_pred, patch_size)
    label_patch = robot.tile_image(label, patch_size)

    w, h = image.shape
    # find patch with highest loss
    error = 1e7
    for i in pred_patches:
        curr_loss = robot.loss(pred_patches[i], label[i])
        if(error < curr_loss):
            error = curr_loss
            row, column = i / (w / patch_size), i % (w / patch_size)

    # densely annotate this patch


def make_edges3d(segmentation):
    """ Make 3d edge volume from 3d segmentation
    """
    # NOTE we add one here to make sure that we don't have zero in the segmentation
    gz = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1, 1))
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
    return ((gx ** 2 + gy ** 2 + gz ** 2) > 0)

#train_data = h5py.File('train.hdf','r')
#file = open('neuronid.txt','r')
#lines = file.read().split('    ')
# print(len(lines))

# read data and modify it


def load_data():
    print("lol")
    with h5py.File('train.hdf', 'r') as f:
        x = np.array(f.get('volumes/labels/neuron_ids'))
        y = np.array(f.get('volumes/raw'))

    labels = []
    ip = []

    for i in range(0, 1):
        labels.append(make_edges3d(np.expand_dims(x[i], axis=0)))
        ip.append(make_edges3d(np.expand_dims(y[i], axis=0)))

    labels = np.asarray(labels)[:, :, 0:32, 0:32]
    ip = NDArray(np.asarray(ip)[:, :, 0:32, 0:32])
    print("data loaded")
    return (ip, labels)


if __name__ == "__main__":

    # load data
    ip, labels = load_data()

    # start the server
    new_server = TikTorchServer()

    # load the model
    with open("state.nn", mode="rb") as f:
        binary_state = f.read()
    with open("model.py", mode='rb') as f:
        model_file = f.read()

    base_config = {
        "model_class_name": "DUNet2D",
        "model_init_kwargs": {"in_channels": 1, "out_channels": 1},
        #"input_channels": 1,
        "training": {
            "training_shape": [1, 32, 32],
            "batch_size": 1,
            "loss_criterion_config": {"method": "MSELoss"},
            "optimizer_config": {"method": "Adam"},
            "num_iterations_done": 1,
            "max_num_iterations_per_update": 2
        },
        "validation": {},
        "dry_run": {
            "skip": True,
            "shrinkage": [0, 0, 0]
        }
    }

    fut = new_server.load_model(
        base_config, model_file, binary_state, b'', ["cpu"])
    print("load model", fut)
    print(fut.result())

    # resume training
    new_server.resume_training()
    print("training resumed")

    # run prediction
    op = new_server.forward(ip)
    op = op.result().as_numpy()
    print("prediction run", op.shape, labels.shape)

    # patch output and find worst patch
    robo = Mr_Robot()
    pred_patches = robo.tile_image(op[0, 0, :, :], 16)
    actual_patches = robo.tile_image(labels[0, 0, :, :], 16)
    # print(len(pred_patches),len(actual_patches))

    w, h, row, column = 32, 32, -1, -1
    error = 1e7
    for i in range(len(pred_patches)):
        #print(pred_patches[i].shape, actual_patches[i].shape)

        curr_loss = robo.loss(pred_patches[i], actual_patches[i])
        print(curr_loss)
        if(error > curr_loss):
            error = curr_loss
            row, column = int(i / (w / patch_size)), int(i % (w / patch_size))

    # add to training data
    ip = ip.as_numpy()[0, :, patch_size * row:patch_size * (row + 1),
                       patch_size * column:patch_size * (column + 1)].astype(int)
    label = labels[0, :, patch_size * row:patch_size * (row + 1),
                       patch_size * column:patch_size * (column + 1)].astype(int)
    print(ip.shape, label )
    print(ip.dtype, label.dtype)
    new_server.update_training_data(NDArrayBatch([NDArray(ip)]), NDArrayBatch([label]))

    # shut down server
    new_server.shutdown()
