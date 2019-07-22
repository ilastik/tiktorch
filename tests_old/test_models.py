import logging
import os
import unittest
from importlib import util as imputils

import h5py
import torch
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch, Cast, Normalize
from torch.autograd import Variable

from tiktorch.device_handler import ModelHandler
from tiktorch.models.dunet import DUNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.DEBUG)


class TestUNet(unittest.TestCase):
    def setUp(self):
        path = "/export/home/jhugger/sfb1129/pretrained_net_constantin/ISBI2012_UNet_pretrained/"
        model_file_name = path + "model.py"  #'/export/home/jhugger/sfb1129/ISBI2012_UNet_pretrained/model.py'
        module_spec = imputils.spec_from_file_location("model", model_file_name)
        module = imputils.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model: torch.nn.Module = getattr(module, "UNet2dGN")(in_channels=1, initial_features=64, out_channels=1)
        state_path = path + "state.nn"  #'/export/home/jhugger/sfb1129/ISBI2012_UNet_pretrained/state.nn'

        try:
            state_dict = torch.load(state_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except:
            raise FileNotFoundError(f"Model weights could not be found at location '{state_path}'!")

        self.handler = ModelHandler(
            model=model, channels=1, device_names="cuda:0", dynamic_shape_code="(32 * (nH + 1), 32 * (nW + 1))"
        )

    def test_model(self):
        self.setUp()
        # shape = self.handler.binary_dry_run([2000, 2000])
        transform = Compose(Normalize(), Cast("float32"))

        # with h5py.File('/export/home/jhugger/sfb1129/sample_C_20160501.hdf') as f:
        with h5py.File("/export/home/jhugger/sfb1129/sample_C_20160501.hdf") as f:
            cremi_raw = f["volumes"]["raw"][0:1, 0:1248, 0:1248]

        input_tensor = torch.from_numpy(transform(cremi_raw[0:1]))
        input_tensor = torch.rand(1, 572, 572)
        print(torch.unsqueeze(input_tensor, 0).shape)
        out = self.handler.forward(torch.unsqueeze(input_tensor, 0))
        import scipy

        scipy.misc.imsave("/export/home/jhugger/sfb1129/tiktorch/out.jpg", out[0, 0].data.cpu().numpy())
        scipy.misc.imsave("/home/jo/server/tiktorch/out.jpg", out[0, 0].data.cpu().numpy())


class TestDenseUNet(unittest.TestCase):
    def setUp(self):
        model_file_name = "/export/home/jhugger/sfb1129/CREMI_DUNet_pretrained/model.py"
        module_spec = imputils.spec_from_file_location("model", model_file_name)
        module = imputils.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model: torch.nn.Module = getattr(module, "DUNet2D")(in_channels=1, out_channels=1)
        state_path = "/export/home/jhugger/sfb1129/CREMI_DUNet_pretrained/state.nn"

        try:
            state_dict = torch.load(state_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except:
            raise FileNotFoundError(f"Model weights could not be found at location '{state_path}'!")

        self.handler = ModelHandler(
            model=model, channels=1, device_names="cuda:0", dynamic_shape_code="(32 * (nH + 1), 32 * (nW + 1))"
        )

    def test_model(self):
        self.setUp()
        shape = self.handler.binary_dry_run([1250, 1250])
        transform = Compose(Normalize(), Cast("float32"))

        with h5py.File("/export/home/jhugger/sfb1129/sample_C_20160501.hdf") as f:
            # with h5py.File('/home/jo/sfb1129/sample_C_20160501.hdf') as f:
            cremi_raw = f["volumes"]["raw"][0:1, 0 : shape[0], 0 : shape[1]]

        input_tensor = torch.from_numpy(transform(cremi_raw[0:1]))
        out = self.handler.forward(torch.unsqueeze(input_tensor, 0))
        import scipy

        scipy.misc.imsave("/export/home/jhugger/sfb1129/tiktorch/out.jpg", out[0, 0].data.cpu().numpy())


class TestModels(unittest.TestCase):
    def test_dunet(self):
        # Make network
        model = DUNet(1, 1)
        # Input variable
        input_variable = Variable(torch.rand(1, 1, 512, 512))
        # noinspection PyCallingNonCallable
        output_variable = model(input_variable)
        self.assertEqual(list(output_variable.data.size()), [1, 1, 512, 512])


if __name__ == "__main__":
    unittest.main()
