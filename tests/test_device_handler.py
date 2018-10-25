import os
import unittest
import h5py
from importlib import util as imputils
import torch
import torch.nn as nn
from tiktorch.device_handler import ModelHandler
from tiktorch.blockinator import Blockinator

class ProcessTest(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Conv2d(3, 512, 3),
                                   nn.Conv2d(512, 512, 3),
                                   nn.Conv2d(512, 512, 3),
                                   nn.Conv2d(512, 3, 3))
        self.handler = ModelHandler(model=self.model,
                                    channels=3,
                                    device_names='cpu',
                                    dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')

    def test_denseUNet(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        from inferno.io.transform import Compose
        from inferno.io.transform.generic import Normalize, Cast, AsTorchBatch
        model_file_name = '/home/jo/config/model.py'
        #'/export/home/jhugger/sfb1129/test_configs_tiktorch/config/model.py'
        module_spec = imputils.spec_from_file_location('model', model_file_name)
        module = imputils.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model: torch.nn.Module = \
            getattr(module, 'DUNet2D')(in_channels=1, out_channels=1)
        state_path = '/home/jo/config/state.nn'
        #'/export/home/jhugger/sfb1129/test_configs_tiktorch/config/state.nn'
        
        try:
            state_dict = torch.load(state_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
        except:
            raise FileNotFoundError(f"Model weights could not be found at location '{state_path}'!")

        handler = ModelHandler(model=model,
                               channels=1,
                               device_names='cpu',
                               dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
        shape = handler.binary_dry_run([1250, 1250])            
        transform = Compose(Normalize(), Cast('float32'))

        #with h5py.File('/export/home/jhugger/sfb1129/sample_C_20160501.hdf') as f:
        with h5py.File('/home/jo/sfb1129/sample_C_20160501.hdf') as f:
            cremi_raw = f['volumes']['raw'][0:10, 0:shape[0], 0:shape[1]]

        for i in range(10):
            input_tensor = torch.from_numpy(transform(cremi_raw[i: i+1]))
            out = handler.forward(torch.unsqueeze(input_tensor, 0))
            import matplotlib.pyplot as plt
            plt.imshow(out[0, 0])
            plt.show()


    def test_process_2D(self):
        self.setUp()
        #shape = self.handler.binary_dry_run([256, 256])
        #out = self.handler.forward(torch.zeros(*([1, 3] + [128, 128]), dtype=torch.float32))

    def test_process_3D(self):
        model = nn.Sequential(nn.Conv3d(1, 512, 3),
                              nn.Conv3d(512, 512, 3),
                              nn.Conv3d(512, 512, 3),
                              nn.Conv3d(512, 1, 3))
        handler = ModelHandler(model=self.model,
                               channels=3,
                               device_names='cpu',
                               dynamic_shape_code='(2 * (nD + 1), 32 * (nH + 1), 32 * (nW + 1))')
        #shape = handler.binary_dry_run([10, 64, 64])
        #out = handler.forward(torch.zeros(*([1, 1] + [4, 64, 64]), dtype=torch.float32))

    def test_halo_computer(self):
        self.setUp()
        halo = self.handler.halo
        halo_in_blocks = self.handler.halo_in_blocks
        print(f"Halo: {halo}")
        print(f"Halo in blocks: {halo_in_blocks}")

    def test_binary_dry_run(self):
        self.setUp()
        handler = ModelHandler(model=self.model,
                               device_names=['cpu'],
                               channels=3,
                               dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
        #device_capacity = handler.binary_dry_run([250, 250])
        #print(f"Max shape that devices can process: {device_capacity}")


#    def test_dry_run():
#        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
#        model = nn.Sequential(nn.Conv2d(3, 512, 3),
#                              nn.Conv2d(512, 512, 3),
#                              nn.Conv2d(512, 512, 3),
#                              nn.Conv2d(512, 3, 3))
#        handler = ModelHandler(model=model,
#                               device_names=['cpu'], #['cuda:0', 'cuda:1'],
#                               in_channels=3, out_channels=3,
#                               dynamic_shape_code='(120 * (nH + 1), 120 * (nW + 1))')
#        handler.dry_run()
#        print(f"GPU0 Specs: {handler.get_device_spec(0)}")
#        print(f"GPU1 Specs: {handler.get_device_spec(1)}")
#
#    def test_dry_run_on_device():
#        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
#        model = nn.Sequential(nn.Conv2d(3, 512, 3),
#                              nn.Conv2d(512, 512, 3),
#                              nn.Conv2d(512, 512, 3),
#                              nn.Conv2d(512, 3, 3))
#        handler = ModelHandler(model=model,
#                               device_names='cuda:0',
#                               channels=3,
#                               dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
#        spec = handler._dry_run_on_device(0)
#        print(f"GPU Specs: {spec}")

        
if __name__ == '__main__':
    unittest.main()
