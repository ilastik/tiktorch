import unittest
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

    def test_process_2D(self):
        self.setUp()

    def test_process_3D(self):
        pass

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
        device_capacity = handler.binary_dry_run([1250, 1250])
        print(f"Max shape that devices can process: {device_capacity}")


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
