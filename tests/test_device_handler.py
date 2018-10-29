import os
import unittest
import h5py
import logging
from importlib import util as imputils
import torch
import torch.nn as nn
from tiktorch.device_handler import ModelHandler
from tiktorch.blockinator import Blockinator

logging.basicConfig(level=logging.DEBUG)

class ProcessTest(unittest.TestCase):
    def test_process_2D(self):
        return 0
        model = nn.Sequential(nn.Conv2d(3, 512, 3),
                              nn.Conv2d(512, 512, 3),
                              nn.Conv2d(512, 512, 3),
                              nn.Conv2d(512, 3, 3))
        handler = ModelHandler(model=model,
                               channels=3,
                               device_names='cpu',
                               dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
        shape = handler.binary_dry_run([256, 256])
        out = handler.forward(torch.zeros(*([1, 3, 128, 128]), dtype=torch.float32))

    def test_process_3D(self):
        model = nn.Sequential(nn.Conv3d(1, 12, 3),
                              nn.Conv3d(12, 12, 3),
                              nn.Conv3d(12, 12, 3),
                              nn.Conv3d(12, 1, 3))
        handler = ModelHandler(model=model,
                               channels=1,
                               device_names='cpu',
                               dynamic_shape_code='(32 * (nD + 1), 32 * (nH + 1), 32 * (nW + 1))')
        shape = handler.binary_dry_run([96, 96, 96])
        out = handler.forward(torch.zeros(*([1, 1, 96, 96, 96]), dtype=torch.float32))

class DryRunTest(unittest.TestCase):
    def test_binary_dry_run_2d(self):
        model = nn.Sequential(nn.Conv2d(3, 2512, 3),
                              nn.Conv2d(2512, 3512, 3),
                              nn.Conv2d(3512, 512, 3),
                              nn.Conv2d(512, 3, 3))
        handler = ModelHandler(model=model,
                               device_names=['cpu'],
                               channels=3,
                               dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
        image_shapes = [[512, 512], [760, 520], [1024, 1250], [3250, 4002]]
        for shape in image_shapes:
            device_capacity = handler.binary_dry_run(shape)
            print(f"Max shape that devices can process: {device_capacity}")

    def test_binary_dry_run_3d(self):
        model = nn.Sequential(nn.Conv3d(3, 512, 3),
                              nn.Conv3d(512, 1512, 3),
                              nn.Conv3d(1512, 512, 3),
                              nn.Conv3d(512, 3, 3))
        handler = ModelHandler(model=model,
                               device_names=['cpu'],
                               channels=3,
                               dynamic_shape_code='(10 * (nD + 1), 32 * (nH + 1), 32 * (nW + 1))')
        volumes = [[10, 512, 1024], [2000, 2000, 2000], [50, 256, 256]]
        for volume in volumes:
            device_capacity = handler.binary_dry_run(volume)
            print(f"Max shape that devices can process: {device_capacity}")

    def test_dry_run(self):
        return 0
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        model = nn.Sequential(nn.Conv2d(3, 512, 3),
                            nn.Conv2d(512, 512, 3),
                            nn.Conv2d(512, 512, 3),
                            nn.Conv2d(512, 3, 3))
        handler = ModelHandler(model=model,
                               device_names=['cpu'], #['cuda:0', 'cuda:1'],
                               in_channels=3, out_channels=3,
                               dynamic_shape_code='(120 * (nH + 1), 120 * (nW + 1))')
        handler.dry_run()
        print(f"GPU0 Specs: {handler.get_device_spec(0)}")
        print(f"GPU1 Specs: {handler.get_device_spec(1)}")

    def test_dry_run_on_device(self):
        return 0
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        model = nn.Sequential(nn.Conv2d(3, 512, 3),
                              nn.Conv2d(512, 512, 3),
                              nn.Conv2d(512, 512, 3),
                              nn.Conv2d(512, 3, 3))
        handler = ModelHandler(model=model,
                               device_names='cuda:0',
                               channels=3,
                               dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
        spec = handler._dry_run_on_device(0)
        print(f"GPU Specs: {spec}")

class HaloTest(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Conv2d(3, 512, 3),
                                   nn.Conv2d(512, 512, 3),
                                   nn.Conv2d(512, 512, 3),
                                   nn.Conv2d(512, 3, 3))
        self.handler = ModelHandler(model=self.model,
                                    channels=3,
                                    device_names='cpu',
                                    dynamic_shape_code='(32 * (nH + 1), 32 * (nW + 1))')
    def test_halo_computer(self):
        self.setUp()
        halo = self.handler.halo
        halo_in_blocks = self.handler.halo_in_blocks
        print(f"Halo: {halo}")
        print(f"Halo in blocks: {halo_in_blocks}")

        
if __name__ == '__main__':
    unittest.main()
