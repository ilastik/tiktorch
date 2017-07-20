import numpy as np
import torch
from tiktorch.wrapper import TikTorch
from tiktorch.utils import TinyConvNet3D

from unittest import TestCase


class CoreTest(TestCase):
    def setUp(self):
        # Build model and object
        model = TinyConvNet3D()
        tiktorch = TikTorch(model=model).configure(window_size=[3, 512, 512],
                                                   num_input_channels=1, 
                                                   num_output_channels=1)
        self.tiktorch = tiktorch
        self.input_array = np.zeros(shape=(1, 3, 512, 512))

    def test_model_forward(self):
        # New setup
        self.setUp()
        # Forward and check
        output = self.tiktorch.forward([self.input_array])
        # noinspection PyTypeChecker
        self.assertEqual(len(output), 1)
        # noinspection PyUnresolvedReferences
        self.assertIsInstance(output[0], np.ndarray)
        # noinspection PyUnresolvedReferences
        self.assertEqual(output[0].shape, (1, 3, 512, 512))

    def test_gpu(self):
        if not torch.cuda.is_available():
            # No GPU, no test
            return
        self.setUp()
        self.assertTrue(not self.tiktorch.is_cuda)
        # Transfer to GPU
        self.tiktorch.cuda()
        self.assertTrue(self.tiktorch.is_cuda)
        # Forward and check
        output = self.tiktorch.forward([self.input_array])
        # noinspection PyTypeChecker
        self.assertEqual(len(output), 1)
        # noinspection PyUnresolvedReferences
        self.assertIsInstance(output[0], np.ndarray)
        # noinspection PyUnresolvedReferences
        self.assertEqual(output[0].shape, (1, 3, 512, 512))