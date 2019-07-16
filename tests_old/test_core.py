from unittest import TestCase

import numpy as np
import torch

from tiktorch.utils import TinyConvNet3D
from tiktorch.wrapper import TikTorch


class CoreTest(TestCase):
    def setUp(self):
        # Build model and object
        model = TinyConvNet3D()
        tiktorch = TikTorch(model=model).configure(
            window_size=[3, 512, 512], num_input_channels=1, num_output_channels=1
        )
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

    def test_multi_gpu(self):
        # Specifies the devices to test on
        devices = [0, 1, 2, 3]
        # Do nothing if cuda is not available
        if not torch.cuda.is_available():
            return
        self.setUp()
        # Assuming 4 GPUs
        self.tiktorch.cuda(*devices)
        self.assertEqual(self.tiktorch.get("devices"), devices)
        # Forward
        output = self.tiktorch.forward([self.input_array.copy() for _ in devices])
        # noinspection PyTypeChecker
        self.assertEqual(len(output), len(devices))
        # noinspection PyUnresolvedReferences
        self.assertIsInstance(output[0], np.ndarray)
        self.assertTrue(all([output_.shape == (1, 3, 512, 512) for output_ in output]))


class ForwardTest(TestCase):
    def test_more_output_channels(self):
        model = TinyConvNet3D(num_input_channels=1, num_output_channels=2)
        tiktorch = TikTorch(model=model).configure(
            window_size=[3, 512, 512], num_input_channels=1, num_output_channels=2
        )
        self.tiktorch = tiktorch
        self.input_array = np.zeros(shape=(1, 3, 512, 512))

        # Forward and check
        output = self.tiktorch.forward([self.input_array])
        # noinspection PyTypeChecker
        self.assertEqual(len(output), 1)
        # noinspection PyUnresolvedReferences
        self.assertIsInstance(output[0], np.ndarray)
        # noinspection PyUnresolvedReferences
        self.assertEqual(output[0].shape, (2, 3, 512, 512))

    def test_less_output_channels(self):
        model = TinyConvNet3D(num_input_channels=3, num_output_channels=2)
        tiktorch = TikTorch(model=model).configure(
            window_size=[3, 512, 512], num_input_channels=3, num_output_channels=2
        )
        self.tiktorch = tiktorch
        self.input_array = np.zeros(shape=(3, 3, 512, 512))

        # Forward and check
        output = self.tiktorch.forward([self.input_array])
        # noinspection PyTypeChecker
        self.assertEqual(len(output), 1)
        # noinspection PyUnresolvedReferences
        self.assertIsInstance(output[0], np.ndarray)
        # noinspection PyUnresolvedReferences
        self.assertEqual(output[0].shape, (2, 3, 512, 512))
