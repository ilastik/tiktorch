import contextlib
import shutil
import tempfile
from unittest import TestCase

import torch

from tiktorch.utils import TinyConvNet3D, WannabeConvNet3D
from tiktorch.wrapper import TikTorch


@contextlib.contextmanager
def tempdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class SimpleIOTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _test_wannabe_serialization(self):
        wannabe = WannabeConvNet3D()
        tik_torch = TikTorch(model=wannabe)

        with tempdir() as d:
            the_path = "{}/testfile.nn".format(d)
            tik_torch.serialize(to_path=the_path)
            new_torch = TikTorch.unserialize(from_path=the_path)
            self.assertIsInstance(new_torch, TikTorch)

    def _test_real_serialization(self):
        wannabe = TinyConvNet3D()
        tik_torch = TikTorch(model=wannabe)

        with tempdir() as d:
            the_path = "{}/testfile.nn".format(d)
            tik_torch.serialize(to_path=the_path)
            new_torch = TikTorch.unserialize(from_path=the_path)
            self.assertIsInstance(new_torch, TikTorch)

    def _test_gpu_serialization(self):
        wannabe = TinyConvNet3D()
        tik_torch = TikTorch(model=wannabe)
        tik_torch.cuda()
        with tempdir() as d:
            the_path = "{}/testfile.nn".format(d)
            tik_torch.serialize(to_path=the_path)
            new_torch = TikTorch.unserialize(from_path=the_path)
            self.assertIsInstance(new_torch, TikTorch)
            self.assertTrue(new_torch.is_cuda)

    def test_serialization(self):
        self._test_real_serialization()
        self._test_wannabe_serialization()
        if torch.cuda.is_available():
            self._test_gpu_serialization()
