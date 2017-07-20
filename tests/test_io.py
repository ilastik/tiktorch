from tiktorch.wrapper import TikTorch
from tiktorch.utils import WannabeConvNet3D, TinyConvNet3D
import contextlib
import tempfile
import shutil

from unittest import TestCase


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
            the_path = '{}/testfile.nn'.format(d)
            tik_torch.serialize(to_path=the_path)
            new_torch = TikTorch.unserialize(from_path=the_path)
            assert isinstance(new_torch, TikTorch)

    def _test_real_serialization(self):
        wannabe = TinyConvNet3D()
        tik_torch = TikTorch(model=wannabe)

        with tempdir() as d:
            the_path = '{}/testfile.nn'.format(d)
            tik_torch.serialize(to_path=the_path)
            new_torch = TikTorch.unserialize(from_path=the_path)
            assert isinstance(new_torch, TikTorch)

    # TODO Test serialization of GPU arrays
    def test_serialization(self):
        self._test_real_serialization()
        self._test_wannabe_serialization()
