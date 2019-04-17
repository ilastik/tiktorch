import numpy
import torch

from tiktorch.tiktypes import TikTensor, LabeledTikTensor, TikTensorBatch, LabeledTikTensorBatch


def test_tt():
    a = LabeledTikTensor(torch.zeros(5, 5), torch.zeros(5, 5))
    assert isinstance(a.as_torch(), tuple)
    assert len(a.as_torch()) == 2


def test_ttb():
    a = LabeledTikTensorBatch([LabeledTikTensor(torch.zeros(5, 5), torch.zeros(5, 5))])
    assert isinstance(a.as_torch(), list)
    assert len(a.as_torch()) == 1
    assert all([len(aa) == 2 for aa in a.as_torch()])

def test_tiktensor_permute_1():
    a = TikTensor(numpy.ones((5, 2, 1, 4, 3)), permute_to="xcyt")
    assert a.as_numpy().shape == (5, 2, 1, 4, 3)
    assert a.as_torch().shape == (3, 2, 4, 5)

def test_tiktensor_permute_2():
    a = TikTensor(numpy.ones((4, 7, 1, 5, 1)), permute_to="cbyt")
    assert a.as_numpy().shape == (4, 7, 1, 5, 1)
    assert a.as_torch().shape == (7, 1, 5, 4)
