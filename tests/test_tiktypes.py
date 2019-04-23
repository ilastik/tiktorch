import torch

from tiktorch.tiktypes import LabeledTikTensor, LabeledTikTensorBatch


def test_tt():
    a = LabeledTikTensor(torch.zeros(5, 5), torch.zeros(5, 5))
    assert isinstance(a.as_torch(), tuple)
    assert len(a.as_torch()) == 2


def test_ttb():
    a = LabeledTikTensorBatch([LabeledTikTensor(torch.zeros(5, 5), torch.zeros(5, 5))])
    assert isinstance(a.as_torch(), list)
    assert len(a.as_torch()) == 1
    assert all([len(aa) == 2 for aa in a.as_torch()])
