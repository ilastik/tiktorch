import torch

from tiktorch.tiktypes import TikTensor, TikTensorBatch


def test_tt():
    a = TikTensor(torch.zeros(5, 5))
    assert isinstance(a.as_torch(with_label=True), tuple)
    assert len(a.as_torch(with_label=True)) == 2

def test_ttb():
    a = TikTensorBatch([TikTensor(torch.zeros(5, 5))])
    assert isinstance(a.as_torch(with_label=True), list)
    assert len(a.as_torch(with_label=True)) == 1
    assert all([len(aa) == 2 for aa in a.as_torch(with_label=True)])

