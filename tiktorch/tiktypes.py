"""
Types defining interop between processes on the server
"""
import torch

from numpy import ndarray
from typing import List, Tuple, Optional, Union, Sequence

from tiktorch.types import NDArray, LabeledNDArray, NDArrayBatch, LabeledNDArrayBatch


class TikTensor:
    """
    Containter for pytorch tensor to transfer additional properties
    e.g. position of array in dataset (id_)
    """

    def __init__(
        self,
        tensor: Union[NDArray, ndarray, torch.Tensor],
        id_: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
    ) -> None:
        if isinstance(tensor, NDArray):
            assert id_ is None
            id_ = tensor.id
            tensor = torch.from_numpy(tensor.as_numpy())
        elif isinstance(tensor, ndarray):
            tensor = torch.from_numpy(tensor)

        self.id = id_
        self._torch = tensor

    def add_label(self, label: Union[NDArray, ndarray, torch.Tensor]) -> "LabeledTikTensor":
        return LabeledTikTensor(tensor=self._torch, label=label, id_=self.id)

    def as_torch(self) -> torch.Tensor:
        return self._torch

    def as_numpy(self) -> ndarray:
        return self._torch.numpy()

    @property
    def dtype(self):
        return self._torch.dtype

    @property
    def shape(self):
        return self._torch.shape


class LabeledTikTensor(TikTensor):
    """
    Containter for pytorch tensor with a label to transfer additional properties
    e.g. position of array in dataset (id_)
    """

    def __init__(
        self,
        tensor: Union[LabeledNDArray, ndarray, torch.Tensor],
        label: Optional[Union[NDArray, ndarray, torch.Tensor]],
        id_: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
    ) -> None:
        if isinstance(tensor, NDArray):
            assert id_ is None
            id_ = tensor.id
            tensor = torch.from_numpy(tensor.as_numpy())
        elif isinstance(tensor, ndarray):
            tensor = torch.from_numpy(tensor)

        super().__init__(tensor, id_)

        if isinstance(tensor, LabeledNDArray):
            assert label is None
            label = torch.from_numpy(label.as_numpy())
        elif isinstance(label, NDArray):
            assert self.id == label.id
            label = torch.from_numpy(label.as_numpy())
        elif isinstance(label, ndarray):
            label = torch.from_numpy(label)
        elif label is None:
            raise ValueError("missing label")

        self._label = label

    def drop_label(self) -> TikTensor:
        return TikTensor(self._torch, self.id)

    def as_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._torch, self._label

    def as_numpy(self) -> Tuple[ndarray, ndarray]:
        return self._torch.numpy(), self._label.numpy()


class TikTensorBatch:
    """
    Batch of TikTensor
    """

    def __init__(self, tensors: Union[List[TikTensor], NDArrayBatch]):
        if isinstance(tensors, NDArrayBatch):
            tensors = [TikTensor(a) for a in tensors]

        assert all([isinstance(t, TikTensor) for t in tensors])
        self._tensors = tensors

    def tensor_metas(self):
        return [{"dtype": t.dtype.str, "shape": t.shape, "id": t.id} for t in self._tensors]

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        for item in self._tensors:
            yield item

    def as_torch(self) -> List[torch.Tensor]:
        return [t.as_torch() for t in self._tensors]

    def as_numpy(self) -> List[ndarray]:
        return [t.as_numpy() for t in self._tensors]

    @property
    def ids(self) -> List[Tuple[int]]:
        return [t.id for t in self._tensors]

    def add_labels(self, labels: Sequence) -> "LabeledTikTensorBatch":
        assert len(labels) == len(self)
        return LabeledTikTensorBatch([t.add_label(l) for t, l in zip(self, labels)])


class LabeledTikTensorBatch(TikTensorBatch):
    """
    Batch of LabeledTikTensor
    """

    def __init__(self, tensors: Union[List[LabeledTikTensor], LabeledNDArrayBatch]):
        super().__init__(tensors)

    def drop_labels(self) -> TikTensorBatch:
        return TikTensorBatch([t.drop_label() for t in self._tensors])
