"""
Types defining interop between processes on the server
"""
import numpy
import torch

from numpy import ndarray
from typing import List, Tuple, Optional, Union, Sequence

from tiktorch.types import NDArray, LabeledNDArray, NDArrayBatch, LabeledNDArrayBatch

# import Point classes, because these used to be defined here.  # todo: change imports
from tiktorch.types import (
    PointAndBatchPointBase,
    PointBase,
    Point2D,
    Point3D,
    Point4D,
    BatchPointBase,
    BatchPoint2D,
    BatchPoint3D,
    BatchPoint4D,
)


class TikTensor:
    """
    Containter for pytorch tensor to transfer additional properties
    e.g. position of array in dataset (id_)
    """

    def __init__(
        self,
        tensor: Union[NDArray, ndarray, torch.Tensor],
        id_: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
        permute_to: str = "btczyx",
    ) -> None:
        """
        :param tensor: 5d NDArray or ndarray treated as tczyx or torch.Tensor of arbitrary shape (for torch.Tensor permute_to has no effect)
        :param id_: optional id
        :param permute_to: represent internal data in this order (only has an effect on tensor/label if they are not torch.Tensor)
        """
        if isinstance(tensor, NDArray):
            assert id_ is None or id_ == tensor.id
            id_ = tensor.id
            tensor = tensor.as_numpy()

        self.id = id_
        if isinstance(tensor, torch.Tensor):
            # no permuations for torch.Tensor
            self.un_subselect = ...
            self.un_permute = [i for i in range(len(tensor.shape))]
        else:
            if len(tensor.shape) != 5:
                raise ValueError(f"Expected 5d tensor, but got tensor with shape {tensor.shape}")

            tczyx = "tczyx"
            btczyx = "btczyx"
            subselect = []
            for a in btczyx:
                if a in permute_to:
                    if a == "b":
                        subselect.append(None)
                    else:
                        subselect.append(slice(None))
                elif a != "b":
                    if tensor.shape[tczyx.index(a)] != 1:
                        raise ValueError(f"tensor shape {tensor.shape} cannot be permuted to {permute_to}")

                    subselect.append(0)

            self.subselect = subselect
            un_subselect = []
            for a in btczyx:
                if a in permute_to:
                    if a == "b":
                        un_subselect.append(0)
                    else:
                        un_subselect.append(slice(None))
                elif a != "b":
                    un_subselect.append(None)

            self.un_subselect = un_subselect
            reduced_btczyx = [a for a in btczyx if a in permute_to]
            self.permute = [reduced_btczyx.index(a) for a in permute_to]
            self.un_permute = [permute_to.index(a) for a in reduced_btczyx]

            tensor = torch.from_numpy(tensor[self.subselect].transpose(self.permute))

        self._torch = tensor

    def add_label(self, label: Union[NDArray, ndarray, torch.Tensor]) -> "LabeledTikTensor":
        return LabeledTikTensor(tensor=self._torch, label=label, id_=self.id)

    def as_torch(self) -> torch.Tensor:
        """
        :return: underlyin torch tensor (with internal axis order)
        """
        return self._torch

    def as_numpy(self) -> ndarray:
        """
        :return: 5d numpy array with axis order btczyx
        """
        return self._torch.numpy().transpose(self.un_permute)[self.un_subselect]

    @property
    def dtype(self):
        return self._torch.dtype

    @property
    def torch_shape(self):
        """
        :return: shape of torch tensor (specified by 'permute_to')
        """
        return self._torch.shape

    @property
    def numpy_shape(self):
        """
        :return: 5d shape of numpy array
        """
        # todo: test this for correctness
        return numpy.array(self._torch.shape).transpose(self.un_permute)[self.un_subselect]


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
        permute_to: str = "btczyx",
    ) -> None:
        """
        :param tensor: 5d NDArray or ndarray treated as tczyx or torch.Tensor of arbitrary shape (for torch.Tensor permute_to has no effect)
        :param label: 5d NDArray or ndarray treated as tczyx or torch.Tensor of arbitrary shape (for torch.Tensor permute_to has no effect)
        :param id_: optional id
        :param permute_to: represent internal data in this order (only has an effect on tensor/label if they are not torch.Tensor)
        """
        if isinstance(tensor, LabeledNDArray):
            assert label is None
            tensor, label = tensor.as_numpy()
        elif isinstance(label, NDArray):
            assert self.id == label.id
            label = label.as_numpy()
        elif label is None:
            raise ValueError("missing label")

        super().__init__(tensor, id_, permute_to)

        if isinstance(label, torch.Tensor):
            # no permuations for torch.Tensor
            self._label = label
        else:
            self._label = torch.from_numpy(label[self.subselect].transpose(self.permute))

    def drop_label(self) -> TikTensor:
        return TikTensor(self._torch, self.id)

    def as_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._torch, self._label

    def as_numpy(self) -> Tuple[ndarray, ndarray]:
        return (
            self._torch.numpy().transpose(self.un_permute)[self.un_subselect],
            self._label.numpy().transpose(self.un_permute)[self.un_subselect],
        )


class TikTensorBatch:
    """
    Batch of TikTensor
    """

    def __init__(self, tensors: Union[List[TikTensor], NDArrayBatch], permute_to: str = "btczyx"):
        if isinstance(tensors, NDArrayBatch):
            tensors = [TikTensor(a, id_=a.id, permute_to=permute_to) for a in tensors]

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

    def __init__(self, tensors: Union[List[LabeledTikTensor], LabeledNDArrayBatch], permute_to: str = "btczyx"):
        super().__init__(tensors, permute_to)

    def drop_labels(self) -> TikTensorBatch:
        return TikTensorBatch([t.drop_label() for t in self._tensors])

    def add_labels(self, labels: Sequence) -> "LabeledTikTensorBatch":
        raise NotImplementedError(f"Cannot add labels to {self.__class__}")
