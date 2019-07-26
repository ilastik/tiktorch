import itertools
import logging

import numpy
import torch
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from tiktorch.tiktypes import LabeledTikTensorBatch, TikTensor, TikTensorBatch

logger = logging.getLogger(__name__)


class EntryRemoved(Exception):
    pass


class EmptyDataset(Exception):
    pass


class DynamicDataLoaderWrapper(DataLoader):
    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader

    def __iter__(self):
        loader = iter(self._dataloader)
        while True:
            try:
                yield next(loader)
            except EntryRemoved:
                pass


class DynamicDataset(Dataset):
    class _Entry:
        __slots__ = ("weight", "data", "num_updates", "removed")

        def __init__(self, data, weight=1.0):
            self.weight = weight
            self.data = data
            self.num_updates = 0
            self.removed = False

    def __init__(self, transform=None, gamma: float = 0.9, min_weight: float = 0.1) -> None:
        assert transform is None or callable(transform), "Given 'transforms' is not callable"
        self.transform = transform
        # the data dict holds values for each key, these values are typically a tuple of (raw img, label img)

        self._index_by_id = {}
        self._data: List[self._Entry] = []
        self._size = 0

        self.gamma = gamma
        self.min_weight = min_weight

    def __getitem__(self, index):
        entry = self._data[index]
        if entry.removed:
            raise EntryRemoved()

        entry.weight = max(self.min_weight, entry.weight * self.gamma)

        result = entry.data
        if self.transform is not None:
            result = self.transform(*result)

        return [torch.as_tensor(f, dtype=torch.float) for f in result]

    def __len__(self):
        return self._size

    def _update_or_create(self, id_, torch_image, torch_label) -> None:
        if id_ in self._index_by_id:
            idx = self._index_by_id[id_]
            entry = self._data[idx]
            entry.data = torch_image, torch_label
            entry.num_updates += 1
            entry.weight += 1.0

            if entry.removed:
                entry.removed = False
                self._size += 1

        else:
            new_entry = self._Entry(data=(torch_image, torch_label))
            new_idx = len(self._data)
            self._data.append(new_entry)
            self._index_by_id[id_] = new_idx
            self._size += 1

    def remove(self, id_):
        idx = self._index_by_id.get(id_)
        if idx is None:
            logger.warning("Trying to delete non existing key from dataset %s", id_)
            return

        self._data[idx].removed = True
        self._data[idx].weight = 0.0
        self._data[idx].data = None
        self._size -= 1

    def update(self, images: TikTensorBatch, labels: TikTensorBatch) -> None:
        """
        :param keys: list of keys to identify each value by
        :param values: list of new values. (remove from dataset if not bool(value). The count will be kept.)
        """
        if len(images) != len(labels):
            raise ValueError("images and labels should have length")

        for image, label in zip(images, labels):
            if image.id != label.id:
                raise ValueError(f"image(id={image.id}) and label(id={label.id}) should have same ids")

            numpy_image = image.as_numpy()
            numpy_label = label.as_numpy()
            print("UPDATING DATASET WITH", numpy_image.shape, numpy_label.shape)

            id_ = image.id

            self._update_or_create(id_, numpy_image, numpy_label)

    def get_weights(self) -> torch.DoubleTensor:
        return torch.DoubleTensor([e.weight for e in self._data])


class DynamicWeightedRandomSampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        dataset (DynamicDataset): providing get_weights method
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, dataset: DynamicDataset) -> None:
        self._dataset = dataset

    def __iter__(self):
        while True:
            if not len(self._dataset):
                raise EmptyDataset()

            weights = self._dataset.get_weights()
            val = torch.multinomial(weights, num_samples=1)
            yield int(val)

    def __len__(self):
        return len(self._dataset)
