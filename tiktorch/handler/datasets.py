import itertools
import torch

from collections import OrderedDict
from torch.utils.data.dataset import Dataset

from tiktorch.tiktypes import LabeledTikTensorBatch, TikTensorBatch


class DynamicDataset(Dataset):
    """
    A torch dataset based on a dict
    The indexed.py package might be a good candidate to do the indexing more efficiently.
    Comparison to the current generator implementation at: https://stackoverflow.com/a/47202816
    """

    def __init__(self, data: LabeledTikTensorBatch = None, transform=None, gamma: float = 0.9) -> None:
        assert transform is None or callable(transform), "Given 'transforms' is not callable"
        self.transform = transform
        # the data dict holds values for each key, these values are typically a tuple of (raw img, label img)
        if data is None:
            data = LabeledTikTensorBatch([])

        self.data = OrderedDict(zip(data.ids, data.as_torch()))
        # update counts keeps track of how many times a specific key has been added/updated
        self.weights = {key: 1.0 for key in data.ids}
        self.update_counts = {key: 1 for key in data.ids}
        self.recently_removed = set()
        self.gamma = gamma

    def __getitem__(self, index, update_weight=True):
        key, fetched = next(itertools.islice(self.data.items(), index, index + 1))
        if update_weight:
            self.weights[key] *= self.gamma

        if self.transform is None:
            return fetched
        else:
            return self.transform(*fetched)

    def __len__(self):
        return len(self.data)

    def update(self, images: TikTensorBatch, labels: TikTensorBatch) -> None:
        """
        :param keys: list of keys to identify each value by
        :param values: list of new values. (remove from dataset if not bool(value). The count will be kept.)
        """
        # self.data.update(zip(keys, values))
        # update update counts
        for image, label in zip(images, labels):
            assert image.id == label.id
            key = image.id
            self.update_counts[key] = self.update_counts.get(key, 0) + 1
            image = image.as_torch()
            label = label.as_torch()

            if label.any():
                # add sample-label pair to dataset
                self.data[key] = image, label
                self.recently_removed.discard(key)
                self.weights[key] = self.weights.get(key, 0) + 1  # todo: take update counts into account properly
            elif key in self.data.keys():
                # flag to remove sample-label pair from dataset
                # del self.data[key]  # todo: make truly dynamic. problem: let sampler know
                self.weights[key] = 0
                self.recently_removed.add(key)

                # the following line is obsolete. It's put here for safety (better to train on an empty patch than on
                # an old label, if this patch somehow gets sampled anyway somehow) todo: remove after testing
                self.data[key] = image, label

    def reset_indices(self) -> torch.Tensor:
        """
        Removes deleted samples from the dataset.
        :return: new weights
        """
        for deleted_key in self.recently_removed:
            del self.data[deleted_key]
            del self.weights[deleted_key]

        return torch.DoubleTensor([self.weights[key] for key in self.data.keys()])
