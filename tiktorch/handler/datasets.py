import itertools
import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from typing import Iterable


class DynamicDataset(Dataset):
    """
    A torch dataset based on a dict
    The indexed.py package might be a good candidate to do the indexing more efficiently.
    Comparison to the current generator implementation at: https://stackoverflow.com/a/47202816
    """

    def __init__(self, keys: Iterable = [], values: Iterable = [], transforms=None, gamma: float = .9) -> None:
        assert len(keys) == len(values)
        assert transforms is None or callable(transforms), "Given 'transforms' is not callable"
        self.transforms = transforms
        # the data dict holds values for each key, these values are typically a tuple of (raw img, label img)
        self.data = dict(zip(keys, values))
        # update counts keeps track of how many times a specific key has been added/updated
        self.weights = {key: 1. for key in keys}
        self.update_counts = {key: 1 for key in keys}
        self.removed_in_this_epoch = set()
        self.gamma = gamma

    def __getitem__(self, index):
        key, fetched = next(itertools.islice(self.data.items(), index, index + 1))
        self.weights[key] *= self.gamma

        if self.transforms is None:
            return fetched
        elif callable(self.transforms):
            return self.transforms(*fetched)
        else:
            raise RuntimeError

    def __len__(self):
        return len(self.data)

    def update(self, keys : Iterable, values : Iterable) -> None:
        """
        :param keys: list of keys to identify each value by
        :param values: list of new values. (remove from dataset if not bool(value). The count will be kept.)
        """
        # self.data.update(zip(keys, values))
        # update update counts
        for key, value in zip(keys, values):
            self.update_counts[key] = self.update_counts.get(key, 0) + 1
            # remove deleted samples (values)
            if value is None:
                if key in self.data.keys():
                    # del self.data[key]  # todo: make truly dynamic. problem: let sampler know
                    self.data[key] = tuple([*self.data[key][:-1], torch.zeros_like(self.data[key][-1])])
                    self.removed_in_this_epoch.add(key)
                    self.weights[key] = 0
            else:
                self.data[key] = value
                self.removed_in_this_epoch.discard(key)

                self.weights[key] = 1. # todo: take update counts into account

    def reset_indices(self) -> torch.Tensor:
        """
        Removes deleted samples from the dataset.
        :return: new weights
        """
        for deleted_key in self.removed_in_this_epoch:
            del self.data[deleted_key]

        return torch.DoubleTensor([self.weights[key] for key in self.data.keys()])
