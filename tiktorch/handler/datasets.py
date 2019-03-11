import itertools


from torch.utils.data.dataset import Dataset
from typing import Iterable


class DynamicDataset(Dataset):
    """
    A torch dataset based on a dict
    The indexed.py package might be a good candidate to do the indexing more efficiently.
    Comparison to the current generator implementation at: https://stackoverflow.com/a/47202816
    """

    def __init__(self, keys: Iterable = [], values: Iterable = [], transforms=None) -> None:
        assert len(keys) == len(values)
        assert transforms is None or callable(transforms), "Given 'transforms' is not callable"
        self.transforms = transforms
        # the data dict holds values for each key, these values are typically a tuple of (raw img, label img)
        self.data = dict(zip(keys, values))
        # update counts keeps track of how many times a specific key has been added/updated
        self.update_counts = {key: 1 for key in keys}

    def __getitem__(self, index):
        # todo: take self.update_counts into account
        fetched = next(itertools.islice(self.data.values(), index, index + 1))

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
        self.data.update(zip(keys, values))
        # update update counts
        for key, value in zip(keys, values):
            self.update_counts[key] = self.update_counts.get(key, default=0) + 1
            # remove deleted samples (values)
            if not bool(value):
                del self.data[key]
