import itertools


from torch.utils.data.dataset import Dataset
from typing import Iterable


class DynamicDataset(Dataset):
    """
    A torch dataset based on a dict
    The indexed.py package might be a good candidate to do the indexing more efficiently.
    Comparison to the current generator implementation at: https://stackoverflow.com/a/47202816
    """

    def __init__(self, keys: Iterable = [], values: Iterable = [], transfroms=None) -> None:
        assert len(keys) == len(values)
        assert transfroms is None or callable(transfroms), "Given 'transforms' is not callable"
        self.data = dict(zip(keys, values))

    def __getitem__(self, index):
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
        self.data.update(zip(keys, values))
