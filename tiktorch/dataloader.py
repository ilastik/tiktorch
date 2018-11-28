import numpy as np
from torch.utils.data import Dataset, DataLoader
from inferno.io.transform import Compose
from inferno.io.transform.image import ElasticTransform, RandomFlip, RandomRotate
from inferno.io.transform.generic import AsTorchBatch, Normalize
from inferno.utils.io_utils import yaml2dict


class TikTorchDataset(Dataset):
    def __init__(self, raw_volume, labels, config=None):
        super(TikTorchDataset, self).__init__()
        self.raw_volume = raw_volume
        self.labels = labels
        self.transform_raw = self.get_transforms_raw()
        self.transform = self.get_transforms()

    def get_transforms_raw(self):
        return Compose(Normalize())

    def get_transforms(self):
        # TODO: data augmentation via config file
        transforms = Compose(RandomFlip(),
                             RandomRotate(),
                             ElasticTransform(alpha=2000., sigma=50.),
                             AsTorchBatch(2))
        return transforms

    def __getitem__(self, index):
        return self.transform(self.transform_raw(self.raw_volume[index][0, 0, :, :]),
                              self.labels[index][0, 0, :, :])

    def __len__(self):
        assert len(self.raw_volume) == len(self.labels)
        return len(self.raw_volume)


def get_dataloader(raw, labels, config=None):
    """
    Gets loaders given a the path to a configuration file.

    Parameters
    ----------
    config : str or dict
        (Path to) Data configuration.

    Returns
    -------
    torch.utils.data.dataloader.DataLoader
        Data loader built as configured.
    """
    #config = yaml2dict(config)
    dataset = TikTorchDataset(raw, labels)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader
