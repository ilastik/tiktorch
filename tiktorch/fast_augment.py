import torch
import torch.nn.functional as F
import random
import logging


class AugmentationSuite(object):
    """Data Augmentation on CPU and GPU."""
    def __init__(self, normalize=True, random_flips=True, random_transpose=True,
                 random_rotate=True, elastic_transform=True, patch_ignore_labels=True,
                 allow_z_flips=False):
        self.do_normalize = normalize
        self.do_random_flips = random_flips
        self.do_random_transpose = random_transpose
        self.do_elastic_transform = elastic_transform
        self.do_random_rotate = random_rotate
        self.do_patch_ignore_labels = patch_ignore_labels
        self.allow_z_flips = allow_z_flips

    def normalize(self, data):
        data = F.batch_norm(data[None], None, None, None, None, True, 0.)[0]
        return data

    def random_flips(self, data, label):
        mode = random.choice(['ud', 'lr', 'z',
                              'udlr', 'udz', 'lrz',
                              'udlrz', '---'])
        if 'ud' in mode:
            # Negative step not supported as of yet in 0.4.1
            rev_idx = list(reversed(list(range(data.shape[-2]))))
            data = data[..., rev_idx, :]
            label = label[..., rev_idx, :]
        if 'lr' in mode:
            rev_idx = list(reversed(list(range(data.shape[-1]))))
            data = data[..., rev_idx]
            label = label[..., rev_idx]
        if 'z' in mode and self.allow_z_flips:
            rev_idx = list(reversed(list(range(data.shape[-3]))))
            data = data[..., rev_idx, :, :]
            label = label[..., rev_idx, :, :]
        return data, label

    def random_rotate(self, data, label):
        # TODO
        return data, label

    def random_transpose(self, data, label):
        toss = random.choice([True, False])
        if toss:
            data = data.transpose(-1, -2)
            label = label.transpose(-1, -2)
        return data, label

    def elastic_transform(self, data, label):
        # TODO Use grid_sample
        return data, label

    def patch_ignore_labels(self, label, copy=False):
        if copy:
            label = label.clone()
        # Obtain weight map
        weights = label.gt(0)
        # Label value 0 actually corresponds to Ignore. Subtract 1 from all pixels that will be
        # weighted to account for that
        label[weights] -= 1
        return label, weights.float()

    def __call__(self, data, label):
        logger = logging.getLogger('AugmentationSuite.__call__')
        init_data_shape = data.shape
        init_label_shape = label.shape
        with torch.no_grad():
            try:
                if self.do_normalize:
                    data = self.normalize(data)
                if self.do_random_flips:
                    data, label = self.random_flips(data, label)
                if self.do_random_transpose:
                    data, label = self.random_transpose(data, label)
                if self.do_random_rotate:
                    data, label = self.random_rotate(data, label)
                if self.do_elastic_transform:
                    data, label = self.elastic_transform(data, label)
                if self.do_patch_ignore_labels:
                    label, weights = self.patch_ignore_labels(label)
                else:
                    weights = None
            except Exception:
                logger.error(f"data.shape = {data.shape} (initially {init_data_shape}), "
                             f"label.shape = {label.shape} (initially {init_label_shape})")
                raise
        return data, label, weights


def test_augmentor():
    data = torch.rand(1, 100, 100)
    label = torch.randint(0, 2, (1, 100, 100))
    augmentor = AugmentationSuite()
    out = augmentor(data, label)


if __name__ == '__main__':
    test_augmentor()

