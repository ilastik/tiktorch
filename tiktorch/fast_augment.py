import torch
import torch.nn.functional as F
import random
import logging
import numpy as np
from inferno.io.transform.image import ElasticTransform


class AugmentationSuite(object):
    """Native Data Augmentation on CPU and GPU."""

    # This is not debugged yet :-(
    USE_NATIVE_ELASTIC_TRAFO = False

    def __init__(self, normalize=True, random_flips=True, random_transpose=True,
                 random_rotate=True, elastic_transform=True, patch_ignore_labels=True,
                 allow_z_flips=False, elastic_transform_scale=2000, elastic_transform_sigma=50,
                 elastic_transform_kernel_size=None, invert_binary_labels=False):
        self.do_normalize = normalize
        self.do_random_flips = random_flips
        self.do_random_transpose = random_transpose
        self.do_elastic_transform = elastic_transform
        self.do_random_rotate = random_rotate
        self.do_patch_ignore_labels = patch_ignore_labels
        self.allow_z_flips = allow_z_flips
        self.flow_scale = elastic_transform_scale
        self.elastic_transform_scale = elastic_transform_scale
        self.elastic_transform_sigma = elastic_transform_sigma
        self.invert_binary_labels = invert_binary_labels
        if elastic_transform_kernel_size is None:
            self.elastic_transform_kernel_size = 2 * self.elastic_transform_sigma + 1
        else:
            self.elastic_transform_kernel_size = elastic_transform_kernel_size
        self._elastic_transformer = ElasticTransform(self.elastic_transform_scale,
                                                     self.elastic_transform_sigma)
        # Privates
        self._gaussian_kernels = {}

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
        if self.USE_NATIVE_ELASTIC_TRAFO:
            data, perturbation = self._random_flow(data, sigma=self.elastic_transform_sigma,
                                                   kernel_size=self.elastic_transform_kernel_size,
                                                   scale=self.elastic_transform_scale)
            label, _ = self._random_flow(label, sigma=self.elastic_transform_sigma,
                                         scale=self.elastic_transform_scale,
                                         kernel_size=self.elastic_transform_kernel_size,
                                         perturbation=perturbation)
        else:
            data_np = data.cpu().numpy()
            label_np = label.cpu().numpy()
            data_np, label_np = self._elastic_transformer(data_np, label_np)
            data = torch.from_numpy(data_np).type_as(data)
            label = torch.from_numpy(label_np).type_as(label)
        return data, label

    def patch_ignore_labels(self, label, copy=False):
        if copy:
            label = label.clone()
        # Obtain weight map
        weights = label.gt(0)
        # Label value 0 actually corresponds to Ignore. Subtract 1 from all pixels that will be
        # weighted to account for that
        label[weights] -= 1
        if self.invert_binary_labels:
            label[weights] = 1 - label[weights]
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
                if self.do_patch_ignore_labels:
                    label, weights = self.patch_ignore_labels(label)
                else:
                    weights = None
                if self.do_elastic_transform:
                    data, label = self.elastic_transform(data, label)
            except Exception:
                logger.error(f"data.shape = {data.shape} (initially {init_data_shape}), "
                             f"label.shape = {label.shape} (initially {init_label_shape})")
                raise
        return data, label, weights

    def _gaussian_smoothing_2d(self, tensor, sigma, kernel_size):
        # tensor.shape = (N, C, H, W)
        if len(tensor.shape) == 4:
            n, c, h, w = tensor.shape
            d = None
        elif len(tensor.shape) == 5:
            n, c, d, h, w = tensor.shape
        else:
            raise NotImplementedError
        # TODO: Make this faster with separable convolutions
        if self._gaussian_kernels.get((sigma, kernel_size)) is None:
            # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
            x_cord = torch.arange(kernel_size, device=tensor.device)
            x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

            mean = (kernel_size - 1) / 2.
            variance = sigma ** 2.

            # Calculate the 2-dimensional gaussian kernel which is
            # the product of two gaussian distributions for two different
            # variables (in this case called x and y)
            gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                              torch.exp(
                                  -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                                  (2 * variance)
                              )
            # Make sure sum of values in gaussian kernel equals 1.
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            self._gaussian_kernels[sigma, kernel_size] = gaussian_kernel
        gaussian_kernel = self._gaussian_kernels[sigma, kernel_size]
        # Reshape to 2/3d depthwise convolutional weight
        if d is None:
            # 2D
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
            gaussian_kernel = gaussian_kernel.repeat(c, 1, 1, 1)
            conv = F.conv2d
            padding = (kernel_size // 2, kernel_size // 2)
        else:
            # 3D
            gaussian_kernel = gaussian_kernel.view(1, 1, 1, kernel_size, kernel_size)
            gaussian_kernel = gaussian_kernel.repeat(c, 1, 1, 1, 1)
            conv = F.conv3d
            padding = (0, kernel_size // 2, kernel_size // 2)
        # Smooth
        smoothed_image = conv(tensor, gaussian_kernel, padding=padding, groups=c)
        return smoothed_image

    def _smooth_random_image(self, like, num_images, sigma, kernel_size):
        assert like.dim() == 3, "Only 2D supported for now."
        # Generate a random NCHW tensor scaled between [-1, 1] (modulo smoothing)
        rand_image = torch.rand((num_images, 1, like.shape[-2], like.shape[-1]),
                                device=like.device).sub_(0.5).mul_(2.)
        # Smooth
        smoothed_image = self._gaussian_smoothing_2d(rand_image, sigma, kernel_size)
        # Squeeze out the channel and return (NHW)
        return smoothed_image.squeeze(1)

    def _random_flow(self, image, sigma, kernel_size, scale, perturbation=None):
        # image is chw or cdhw. Make corresponding flow vectors.
        init_image_shape = image.shape
        # TODO 3D support
        if image.dim() == 4:
            assert image.shape[1] == 1, "Only 2D supported for now."
            image = image.squeeze(1)
        # Image is now CHW
        ii, jj = np.meshgrid(np.linspace(-1, 1, image.shape[-2], dtype='float32'),
                             np.linspace(-1, 1, image.shape[-1], dtype='float32'))
        ii, jj = map(torch.from_numpy, [ii, jj])
        # The grid is 2HW
        ij_grid = torch.stack([ii, jj], dim=0).float()
        # Perturb grid with a random field (2HW)
        if perturbation is None:
            perturbation = self._smooth_random_image(image, num_images=2, sigma=sigma,
                                                     kernel_size=kernel_size)
        perturbed_grid = ij_grid + (scale * perturbation)
        # Reshape to 1HW2
        perturbed_grid = perturbed_grid.transpose(0, -1)[None]
        # transformed image is CHW again
        transformed_image = F.grid_sample(image[None], perturbed_grid)[0]
        # Reshape to the init shape
        transformed_image = transformed_image.view(*init_image_shape)
        # Done
        return transformed_image, perturbation


def test_augmentor():
    data = torch.rand(1, 1, 100, 100)
    label = torch.randint(0, 2, (1, 1, 100, 100))
    augmentor = AugmentationSuite()
    out_data, out_label, out_weight = augmentor(data, label)
    assert out_data.shape == (1, 1, 100, 100)
    assert out_label.shape == (1, 1, 100, 100)
    assert out_weight.shape == (1, 1, 100, 100)


if __name__ == '__main__':
    test_augmentor()

