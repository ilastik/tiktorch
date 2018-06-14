import numpy as np
import torch

from . import utils


class TikIO(object):

    class ShapeError(Exception):
        pass

    def __init__(self):
        self._tensors = None

    @staticmethod
    def to_torch_tensors(tensors):
        torch_tensors = []
        for tensor_num, tensor in enumerate(tensors):
            if isinstance(tensor, np.ndarray):
                torch_tensor = torch.from_numpy(tensor)
            elif isinstance(tensor, torch.Tensor):
                torch_tensor = tensor
            else:
                raise TypeError(f"Tensor at position must be numpy.ndarray or "
                                f"torch.Tensors, got {type(tensor)} instead.")
            torch_tensors.append(torch_tensor)
        return torch_tensors

    @property
    def tensors(self):
        utils.assert_(self._tensors is not None,
                      "Trying to acess `TikIn.tensors` with it yet to be defined.",
                      ValueError)
        return self._tensors

    @tensors.setter
    def tensors(self, value):
        self._tensors = self.to_torch_tensors(self.validate_shape(value))

    @staticmethod
    def validate_shape(tensors):
        utils.assert_([tensor.shape == tensors[0].shape for tensor in tensors],
                      f"Input `tensors` to TikIn must all have the same shape. "
                      f"Got tensors of shape: {[tensor.shape for tensor in tensors]}",
                      TikIn.ShapeError)
        utils.assert_(len(tensors[0].shape) in [2, 3, 4], f"Tensors must be of dimensions "
                                                          f"2 (HW), 3 (CHW/DHw), or 4 (CDHW). "
                                                          f"Got {len(tensors[0].shape)} instead.",
                      TikIO.ShapeError)
        return tensors


class TikIn(TikIO):

    def __init__(self, tensors=None):
        """
        Input Object for TikTorch.

        Parameters
        ----------
        tensors: list of numpy.ndarray or list of torch.Tensor
            Input Batch to TikTorch. Must be a list of tensors of the same shapes in any of the
            following formats:
            (C = Channel, H = Height, W = Width, D = Depth)
                a. (C, H, W)
                b. (C, D, H, W)
                c. (H, W)
                d. (D, H, W)
            C defaults to 1 if not provided. To resolve between (a) and (d), TikTorch uses
            the `input_shape` provided for model at build time.
        """
        super(TikIn, self).__init__()
        if tensors is not None:
            self.tensors = tensors

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def format(self):
        return {2: 'HW', 3: 'CHW/DHW', 4: 'CDHW'}[len(self.shape)]

    def reshape(self, *shape):
        return [tensor.reshape(*shape) for tensor in self.tensors]

    # MUSHROOMS MUUSHROOMS!
    def batcher(self, network_input_shape: list):
        """
        Build batch for the network.

        Parameters
        ----------
        network_input_shape: list
            Input shape to the network.

        Returns
        -------
        torch.Tensor
        """
        network_input_format = {3: 'CHW', 4: 'CDHW'}[len(network_input_shape)]
        if self.format == 'HW':
            utils.assert_(network_input_format == 'CHW',
                          f"Input format is HW, which is not compatible with the network "
                          f"input format {network_input_format} (must be CHW).",
                          TikIn.ShapeError)
            utils.assert_(network_input_shape[0] == 1,
                          f"Input format is HW, for which the number of input channels (C)"
                          f"to the network must be 1. Got C = {network_input_shape[0]} instead.",
                          TikIn.ShapeError)
            pre_cat = self.reshape(1, *self.shape)
        elif self.format == 'CDHW':
            utils.assert_(network_input_format == 'CDHW',
                          f"Input format (CDHW) is not compatible with network input format "
                          f"({network_input_format}).",
                          TikIn.ShapeError)
            utils.assert_(self.shape[0] == network_input_shape[0],
                          f"Number of input channels in input ({self.shape[0]}) is not "
                          f"consistent with what the network expects ({network_input_shape[0]}).",
                          TikIn.ShapeError)
            pre_cat = self.tensors
        elif self.format == 'CHW/DHW':
            if network_input_format == 'CHW':
                # input format is CHW
                utils.assert_(self.shape[0] == network_input_shape[0],
                              f"Number of input channels in input ({self.shape[0]}) is "
                              f"not compatible with the number of input channels to "
                              f"the network ({network_input_shape[0]})",
                              TikIn.ShapeError)
                pre_cat = self.tensors
            elif network_input_format == 'CDHW':
                utils.assert_(network_input_shape[0] == 1,
                              f"Input format DHW requires that the number of input channels (C) "
                              f"to the network is 1. Got C = {network_input_shape[0]} instead.",
                              TikIn.ShapeError)
                pre_cat = self.reshape(1, *self.shape)
            else:
                raise TikIn.ShapeError(f"Input format {self.format} is not compatible with "
                                       f"the network input format {network_input_format}.")
        else:
            raise ValueError("Internal Error: Invalid Format.")
        # Concatenate to a batch
        batch = torch.stack(pre_cat, dim=0)
        return batch


class TikOut(TikIO):
    def __init__(self, batch):
        super(TikOut, self).__init__()
        self.tensors = self.unbatcher(batch)

    def unbatcher(self, batch):
        utils.assert_(len(batch.shape) in [4, 5], f"`batch` must either be a NCHW or NCDHW tensor, "
                                                  f"got one with dimension {len(batch.shape)} "
                                                  f"instead.",
                      TikIO.ShapeError)
        return list(batch)