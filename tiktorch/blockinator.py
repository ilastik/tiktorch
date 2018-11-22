import torch
import numpy as np
import torch.nn.functional as thf
from tiktorch.utils import DynamicShape
from contextlib import contextmanager
from functools import reduce

class slicey(object):
    def __init__(self, start=None, stop=None, step=None, padding=(0, 0), shape=None):
        if shape is None:
            # Vanilla behaviour
            self.start = self.__istart = start
            self.stop = self.__istop = stop
            self.step = self.__istep = step
            self.padding = self.__ipadding = padding
            self.shape = self.__ishape = shape
        else:
            self.__istart = start
            self.__istop = stop
            self.__istep = step
            self.__ipadding = padding
            self.__ishape = shape
            # Compute real starts and stops
            start = 0 if start is None else start
            stop = shape if stop is None else stop
            # Add in padding
            start -= padding[0]
            stop += padding[1]
            padding = [0, 0]
            # Check if we ran out of volume bounds
            if start < 0:
                padding[0] = -start
                start = 0
            if stop > shape:
                padding[1] = stop - shape
                stop = shape
            # Set
            self.start = start
            self.stop = stop
            self.step = step
            self.padding = tuple(padding)
            self.shape = shape

    @classmethod
    def from_(cls, sl, padding=None, shape=None):
        if isinstance(sl, slice):
            return cls(sl.start, sl.stop, sl.step,
                       padding=((0, 0) if padding is None else padding),
                       shape=shape)
        elif isinstance(sl, slicey):
            return sl
        else:
            raise TypeError

    @property
    def slice(self):
        # noinspection PyTypeChecker
        return slice(self.start, self.stop, self.step)

    @property
    def islice(self):
        return slice(self.__istart, self.__istop, self.__istep)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.start}:{self.stop}/{self.shape}:{self.step} + " \
               f"{self.padding})"


class Blockinator(object):
    def __init__(self, data, base_shape, num_channel_axes=0,
                 pad_fn=(lambda tensor, padding: tensor)):
        """
        Parameters
        ----------
        num_channel_axes: int
        Specifies the position of the channel axis in data (tensor).
        E.g. data.shape == (N, C, H, W) then num_channel_axes = 2.
        """
        # Privates
        self._processor = None
        # Publics
        self.data = data
        self.num_channel_axes = num_channel_axes
        self.base_shape = base_shape
        self.pad_fn = pad_fn

    @property
    def block_shape(self):
        return self.base_shape

    @property
    def spatial_shape(self):
        return self.data.shape[self.num_channel_axes:]

    @property
    def num_blocks(self):
        return tuple(shape//size for shape, size in zip(self.spatial_shape, self.block_shape))

    def get_slice(self, *block):
        return tuple(slice(_size * _block, _size * (_block + 1))
                     for _block, _size in zip(block, self.block_shape))

    def space_cake(self, *slices):
        # This function slices the data, and adds a halo if requested.
        # Convert all slice to sliceys
        slices = [slicey.from_(sl) for sl in slices]
        # Pad out-of-array values
        # Get unpadded volume
        unpadded_volume = self.data[tuple(slice(0, None) for _ in range(self.num_channel_axes)) +
                                    tuple(sl.slice for sl in slices)]
        padding = [None] * self.num_channel_axes + [sl.padding for sl in slices]
        padded_volume = self.pad_fn(unpadded_volume, padding)
        return padded_volume

    def fetch(self, item):
        # Case: item is a slice object (i.e. slice along the first axis)
        if isinstance(item, slice):
            item = (item,) + (slice(0, None),) * (len(self.spatial_shape) - 1)

        if isinstance(item, tuple):
            if all(isinstance(_elem, int) for _elem in item):
                # Case: item a tuple ints
                full_slice = self.get_slice(*item)
            elif all(isinstance(_elem, slice) for _elem in item):
                # Case: item is a tuple of slices
                # Define helper functions
                def _process_starts(start, num_blocks):
                    if start is None:
                        return 0
                    elif start >= 0:
                        return start
                    else:
                        return num_blocks + start

                def _process_stops(stop, num_blocks):
                    if stop is None:
                        return num_blocks - 1
                    elif stop > 0:
                        return stop - 1
                    else:
                        return num_blocks + stop - 1

                # Get the full slice
                starts = [_process_starts(_sl.start, _num_blocks)
                          for _sl, _num_blocks in zip(item, self.num_blocks)]
                stops = [_process_stops(_sl.stop, _num_blocks)
                         for _sl, _num_blocks in zip(item, self.num_blocks)]
                slice_starts = [_sl.start for _sl in self.get_slice(*starts)]
                slice_stops = [_sl.stop for _sl in self.get_slice(*stops)]
                full_slice = [slice(starts, stops)
                              for starts, stops in zip(slice_starts, slice_stops)]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # Time to throw in the halo. Check if a processor is attached
        if self._processor is not None and hasattr(self._processor, 'halo'):
            halo = self._processor.halo
        else:
            halo = None
        if halo is not None:
            assert len(halo) == len(self.spatial_shape)
            # Compute halo in units of block size
            num_halo_blocks = [int(np.ceil(_halo / _block_shape))
                               for _halo, _block_shape in zip(halo, self.block_shape)]
            spatial_padding = [(_num_halo_blocks * _block_shape,) * 2
                               for _num_halo_blocks, _block_shape in zip(num_halo_blocks,
                                                                         self.block_shape)]
            sliceys = [slicey.from_(_sl, _padding, _shape)
                       for _sl, _padding, _shape in zip(full_slice, spatial_padding,
                                                        self.spatial_shape)]
            sliced = self.space_cake(*sliceys)
        else:
            sliced = self.space_cake(*full_slice)
        return sliced

    def __getitem__(self, item):
        return self.fetch(item)

    def process(self):
        # try to process the whole thing at once
        model = self._processor.model
        device = self._processor.device
        try:
            print(device)
            print(self.data.shape)
            with torch.no_grad():
                output_tensor = model.to(device)(self.data.to(device)).cpu()
            return self._processor.crop_halo(output_tensor)
        except:
            RuntimeError("Tensor could not be processed at once. Processing blockwise....")

        # if it does not work, process it blockwise
        halo = self._processor.halo_in_blocks
        output_tensor = torch.empty_like(self.data).cpu()
        
        for i in range(halo[0], self.num_blocks[0] - halo[0]):
            for j in range(halo[1], self.num_blocks[1] - halo[1]):
                if len(self.num_blocks) == 3:
                    for k in range(halo[2], self.num_blocks[2] - halo[2]):
                        with torch.no_grad():
                            out = model(self[i, j, k].to(device)).cpu()
                        out = self._processor.crop_halo(out, self.num_channel_axes)
                        output_tensor[[slice(None)] * self.num_channel_axes + [sl for sl in self.get_slice(i, j, k)]] = out
                else:
                    with torch.no_grad():
                        out = model(self[i, j].to(device)).cpu()
                        out = self._processor.crop_halo(out, self.num_channel_axes)
                        output_tensor[[slice(None)] * self.num_channel_axes + [sl for sl in self.get_slice(i, j)]] = out
        return self._processor.crop_output_tensor(output_tensor, self.num_channel_axes)

    @property
    def processor(self):
        assert self._processor is not None
        self._processor

    @contextmanager
    def attach(self, processor):
        self._processor = processor
        yield
        self._processor = None


def np_pad(x, padding):
    np_padding = []
    for _dim_pad in padding:
        if _dim_pad is None:
            np_padding.append((0,0))
        else:
            np_padding.append(_dim_pad)
    return np.pad(x, np_padding, mode='reflect')


def th_pad(x, padding):
    """
    Parameters
    ----------
    x: torch.Tensor
    """
    torch_padding = []
    for _dim_pad in reversed(padding):
        if _dim_pad is not None:
            torch_padding.extend(_dim_pad)
    # pytorch 0.4.1 only implements reflect padding for tensors of shape (N, C, H, W) (or smaller)
    return thf.pad(x, torch_padding, mode='reflect') if x.dim() < 5 else thf.pad(x, torch_padding)


def _test_blocky_basic():
    dynamic_shape = DynamicShape('(32 * (nH + 1), 32 * (nW + 1))')
    block = Blockinator(torch.rand(256, 256), dynamic_shape)
    assert block.num_blocks == (8, 8)
    assert block.get_slice(0, 0) == (slice(0, 32, None), slice(0, 32, None))
    assert list(block[:-1].shape) == [224, 256]


def _test_blocky_halo():
    from argparse import Namespace
    dynamic_shape = DynamicShape('(32 * (nH + 1), 32 * (nW + 1))')
    block = Blockinator(torch.rand(256, 256), dynamic_shape)
    processor = Namespace(halo=[4, 4])
    with block.attach(processor):
        out = block[1:3, 2:4]
    print(out.shape)

def _test_pad_function():
    t4d = torch.rand(1, 1, 32, 32)
    n4d = np.random.rand(1, 1, 32, 32)
    p4d = (None, None, (4, 4), (9, 9))
    assert th_pad(x=t4d, padding=p4d).shape == np_pad(n4d, p4d).shape

    t5d = torch.rand(1, 3, 10, 64, 64)
    n5d = np.random.rand(1, 3, 10, 64, 64)
    p5d = (None, None, (5, 5), (8, 8), (16, 16))
    assert th_pad(t5d, p5d).shape == np_pad(n5d, p5d).shape

if __name__ == '__main__':
    #dynamic_shape = DynamicShape('(32 * (nH + 1), 32 * (nW + 1))')
    #block = Blockinator(torch.rand(256, 256), dynamic_shape)
    #for i in range(block.num_blocks[0]):
    #    for j in range(block.num_blocks[1]):
    #        print(block[i, j].shape, block.get_slice(i, j))
    _test_blocky_halo()
    _test_pad_function()
