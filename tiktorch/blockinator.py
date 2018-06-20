import torch
import numpy as np
import torch.nn.functional as thf
from tiktorch.utils import DynamicShape
from contextlib import contextmanager

thf.pad()

class slicey(object):
    def __init__(self, start=None, stop=None, step=None, padding=(0, 0)):
        self.start = start,
        self.stop = stop
        self.step = step
        self.padding = padding

    @classmethod
    def from_(cls, sl):
        if isinstance(sl, slice):
            return cls(sl.start, sl.stop, sl.step)
        elif isinstance(sl, slicey):
            return slicey
        else:
            raise TypeError

    @property
    def slice(self):
        # noinspection PyTypeChecker
        return slice(self.start, self.stop, self.step)


class Blockinator(object):
    def __init__(self, data, dynamic_shape, num_channel_axes=0, pad_fn=None):
        # Privates
        self._processor = None
        # Publics
        self.data = data
        self.num_channel_axes = num_channel_axes
        self.dynamic_shape = dynamic_shape
        self.pad_fn = pad_fn

    @property
    def block_shape(self):
        return self.dynamic_shape.base_shape

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
        # TODO Pad out-of-array values
        return self.data[tuple(slice(0, None) for _ in range(self.num_channel_axes)) +
                         tuple(sl.slice for sl in slices)]

    def fetch(self, item):
        # Case: item is a slice object (i.e. slice along the first axis)
        if isinstance(item, slice):
            item = (item,) + (slice(0, None),) * (len(self.spatial_shape) - 1)

        if isinstance(item, tuple):
            if all(isinstance(_elem, int) for _elem in item):
                # Case: item a tuple ints
                sliced = self.space_cake(*self.get_slice(*item))
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
                sliced = self.space_cake(*full_slice)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return sliced

    def __getitem__(self, item):
        return self.fetch(item)

    def request_halo(self, spatial_slice):
        pass

    @contextmanager
    def attach(self, processor):
        self._processor = processor
        yield
        self._processor = None


def np_pad(x, padding):
    return np.pad(x, padding, mode='reflect')


def th_pad(x, padding):
    torch_padding = []
    for _dim_pad in padding:
        pass
    # TODO


if __name__ == '__main__':
    dynamic_shape = DynamicShape('(32 * (nH + 1), 32 * (nW + 1))')
    block = Blockinator(torch.rand(256, 256), dynamic_shape)
    print(block.num_blocks)
    print(block.get_slice(0, 0))
    print(block[:-1].shape)