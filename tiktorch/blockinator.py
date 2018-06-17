import torch
from tiktorch.utils import DynamicShape


class Blockinator(object):
    def __init__(self, data, dynamic_shape):
        self.data = data
        self.dynamic_shape = dynamic_shape

    @property
    def block_shape(self):
        return self.dynamic_shape.base_shape

    @property
    def num_blocks(self):
        return tuple(shape//size for shape, size in zip(self.data.shape, self.block_shape))

    def get_slice(self, *block):
        return tuple(slice(_size * _block, _size * (_block + 1))
                     for _block, _size in zip(block, self.block_shape))

    def __getitem__(self, item):
        # Case: item is a slice object (i.e. slice along the first axis)
        if isinstance(item, slice):
            item = (item,) + (slice(0, None),) * (len(self.data.shape) - 1)

        if isinstance(item, tuple):
            if all(isinstance(_elem, int) for _elem in item):
                # Case: item a tuple ints
                return self.data[self.get_slice(*item)]
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
                return self.data[tuple(full_slice)]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


if __name__ == '__main__':
    dynamic_shape = DynamicShape('(32 * (nH + 1), 32 * (nW + 1))')
    block = Blockinator(torch.rand(256, 256), dynamic_shape)
    print(block.num_blocks)
    print(block.get_slice(0, 0))
    print(block[:-1].shape)