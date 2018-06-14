import torch


class Blockinator(object):
    def __init__(self, data, dynamic_shape):
        self.data = data
        self.dynamic_shape = dynamic_shape

    @property
    def block_size(self):
        return self.dynamic_shape.base_shape
