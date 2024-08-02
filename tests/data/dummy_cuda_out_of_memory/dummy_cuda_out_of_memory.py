from torch import nn, Tensor
import numpy as np

MAX_SHAPE = (10, 10, 10, 10)


class Dummy(nn.Module):
    def forward(self, input: Tensor):
        input_size = np.prod(input.shape)
        max_size = np.prod(MAX_SHAPE)
        if input_size > max_size:
            raise RuntimeError("out of memory")
        return input + 1
