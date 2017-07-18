import torch
from torch.autograd import Variable


class WannabeConvNet3D(torch.nn.Module):
    """A torch model that pretends to be a 2D convolutional network.
    This exists to just test the pickling machinery."""
    def forward(self, input_):
        assert isinstance(input_, Variable)
        # Expecting 5 dimensional inputs as (NCDHW).
        assert input_.dim() == 4
        return input_
