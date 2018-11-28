import signal
import torch
from torch.autograd import Variable


class delayed_keyboard_interrupt(object):
    """
    Delays SIGINT over critical code.
    Borrowed from:
    https://stackoverflow.com/questions/842557/
    how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py
    """
    # PEP8: Context manager class in lowercase
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def assert_(condition, message='', exception_type=Exception):
    if not condition:
        raise exception_type(message)


def to_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x]


class WannabeConvNet3D(torch.nn.Module):
    """A torch model that pretends to be a 2D convolutional network.
    This exists to just test the pickling machinery."""
    def forward(self, input_):
        assert isinstance(input_, Variable)
        # Expecting 5 dimensional inputs as (NCDHW).
        assert input_.dim() == 5
        return input_


class TinyConvNet3D(torch.nn.Module):
    """Tiny ConvNet with actual parameters."""
    def __init__(self, num_input_channels=1, num_output_channels=1):
        super(TinyConvNet3D, self).__init__()
        self.conv3d = torch.nn.Conv3d(num_input_channels, num_output_channels, 3, padding=1)

    def forward(self, *input):
        return self.conv3d(input[0])


class TinyConvNet2D(torch.nn.Module):
    """Tiny ConvNet with actual parameters."""
    def __init__(self, num_input_channels=1, num_output_channels=1):
        super(TinyConvNet2D, self).__init__()
        self.conv2d = torch.nn.Conv2d(num_input_channels, num_output_channels, 3, padding=1)

    def forward(self, *input):
        return self.conv2d(input[0])


class DynamicShape(object):
    def __init__(self, code):
        self.code = code
        self.formatters = self.strip_to_components(code)

    @property
    def dimension(self):
        return len(self)

    def __len__(self):
        return len(self.formatters)

    @staticmethod
    def strip_to_components(code):
        components = code[1:-1].replace(', ', ',').split(',')
        # Replace components with lambdas
        fmt_strings = [component.replace('nH', '{}').replace('nW', '{}').replace('nD', '{}')
                       for component in components]
        return fmt_strings

    def evaluate(self, *integers):
        return [eval(formatter.format(integer), {}, {})
                for integer, formatter in zip(integers, self.formatters)]

    @property
    def base_shape(self):
        return self.evaluate(*([0]*len(self)))

    __call__ = evaluate