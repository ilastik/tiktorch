import signal
from importlib import util as imputils
from contextlib import contextmanager
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


def define_patched_model(model_file_name, model_class_name, model_init_kwargs):
    # Dynamically import file.
    module_spec = imputils.spec_from_file_location('model', model_file_name)
    module = imputils.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # Build model from file
    model: torch.nn.Module = \
        getattr(module, model_class_name)(**model_init_kwargs)
    # Monkey patch
    model._model_file_name = model_file_name
    model._model_class_name = model_class_name
    model._model_init_kwargs = model_init_kwargs
    return model


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


class BinaryTree:

    class Node:
        def __init__(self, key, data):
            self.key = key
            self.data = data
            self.left = None
            self.right = None

        def __str__(self):
            return f"({self.key}: {self.data})"

        def printTree(self):
            if self.left:
                self.left.printTree()
            print(self)
            if self.right:
                self.right.printTree()

    def __init__(self, input_data: dict=None):
        self._model = None
        self.root = None
        if input_data:
            self._build(input_data)

    @property
    def model(self):
        return self._model
    
    @contextmanager
    def attach(self, model):
        self._model = model
        yield
        self._model = None

    def _build(self, inputs):
        k, v = inputs.popitem()
        self.root = self.Node(k, v)
        while inputs:
            k, v = inputs.popitem()
            self.insert(self.Node(k, v))

    def insert(self, z):
        y = None
        x = self.root
        while x is not None:
            y = x
            x = x.left if z.key < x.key else x.right
        if y is None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

    def search(self):
        assert self.model is not None
        if not self.root:
            raise Exception("Tree is empty!")
        # traverse tree
        def recursive_search(x, last_valid_node=None):
            if x is None:
                return
            try:
                # data to torch.tensor
                _input = x.data + [x.data[-1]]
                _input = torch.zeros(1, 1, *_input)
                _out = self.model(_input)

                if last_valid_node:
                    if last_valid_node.key < x.key:
                        last_valid_node = x
                else:
                    last_valid_node = x
                if x.right:
                    return recursive_search(x.right, last_valid_node)
                else:
                    return last_valid_node
            except RuntimeError:
                if x.left:
                    return recursive_search(x.left, last_valid_node)
                else:
                    return last_valid_node

        return recursive_search(self.root)


def test_binary_search_tree():
    import torch.nn as nn
    inputs = [[6, 32], [6, 112], [6, 272], [9, 32], [9, 192],
              [12, 32], [12, 432], [15, 256], [150, 2000]]
    inputs_dict = {}
    for entry in inputs:
        key = entry[0]*entry[1]
        inputs_dict.update({key: entry})

    tree = BinaryTree(inputs_dict)
    model = nn.Sequential(nn.Conv3d(1, 12, 3),
                          nn.Sigmoid(),
                          nn.Conv3d(12, 24, 3),
                          nn.ReLU())

    with tree.attach(model):
        print('Tree:')
        tree.root.printTree()
        print('------------Searching for optimal shape------------------')
        optimalNode = tree.search()
        optimalShape = optimalNode.data + [optimalNode.data[-1]]
        print(f'Search result: {optimalShape}')


if __name__ == '__main__':
    test_binary_search_tree()
