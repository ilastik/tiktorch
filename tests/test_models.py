from tiktorch.models.dunet import DUNet
import torch
from torch.autograd import Variable
from unittest import TestCase


class TestModels(TestCase):
    def test_dunet(self):
        # Make network
        model = DUNet(1, 1)
        # Input variable
        input_variable = Variable(torch.rand(1, 1, 512, 512))
        # noinspection PyCallingNonCallable
        output_variable = model(input_variable)
        self.assertEqual(list(output_variable.data.size()), [1, 1, 512, 512])


if __name__ == '__main__':
    TestModels().test_dunet()
