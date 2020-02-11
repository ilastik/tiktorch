from torch import nn


class Dummy(nn.Module):
    def forward(self, input):
        x = input
        return x + 1
