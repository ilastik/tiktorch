import torch
import numpy as np
import torch.nn as nn
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D


class Xcoder(nn.Module):
    def __init__(self, previous_in_channels, out_channels, kernel_size, pre_output):
        super(Xcoder, self).__init__()
        assert out_channels % 2 == 0
        self.in_channels = sum(previous_in_channels)
        self.out_channels = out_channels
        self.conv1 = ConvELU2D(
            in_channels=self.in_channels, out_channels=self.out_channels // 2, kernel_size=kernel_size
        )
        self.conv2 = ConvELU2D(
            in_channels=self.in_channels + (self.out_channels // 2),
            out_channels=self.out_channels // 2,
            kernel_size=kernel_size,
        )
        self.pre_output = pre_output

    # noinspection PyCallingNonCallable
    def forward(self, input_):
        conv1_out = self.conv1(input_)
        conv2_inp = torch.cat((input_, conv1_out), 1)
        conv2_out = self.conv2(conv2_inp)
        conv_out = torch.cat((conv1_out, conv2_out), 1)
        if self.pre_output is not None:
            out = self.pre_output(conv_out)
        else:
            out = conv_out
        return out


class Encoder(Xcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size):
        super(Encoder, self).__init__(
            previous_in_channels, out_channels, kernel_size, pre_output=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


class Decoder(Xcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size):
        super(Decoder, self).__init__(
            previous_in_channels, out_channels, kernel_size, pre_output=nn.UpsamplingNearest2d(scale_factor=2)
        )


class Base(Xcoder):
    def __init__(self, previous_in_channels, out_channels, kernel_size):
        super(Base, self).__init__(previous_in_channels, out_channels, kernel_size, pre_output=None)


class Output(Conv2D):
    def __init__(self, previous_in_channels, out_channels, kernel_size):
        self.in_channels = sum(previous_in_channels)
        self.out_channels = out_channels
        super(Output, self).__init__(self.in_channels, self.out_channels, kernel_size)


class DUNetSkeleton(nn.Module):
    def __init__(self, encoders, decoders, base, output, final_activation=None, return_hypercolumns=False):
        super(DUNetSkeleton, self).__init__()
        assert isinstance(encoders, list)
        assert isinstance(decoders, list)
        assert len(encoders) == len(decoders) == 3
        assert isinstance(base, nn.Module)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.poolx2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.poolx4 = nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
        self.poolx8 = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
        self.upx2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upx4 = nn.UpsamplingNearest2d(scale_factor=4)
        self.upx8 = nn.UpsamplingNearest2d(scale_factor=8)
        self.base = base
        self.output = output
        self.final_activation = final_activation
        self.return_hypercolumns = return_hypercolumns

    def forward(self, input_):
        # Say input_ spatial size is 512, i.e. input_.ssize = 512
        # Pre-pool to save computation
        input_2ds = self.poolx2(input_)
        input_4ds = self.poolx4(input_)
        input_8ds = self.poolx8(input_)

        # e0.ssize = 256
        e0 = self.encoders[0](input_)
        e0_2ds = self.poolx2(e0)
        e0_4ds = self.poolx4(e0)
        e0_2us = self.upx2(e0)

        # e1.ssize = 128
        e1 = self.encoders[1](torch.cat((input_2ds, e0), 1))
        e1_2ds = self.poolx2(e1)
        e1_2us = self.upx2(e1)
        e1_4us = self.upx4(e1)

        # e2.ssize = 64
        e2 = self.encoders[2](torch.cat((input_4ds, e0_2ds, e1), 1))
        e2_2us = self.upx2(e2)
        e2_4us = self.upx4(e2)
        e2_8us = self.upx8(e2)

        # b.ssize = 64
        b = self.base(torch.cat((input_8ds, e0_4ds, e1_2ds, e2), 1))
        b_2us = self.upx2(b)
        b_4us = self.upx4(b)
        b_8us = self.upx8(b)

        # d2.ssize = 128
        d2 = self.decoders[0](torch.cat((input_8ds, e0_4ds, e1_2ds, e2, b), 1))
        d2_2us = self.upx2(d2)
        d2_4us = self.upx4(d2)

        # d1.ssize = 256
        d1 = self.decoders[1](torch.cat((input_4ds, e0_2ds, e1, e2_2us, b_2us, d2), 1))
        d1_2us = self.upx2(d1)

        # d0.ssize = 512
        d0 = self.decoders[2](torch.cat((input_2ds, e0, e1_2us, e2_4us, b_4us, d2_2us, d1), 1))

        # out.ssize = 512
        out = self.output(torch.cat((input_, e0_2us, e1_4us, e2_8us, b_8us, d2_4us, d1_2us, d0), 1))

        if self.final_activation is not None:
            out = self.final_activation(out)

        if not self.return_hypercolumns:
            return out
        else:
            out = torch.cat((input_, e0_2us, e1_4us, e2_8us, b_8us, d2_4us, d1_2us, d0, out), 1)
            return out


class DUNet(DUNetSkeleton):
    def __init__(self, in_channels, out_channels, N=16, return_hypercolumns=False):
        # Build encoders
        encoders = [
            Encoder([in_channels], N, 3),
            Encoder([in_channels, N], 2 * N, 3),
            Encoder([in_channels, N, 2 * N], 4 * N, 3),
        ]
        # Build base
        base = Base([in_channels, N, 2 * N, 4 * N], 4 * N, 3)
        # Build decoders
        decoders = [
            Decoder([in_channels, N, 2 * N, 4 * N, 4 * N], 2 * N, 3),
            Decoder([in_channels, N, 2 * N, 4 * N, 4 * N, 2 * N], N, 3),
            Decoder([in_channels, N, 2 * N, 4 * N, 4 * N, 2 * N, N], N, 3),
        ]
        # Build output
        output = Output([in_channels, N, 2 * N, 4 * N, 4 * N, 2 * N, N, N], out_channels, 3)
        # Parse final activation
        final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax2d()
        # dundundun
        super(DUNet, self).__init__(
            encoders=encoders,
            decoders=decoders,
            base=base,
            output=output,
            final_activation=final_activation,
            return_hypercolumns=return_hypercolumns,
        )

    def forward(self, input_):
        # CREMI loaders are usually 3D, so we reshape if necessary
        if input_.dim() == 5:
            reshape_to_3d = True
            b, c, _0, _1, _2 = list(input_.size())
            assert _0 == 1
            input_ = input_.view(b, c * _0, _1, _2)
        else:
            reshape_to_3d = False
        output = super(DUNet, self).forward(input_)
        if reshape_to_3d:
            b, c, _0, _1 = list(output.size())
            output = output.view(b, c, 1, _0, _1)
        return output
