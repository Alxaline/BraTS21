# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
| Model used in `<https://arxiv.org/abs/2011.01045>`_

.. seealso::
    Evo implementation in 2D
    `<https://github.com/digantamisra98/EvoNorm/blob/master/models/evonorm2d.py`_
"""

import warnings
from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.blocks import MaxAvgPool, ResidualSELayer
from monai.networks.layers import same_padding
from monai.networks.layers.factories import Conv

from networks.equiunet2020 import conv1x1, RefUnet


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def instance_std(x, eps=1e-5):
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)


def group_std(x, groups=32, eps=1e-5):
    N, C, H, W, D = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W, D))
    var = torch.var(x, dim=(2, 3, 4, 5), keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W, D))


class EvoNorm3D(nn.Module):

    def __init__(self, input, non_linear=True, version='S0', efficient=True, affine=True, momentum=0.9, eps=1e-5,
                 groups=8, training=True):
        super(EvoNorm3D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.efficient = efficient
        if self.version == 'S0':
            self.swish = MemoryEfficientSwish()
        self.groups = groups
        self.eps = eps
        if self.version not in ['B0', 'S0']:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1, 1))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(1, self.insize, 1, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            self.register_buffer('v', None)
        self.register_buffer('running_var', torch.ones(1, self.insize, 1, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == 'S0':
            if self.non_linear:
                if not self.efficient:
                    num = x * torch.sigmoid(self.v * x)  # Original Swish Implementation, however memory intensive.
                else:
                    num = self.swish(x)  # Experimental Memory Efficient Variant of Swish
                return num / group_std(x, groups=self.groups, eps=self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                var = torch.var(x, dim=(0, 2, 3, 4), unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta


class SimpleASPPEVO(nn.Module):
    """
    A simplified version of the atrous spatial pyramid pooling (ASPP) module.

    Chen et al., Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.
    https://arxiv.org/abs/1802.02611

    Wang et al., A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions
    from CT Images. https://ieeexplore.ieee.org/document/9109297
    """

    def __init__(
            self,
            in_channels: int,
            conv_out_channels: int,
            kernel_sizes: Sequence[int] = (1, 3, 3, 3),
            dilations: Sequence[int] = (1, 2, 4, 6),
    ) -> None:
        """
        Args:
            in_channels: number of input channels.
            conv_out_channels: number of output channels of each atrous conv.
                The final number of output channels is conv_out_channels * len(kernel_sizes).
            kernel_sizes: a sequence of four convolutional kernel sizes.
                Defaults to (1, 3, 3, 3) for four (dilated) convolutions.
            dilations: a sequence of four convolutional dilation parameters.
                Defaults to (1, 2, 4, 6) for four (dilated) convolutions.

        Raises:
            ValueError: When ``kernel_sizes`` length differs from ``dilations``.

        See also:

            :py:class:`monai.networks.layers.Act`
            :py:class:`monai.networks.layers.Conv`
            :py:class:`monai.networks.layers.Norm`

        """
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError(
                "kernel_sizes and dilations length must match, "
                f"got kernel_sizes={len(kernel_sizes)} dilations={len(dilations)}."
            )
        pads = tuple(same_padding(k, d) for k, d in zip(kernel_sizes, dilations))

        self.convs = nn.ModuleList()
        for k, d, p in zip(kernel_sizes, dilations, pads):
            _conv = Conv[Conv.CONV, 3](
                in_channels=in_channels, out_channels=conv_out_channels, kernel_size=k, dilation=d, padding=p
            )
            self.convs.append(_conv)

        out_channels = conv_out_channels * len(pads)  # final conv. output channels
        self.conv_k1 = ConvEvo(
            in_channels=out_channels,
            out_channels=out_channels,
            dropout_p=0,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, channel, spatial_1[, spatial_2, ...]).
        """
        x_out = torch.cat([conv(x) for conv in self.convs], dim=1)
        x_out = self.conv_k1(x_out)
        return x_out


class ConvEvoBlockCorrected(nn.Module):
    """Two convolution layers with Evo norm, dropout and SE block"""

    def __init__(self, in_channels, out_channels, dropout_p, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.conv_conv_se = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            EvoNorm3D(out_channels),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            EvoNorm3D(out_channels),
            nn.Dropout(dropout_p),
            ResidualSELayer(spatial_dims=3, in_channels=out_channels, r=2, acti_type_1=("relu", {"inplace": True}),
                            acti_type_2="sigmoid"),
        )

    def forward(self, x):
        return self.conv_conv_se(x)


class ConvEvo(nn.Module):
    """Conv Evo"""

    def __init__(self, in_channels, out_channels, dropout_p, kernel_size=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.evo = EvoNorm3D(out_channels)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.drop(self.evo(self.conv(x)))


class EquiUnetASSPEvo(nn.Module):
    """
    """
    name = "EquiUnetASSPEvo"

    def __init__(self, inplanes, num_classes, features, norm_layer=None, act="relu", deep_supervision=False, dropout=0,
                 refinement=False):
        super(EquiUnetASSPEvo, self).__init__()
        warnings.warn("norm layer and activation specified will not be used ! only EVO !!")

        print(f"EquiUnetASSP features: {features}")
        self.deep_supervision = deep_supervision
        # self.norm_layer = get_norm_layer(norm_layer)

        self.act = act.upper()
        self.refinement = refinement

        f0_half = int(features[0] / 2)
        f1_half = int(features[1] / 2)
        f2_half = int(features[2] / 2)

        self.encoder1 = ConvEvoBlockCorrected(inplanes, features[0], dropout)
        self.encoder2 = ConvEvoBlockCorrected(2 * features[0], features[1], dropout)
        self.encoder3 = ConvEvoBlockCorrected(2 * features[1], features[2], dropout)
        self.encoder4 = ConvEvoBlockCorrected(2 * features[2], features[3], dropout)

        # bridge
        self.bridge1 = ConvEvo(features[0], f0_half, dropout_p=dropout)

        self.bridge2 = ConvEvo(features[1], f1_half, dropout_p=dropout)
        self.bridge3 = ConvEvo(features[2], f2_half, dropout_p=dropout)

        self.aspp = SimpleASPPEVO(
            features[3], int(features[3] / 4), kernel_sizes=[1, 3, 3, 3], dilations=[1, 2, 4, 6]
        )

        self.downsample = MaxAvgPool(spatial_dims=3, kernel_size=2)

        self.upconv3 = ConvEvo(features[3], features[3] // 4, dropout_p=dropout)
        self.decoder3 = ConvEvoBlockCorrected(features[2], features[2], dropout)
        self.upconv2 = ConvEvo(features[2], features[2] // 4, dropout_p=dropout)
        self.decoder2 = ConvEvoBlockCorrected(features[1], features[1], dropout)
        self.upconv1 = ConvEvo(features[1], features[1] // 4, dropout_p=dropout)
        self.decoder1 = ConvEvoBlockCorrected(features[0], features[0], dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.out_conv = conv1x1(features[0], num_classes)
        #
        if self.deep_supervision:
            self.deep3 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        if self.refinement:
            ## -------------Refine Module-------------
            self.refunet = RefUnet(num_classes, features[0], act=self.act, norm_layer=self.norm_layer, dilation=1,
                                   dropout=dropout)

        # init_weights(self, init_type="kaiming")

    def forward(self, x):

        down1 = self.encoder1(x)

        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)
        assp = self.aspp(down4)

        # bridge
        down1b = self.bridge1(down1)
        down2b = self.bridge2(down2)
        down3b = self.bridge3(down3)

        upconv3 = self.upconv3(assp)
        # # Decoder
        up3 = self.upsample(upconv3)

        up3 = self.decoder3(torch.cat([down3b, up3], dim=1))

        upconv2 = self.upconv2(up3)
        up2 = self.upsample(upconv2)

        up2 = self.decoder2(torch.cat([down2b, up2], dim=1))
        upconv1 = self.upconv1(up2)

        up1 = self.upsample(upconv1)
        up1 = self.decoder1(torch.cat([down1b, up1], dim=1))
        #
        out = self.out_conv(up1)

        if self.refinement:
            out = [self.refunet(out), out]

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [up3, up2],
                    [self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps
        return out


if __name__ == '__main__':
    from torch.cuda.amp import autocast

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = EquiUnetASSPEvo(inplanes=4, num_classes=3, features=[48 * 2 ** i for i in range(4)], norm_layer="group",
                          deep_supervision=True, act="leakyrelu").to(device)
    in_ = torch.ones((1, 4, 64, 64, 64)).to(device)
    with autocast(enabled=True):
        out_ = net(in_)
    print([output_.shape for output_ in out_])
