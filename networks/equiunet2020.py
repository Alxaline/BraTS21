# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
| Model used in `<https://arxiv.org/abs/2011.01045>`_

.. seealso::
    `<https://github.com/lescientifik/open_brats2020>`_
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from networks.factory import get_norm_layer, get_act, init_weights


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """
    3x3 convolution with padding
    """

    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv3x3_2d(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """
    3x3 convolution with padding
    """

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """
    1x1 convolution
    """
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv1x1_2d(in_planes, out_planes, stride=1, bias=True):
    """
    1x1 convolution
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ConvBnRelu(nn.Sequential):

    def __init__(self, inplanes, planes, act, norm_layer=None, dilation=1, dropout=0):
        if norm_layer is not None:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv',
                         conv3x3(inplanes, planes, dilation=dilation)),
                        ('bn', norm_layer(planes)),
                        (act, get_act(act)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )
        else:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation, bias=True)),
                        (act, get_act(act)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )


class ConvBnRelu2D(nn.Sequential):

    def __init__(self, inplanes, planes, act, norm_layer=None, dilation=1, dropout=0):
        if norm_layer is not None:
            super(ConvBnRelu2D, self).__init__(
                OrderedDict(
                    [
                        ('conv',
                         conv3x3_2d(inplanes, planes, dilation=dilation)),
                        ('bn', norm_layer(planes)),
                        (act, get_act(act)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )
        else:
            super(ConvBnRelu2D, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3_2d(inplanes, planes, dilation=dilation, bias=True)),
                        (act, get_act(act)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )


class UBlock(nn.Sequential):
    """
    Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, act, norm_layer, dilation=(1, 1), dropout=0,
                 ):
        super(UBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1',
                     ConvBnRelu(inplanes, midplanes, act, norm_layer, dilation=dilation[0], dropout=dropout,
                                )),
                    (
                        'ConvBnRelu2',
                        ConvBnRelu(midplanes, outplanes, act, norm_layer, dilation=dilation[1], dropout=dropout,
                                   )),
                ])
        )


class UBlock2D(nn.Sequential):
    """
    Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, act, norm_layer, dilation=(1, 1), dropout=0,
                 ):
        super(UBlock2D, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1',
                     ConvBnRelu2D(inplanes, midplanes, act, norm_layer, dilation=dilation[0], dropout=dropout,
                                  )),
                    (
                        'ConvBnRelu2',
                        ConvBnRelu2D(midplanes, outplanes, act, norm_layer, dilation=dilation[1], dropout=dropout,
                                     )),
                ])
        )


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=None):
        super(BasicConv, self).__init__()
        bias = False
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None):
        super(ChannelGate, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    @staticmethod
    def forward(x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, norm_layer=None):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, norm_layer=norm_layer)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, norm_layer=None):
        super(CBAM, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(norm_layer)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class UBlockCbam(nn.Sequential):
    def __init__(self, inplanes, midplanes, outplanes, act, norm_layer, dilation=(1, 1), dropout=0,
                 ):
        super(UBlockCbam, self).__init__(
            OrderedDict(
                [
                    ('UBlock',
                     UBlock(inplanes, midplanes, outplanes, act, norm_layer, dilation=dilation, dropout=dropout,
                            )),
                    ('CBAM', CBAM(outplanes, norm_layer=norm_layer)),
                ])
        )


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch, act, norm_layer=None, dilation=1, dropout=0):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv3d(in_ch, inc_ch, 3, padding=1)

        self.hx1 = ConvBnRelu(inc_ch, inc_ch, act, norm_layer=norm_layer, dilation=dilation, dropout=dropout)
        self.hx2 = ConvBnRelu(inc_ch, inc_ch, act, norm_layer=norm_layer, dilation=dilation, dropout=dropout)
        self.hx3 = ConvBnRelu(inc_ch, inc_ch, act, norm_layer=norm_layer, dilation=dilation, dropout=dropout)
        self.hx4 = ConvBnRelu(inc_ch, inc_ch, act, norm_layer=norm_layer, dilation=dilation, dropout=dropout)

        #####
        self.hx5 = ConvBnRelu(inc_ch, inc_ch, act, norm_layer=norm_layer, dilation=dilation, dropout=dropout)
        #####

        self.d4 = ConvBnRelu(inc_ch * 2, inc_ch, act, norm_layer=norm_layer, dilation=dilation, dropout=dropout)
        self.d3 = ConvBnRelu(inc_ch * 2, inc_ch, act, norm_layer=norm_layer, dilation=dilation, dropout=dropout)
        self.d2 = ConvBnRelu(inc_ch * 2, inc_ch, act, norm_layer=norm_layer, dilation=dilation, dropout=dropout)
        self.d1 = ConvBnRelu(inc_ch * 2, inc_ch, act, norm_layer=norm_layer, dilation=dilation, dropout=dropout)

        self.conv_d0 = nn.Conv3d(inc_ch, in_ch, 3, padding=1)

        self.pool = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.hx1(hx)
        hx = self.pool(hx1)

        hx2 = self.hx2(hx)
        hx = self.pool(hx2)

        hx3 = self.hx3(hx)
        hx = self.pool(hx3)

        hx4 = self.hx4(hx)
        hx = self.pool(hx4)

        hx5 = self.hx5(hx)
        hx = self.upscore2(hx5)

        d4 = self.d4(torch.cat((hx, hx4), 1))
        hx = self.upscore2(d4)

        d3 = self.d3(torch.cat((hx, hx3), 1))
        hx = self.upscore2(d3)

        d2 = self.d2(torch.cat((hx, hx2), 1))
        hx = self.upscore2(d2)

        d1 = self.d1(torch.cat((hx, hx1), 1))

        residual = self.conv_d0(d1)

        return x + residual


class Unet(nn.Module):
    """
    Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, features, norm_layer=None, act="relu", deep_supervision=False, dropout=0,
                 ):
        super(Unet, self).__init__()

        print(f"Unet features: {features}")
        self.deep_supervision = deep_supervision
        self.norm_layer = get_norm_layer(norm_layer)
        self.act = act

        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], self.act, self.norm_layer, dropout=dropout,
                               )
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], self.act, self.norm_layer, dropout=dropout,
                               )
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], self.act, self.norm_layer, dropout=dropout,
                               )
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], self.act, self.norm_layer, dropout=dropout,
                               )

        self.bottom = UBlock(features[3], features[3], features[3], self.act, self.norm_layer, (2, 2), dropout=dropout,
                             )

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], self.act, self.norm_layer, dropout=dropout,
                                   )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], self.act, self.norm_layer, dropout=dropout,
                               )
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], self.act, self.norm_layer, dropout=dropout,
                               )
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, self.act, self.norm_layer,
                               dropout=dropout, )

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        init_weights(self, init_type="kaiming")

    def forward(self, x):

        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder

        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        out = self.outconv(up1)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps
        return out


class EquiUnet(nn.Module):
    """
    Almost the most basic U-net: all Block have the same size if they are at the same level.
    """
    name = "EquiUnet"

    def __init__(self, inplanes, num_classes, features, norm_layer=None, act="relu", deep_supervision=False, dropout=0,
                 refinement=False):
        super(EquiUnet, self).__init__()

        print(f"EquiUnet features: {features}")
        self.deep_supervision = deep_supervision
        self.norm_layer = get_norm_layer(norm_layer)
        self.act = act
        self.refinement = refinement

        self.encoder1 = UBlock(inplanes, features[0], features[0], self.act, self.norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1], features[1], self.act, self.norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2], features[2], self.act, self.norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3], features[3], self.act, self.norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], self.act, self.norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], self.act, self.norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], self.act, self.norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], self.act, self.norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], self.act, self.norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        if self.refinement:
            ## -------------Refine Module-------------
            self.refunet = RefUnet(num_classes, features[0], act=self.act, norm_layer=self.norm_layer, dilation=1,
                                   dropout=dropout)

        init_weights(self, init_type="kaiming")

    def forward(self, x):

        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))
        # Decoder

        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        out = self.outconv(up1)

        if self.refinement:
            out = [self.refunet(out), out]

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps
        return out


class AttEquiUnet(Unet):
    def __init__(self, inplanes, num_classes, features, norm_layer=None, act="relu", deep_supervision=False, dropout=0,
                 ):
        super(Unet, self).__init__()

        print(f"AttEquiUnet features: {features}")
        self.deep_supervision = deep_supervision
        self.norm_layer = get_norm_layer(norm_layer)
        self.act = act

        self.encoder1 = UBlockCbam(inplanes, features[0], features[0], self.act, self.norm_layer, dropout=dropout,
                                   )
        self.encoder2 = UBlockCbam(features[0], features[1], features[1], self.act, self.norm_layer, dropout=dropout,
                                   )
        self.encoder3 = UBlockCbam(features[1], features[2], features[2], self.act, self.norm_layer, dropout=dropout,
                                   )
        self.encoder4 = UBlockCbam(features[2], features[3], features[3], self.act, self.norm_layer, dropout=dropout,
                                   )

        self.bottom = UBlockCbam(features[3], features[3], features[3], self.act, self.norm_layer, (2, 2),
                                 dropout=dropout, )

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], self.act, self.norm_layer, dropout=dropout,
                       ),
            CBAM(features[2], norm_layer=self.norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], self.act, self.norm_layer, dropout=dropout,
                               )
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], self.act, self.norm_layer, dropout=dropout,
                               )
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], self.act, self.norm_layer, dropout=dropout,
                               )

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        init_weights(self, init_type="kaiming")


# if __name__ == '__main__':
#     from torch.cuda.amp import autocast
#
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     net = EquiUnet(inplanes=4, num_classes=3, features=[48 * 2 ** i for i in range(4)], norm_layer="group",
#                    deep_supervision=True).to(device)
#     in_ = torch.rand((1, 4, 64, 64, 64)).to(device)
#     with autocast(enabled=True):
#         out_ = net(in_)
#     print([output_.shape for output_ in out_])
