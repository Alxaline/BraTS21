# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
| adapted from: `<https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py>`_
"""
import torch
import torch.nn as nn

from networks.factory import get_norm_layer, get_act, init_weights


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, act):
        super(ConvBlock, self).__init__()
        if norm_layer is not None:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                get_act(act),
                nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                get_act(act),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                get_act(act),
                nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                get_act(act),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, act):
        super(UpConv, self).__init__()
        if norm_layer is not None:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                get_act(act),
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                get_act(act),
            )

    def forward(self, x):
        x = self.up(x)
        return x


class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, norm_layer, act, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out

        if norm_layer is not None:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                get_act(act),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                get_act(act),
            )

    def forward(self, x):
        x1 = 0
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNNblock(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, act, t=2):
        super(RRCNNblock, self).__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, norm_layer, act, t=t),
            RecurrentBlock(ch_out, norm_layer, act, t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int, act):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = get_act(act)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class Unet(nn.Module):
    def __init__(self, img_ch, output_ch, features, norm_layer='group', act="relu", deep_supervision=True):
        super(Unet, self).__init__()

        print(f"Unet features: {features}")
        self.features = features
        self.deep_supervision = deep_supervision
        self.norm_layer = get_norm_layer(norm_layer)
        self.act = act

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=self.features[0], norm_layer=self.norm_layer, act=self.act)
        self.Conv2 = ConvBlock(ch_in=self.features[0], ch_out=self.features[1], norm_layer=self.norm_layer,
                               act=self.act)
        self.Conv3 = ConvBlock(ch_in=self.features[1], ch_out=self.features[2], norm_layer=self.norm_layer,
                               act=self.act)
        self.Conv4 = ConvBlock(ch_in=self.features[2], ch_out=self.features[3], norm_layer=self.norm_layer,
                               act=self.act)

        self.Up4 = UpConv(ch_in=self.features[3], ch_out=self.features[2], norm_layer=self.norm_layer, act=self.act)
        self.Up_conv4 = ConvBlock(ch_in=self.features[3], ch_out=self.features[2], norm_layer=self.norm_layer,
                                  act=self.act)

        self.Up3 = UpConv(ch_in=self.features[2], ch_out=self.features[1], norm_layer=self.norm_layer, act=self.act)
        self.Up_conv3 = ConvBlock(ch_in=self.features[2], ch_out=self.features[1], norm_layer=self.norm_layer,
                                  act=self.act)

        self.Up2 = UpConv(ch_in=self.features[1], ch_out=self.features[0], norm_layer=self.norm_layer, act=self.act)
        self.Up_conv2 = ConvBlock(ch_in=self.features[1], ch_out=self.features[0], norm_layer=self.norm_layer,
                                  act=self.act)

        self.Conv_1x1 = nn.Conv3d(self.features[0], output_ch, kernel_size=1, stride=1, padding=0)

        # ------------- DeepSup --------------
        if self.deep_supervision:
            self.upscore4 = nn.Upsample(scale_factor=8)
            self.upscore3 = nn.Upsample(scale_factor=4)
            self.upscore2 = nn.Upsample(scale_factor=2)

            self.outconv4 = nn.Conv3d(self.features[3], output_ch, kernel_size=1, stride=1, padding=0)
            self.outconv3 = nn.Conv3d(self.features[2], output_ch, kernel_size=1, stride=1, padding=0)
            self.outconv2 = nn.Conv3d(self.features[1], output_ch, kernel_size=1, stride=1, padding=0)

        init_weights(self, init_type="kaiming")

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # decoding + concat path
        d4 = self.Up4(x4)

        d4 = torch.cat((x3, d4), dim=1)
        d4_up = self.Up_conv4(d4)
        d3 = self.Up3(d4_up)
        d3 = torch.cat((x2, d3), dim=1)
        d3_up = self.Up_conv3(d3)

        d2 = self.Up2(d3_up)
        d2 = torch.cat((x1, d2), dim=1)
        d2_up = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2_up)

        if self.deep_supervision:
            d4 = self.outconv4(x4)
            d4 = self.upscore4(d4)
            d3 = self.outconv3(d4_up)
            d3 = self.upscore3(d3)
            d2 = self.outconv2(d3_up)
            d2 = self.upscore2(d2)

            return d1, d2, d3, d4

        return d1


class R2Unet(nn.Module):
    def __init__(self, img_ch, output_ch, features, t=2, norm_layer='group', act="relu", deep_supervision=True):
        super(R2Unet, self).__init__()

        print(f"R2Unet features: {features}")
        self.features = features
        self.deep_supervision = deep_supervision
        self.norm_layer = get_norm_layer(norm_layer)
        self.act = act

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNNblock(ch_in=img_ch, ch_out=self.features[0], t=t, norm_layer=self.norm_layer, act=self.act)

        self.RRCNN2 = RRCNNblock(ch_in=self.features[0], ch_out=self.features[1], t=t, norm_layer=self.norm_layer,
                                 act=self.act)

        self.RRCNN3 = RRCNNblock(ch_in=self.features[1], ch_out=self.features[2], t=t, norm_layer=self.norm_layer,
                                 act=self.act)

        self.RRCNN4 = RRCNNblock(ch_in=self.features[2], ch_out=self.features[3], t=t, norm_layer=self.norm_layer,
                                 act=self.act)

        self.Up4 = UpConv(ch_in=self.features[3], ch_out=self.features[2], norm_layer=self.norm_layer, act=self.act)
        self.Up_RRCNN4 = RRCNNblock(ch_in=self.features[3], ch_out=self.features[2], t=t, norm_layer=self.norm_layer,
                                    act=self.act)

        self.Up3 = UpConv(ch_in=self.features[2], ch_out=self.features[1], norm_layer=self.norm_layer, act=self.act)
        self.Up_RRCNN3 = RRCNNblock(ch_in=self.features[2], ch_out=self.features[1], t=t, norm_layer=self.norm_layer,
                                    act=self.act)

        self.Up2 = UpConv(ch_in=self.features[1], ch_out=self.features[0], norm_layer=self.norm_layer, act=self.act)
        self.Up_RRCNN2 = RRCNNblock(ch_in=self.features[1], ch_out=self.features[0], t=t, norm_layer=self.norm_layer,
                                    act=self.act)

        self.Conv_1x1 = nn.Conv3d(self.features[0], output_ch, kernel_size=1, stride=1, padding=0)

        # ------------- DeepSup --------------
        if self.deep_supervision:
            self.upscore4 = nn.Upsample(scale_factor=8)
            self.upscore3 = nn.Upsample(scale_factor=4)
            self.upscore2 = nn.Upsample(scale_factor=2)

            self.outconv4 = nn.Conv3d(self.features[3], output_ch, kernel_size=1, stride=1, padding=0)
            self.outconv3 = nn.Conv3d(self.features[2], output_ch, kernel_size=1, stride=1, padding=0)
            self.outconv2 = nn.Conv3d(self.features[1], output_ch, kernel_size=1, stride=1, padding=0)

        init_weights(self, init_type="kaiming")

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        # decoding + concat path
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4_up = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4_up)
        d3 = torch.cat((x2, d3), dim=1)
        d3_up = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3_up)
        d2 = torch.cat((x1, d2), dim=1)
        d2_up = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2_up)

        if self.deep_supervision:
            d4 = self.outconv4(x4)
            d4 = self.upscore4(d4)
            d3 = self.outconv3(d4_up)
            d3 = self.upscore3(d3)
            d2 = self.outconv2(d3_up)
            d2 = self.upscore2(d2)

            return d1, d2, d3, d4

        return d1


class AttUnet(nn.Module):
    def __init__(self, img_ch, output_ch, features, norm_layer='group', act="relu", deep_supervision=True):
        super(AttUnet, self).__init__()

        print(f"AttUnet features: {features}")
        self.features = features
        self.deep_supervision = deep_supervision
        self.norm_layer = get_norm_layer(norm_layer)
        self.act = act

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=self.features[0], norm_layer=self.norm_layer, act=self.act)
        self.Conv2 = ConvBlock(ch_in=self.features[0], ch_out=self.features[1], norm_layer=self.norm_layer,
                               act=self.act)
        self.Conv3 = ConvBlock(ch_in=self.features[1], ch_out=self.features[2], norm_layer=self.norm_layer,
                               act=self.act)
        self.Conv4 = ConvBlock(ch_in=self.features[2], ch_out=self.features[3], norm_layer=self.norm_layer,
                               act=self.act)

        self.Up4 = UpConv(ch_in=self.features[3], ch_out=self.features[2], norm_layer=self.norm_layer, act=self.act)
        self.Att4 = AttentionBlock(f_g=self.features[2], f_l=self.features[2], f_int=self.features[1], act=self.act)
        self.Up_conv4 = ConvBlock(ch_in=self.features[3], ch_out=self.features[2], norm_layer=self.norm_layer,
                                  act=self.act)

        self.Up3 = UpConv(ch_in=self.features[2], ch_out=self.features[1], norm_layer=self.norm_layer, act=self.act)
        self.Att3 = AttentionBlock(f_g=self.features[1], f_l=self.features[1], f_int=self.features[0], act=self.act)
        self.Up_conv3 = ConvBlock(ch_in=self.features[2], ch_out=self.features[1], norm_layer=self.norm_layer,
                                  act=self.act)

        self.Up2 = UpConv(ch_in=self.features[1], ch_out=self.features[0], norm_layer=self.norm_layer, act=self.act)
        self.Att2 = AttentionBlock(f_g=self.features[0], f_l=self.features[0], f_int=self.features[0] // 2,
                                   act=self.act)
        self.Up_conv2 = ConvBlock(ch_in=self.features[1], ch_out=self.features[0], norm_layer=self.norm_layer,
                                  act=self.act)

        self.Conv_1x1 = nn.Conv3d(self.features[0], output_ch, kernel_size=1, stride=1, padding=0)

        # ------------- DeepSup --------------
        if self.deep_supervision:
            self.upscore4 = nn.Upsample(scale_factor=8)
            self.upscore3 = nn.Upsample(scale_factor=4)
            self.upscore2 = nn.Upsample(scale_factor=2)

            self.outconv4 = nn.Conv3d(self.features[3], output_ch, kernel_size=1, stride=1, padding=0)
            self.outconv3 = nn.Conv3d(self.features[2], output_ch, kernel_size=1, stride=1, padding=0)
            self.outconv2 = nn.Conv3d(self.features[1], output_ch, kernel_size=1, stride=1, padding=0)

        init_weights(self, init_type="kaiming")

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # decoding + concat path
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4_up = self.Up_conv4(d4)

        d3 = self.Up3(d4_up)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3_up = self.Up_conv3(d3)

        d2 = self.Up2(d3_up)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2_up = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2_up)

        if self.deep_supervision:
            d4 = self.outconv4(x4)
            d4 = self.upscore4(d4)
            d3 = self.outconv3(d4_up)
            d3 = self.upscore3(d3)
            d2 = self.outconv2(d3_up)
            d2 = self.upscore2(d2)

            return d1, d2, d3, d4

        return d1


class R2AttUnet(nn.Module):
    def __init__(self, img_ch, output_ch, features, t=2, norm_layer='group', act="relu", deep_supervision=True):
        super(R2AttUnet, self).__init__()

        print(f"R2AttUnet features: {features}")
        self.features = features
        self.deep_supervision = deep_supervision
        self.norm_layer = get_norm_layer(norm_layer)
        self.act = act

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNNblock(ch_in=img_ch, ch_out=self.features[0], t=t, norm_layer=self.norm_layer, act=self.act)

        self.RRCNN2 = RRCNNblock(ch_in=self.features[0], ch_out=self.features[1], t=t, norm_layer=self.norm_layer,
                                 act=self.act)

        self.RRCNN3 = RRCNNblock(ch_in=self.features[1], ch_out=self.features[2], t=t, norm_layer=self.norm_layer,
                                 act=self.act)

        self.RRCNN4 = RRCNNblock(ch_in=self.features[2], ch_out=self.features[3], t=t, norm_layer=self.norm_layer,
                                 act=self.act)

        self.Up4 = UpConv(ch_in=self.features[3], ch_out=self.features[2], norm_layer=self.norm_layer, act=self.act)
        self.Att4 = AttentionBlock(f_g=self.features[2], f_l=self.features[2], f_int=self.features[1], act=self.act)
        self.Up_RRCNN4 = RRCNNblock(ch_in=self.features[3], ch_out=self.features[2], t=t, norm_layer=self.norm_layer,
                                    act=self.act)

        self.Up3 = UpConv(ch_in=self.features[2], ch_out=self.features[1], norm_layer=self.norm_layer, act=self.act)
        self.Att3 = AttentionBlock(f_g=self.features[1], f_l=self.features[1], f_int=self.features[0], act=self.act)
        self.Up_RRCNN3 = RRCNNblock(ch_in=self.features[2], ch_out=self.features[1], t=t, norm_layer=self.norm_layer,
                                    act=self.act)

        self.Up2 = UpConv(ch_in=self.features[1], ch_out=self.features[0], norm_layer=self.norm_layer, act=self.act)
        self.Att2 = AttentionBlock(f_g=self.features[0], f_l=self.features[0], f_int=self.features[0] // 2,
                                   act=self.act)
        self.Up_RRCNN2 = RRCNNblock(ch_in=self.features[1], ch_out=self.features[0], t=t, norm_layer=self.norm_layer,
                                    act=self.act)

        self.Conv_1x1 = nn.Conv3d(self.features[0], output_ch, kernel_size=1, stride=1, padding=0)

        # ------------- DeepSup --------------
        if self.deep_supervision:
            self.upscore4 = nn.Upsample(scale_factor=8)
            self.upscore3 = nn.Upsample(scale_factor=4)
            self.upscore2 = nn.Upsample(scale_factor=2)

            self.outconv4 = nn.Conv3d(self.features[3], output_ch, kernel_size=1, stride=1, padding=0)
            self.outconv3 = nn.Conv3d(self.features[2], output_ch, kernel_size=1, stride=1, padding=0)
            self.outconv2 = nn.Conv3d(self.features[1], output_ch, kernel_size=1, stride=1, padding=0)

        init_weights(self, init_type="kaiming")

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        # decoding + concat path
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4_up = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4_up)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3_up = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3_up)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2_up = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2_up)

        if self.deep_supervision:
            d4 = self.outconv4(x4)
            d4 = self.upscore4(d4)
            d3 = self.outconv3(d4_up)
            d3 = self.upscore3(d3)
            d2 = self.outconv2(d3_up)
            d2 = self.upscore2(d2)

            return d1, d2, d3, d4

        return d1


from monai.networks.nets import DynUNet


class WrapperDynUNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels,
                 kernel_size=([3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]),
                 strides=([1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]),
                 upsample_kernel_size=([2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]),
                 norm_name="instance", deep_supervision=True, deep_supr_num=3, res_block=False):
        super(WrapperDynUNet, self).__init__()
        self.deep_supervision = deep_supervision
        self.net = DynUNet(spatial_dims, in_channels, out_channels, kernel_size, strides,
                           upsample_kernel_size,
                           norm_name, deep_supervision, deep_supr_num, res_block)

    def forward(self, x):
        output = self.net(x)
        if self.net.training and self.deep_supervision:
            return list(map(lambda y: torch.squeeze(y, dim=1), torch.split(output, split_size_or_sections=1, dim=1)))
        else:
            return output
