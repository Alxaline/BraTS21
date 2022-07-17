# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
import torch
from monai.networks.layers.factories import Act
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class WSConv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True).mean(dim=4,
                                                                                                         keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvAWS3d(nn.Conv3d):
    """AWS (Adaptive Weight Standardization)
    This is a variant of Weight Standardization
    (https://arxiv.org/pdf/1903.10520.pdf)
    It is used in DetectoRS to avoid NaN
    (https://arxiv.org/pdf/2006.02334.pdf)
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the conv kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If set True, adds a learnable bias to the
            output. Default: True
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.register_buffer('weight_gamma',
                             torch.ones(self.out_channels, 1, 1, 1, 1))
        self.register_buffer('weight_beta',
                             torch.zeros(self.out_channels, 1, 1, 1, 1))

    def _get_weight(self, weight):
        weight_flat = weight.view(weight.size(0), -1)
        mean = weight_flat.mean(dim=1).view(-1, 1, 1, 1, 1)
        std = torch.sqrt(weight_flat.var(dim=1) + 1e-5).view(-1, 1, 1, 1, 1)
        weight = (weight - mean) / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x):
        weight = self._get_weight(self.weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override default load function.
        AWS overrides the function _load_from_state_dict to recover
        weight_gamma and weight_beta if they are missing. If weight_gamma and
        weight_beta are found in the checkpoint, this function will return
        after super()._load_from_state_dict. Otherwise, it will compute the
        mean and std of the pretrained weights and store them in weight_beta
        and weight_gamma.
        """

        self.weight_gamma.data.fill_(-1)
        local_missing_keys = []
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, local_missing_keys,
                                      unexpected_keys, error_msgs)
        if self.weight_gamma.data.mean() > 0:
            for k in local_missing_keys:
                missing_keys.append(k)
            return
        weight = self.weight.data
        weight_flat = weight.view(weight.size(0), -1)
        mean = weight_flat.mean(dim=1).view(-1, 1, 1, 1, 1)
        std = torch.sqrt(weight_flat.var(dim=1) + 1e-5).view(-1, 1, 1, 1, 1)
        self.weight_beta.data.copy_(mean)
        self.weight_gamma.data.copy_(std)
        missing_gamma_beta = [
            k for k in local_missing_keys
            if k.endswith('weight_gamma') or k.endswith('weight_beta')
        ]
        for k in missing_gamma_beta:
            local_missing_keys.remove(k)
        for k in local_missing_keys:
            missing_keys.append(k)


class BCNorm(nn.Module):

    def __init__(self, num_channels, num_groups, estimate=False, eps=1e-5):
        super(BCNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = Parameter(torch.ones(1, num_groups, 1))
        self.bias = Parameter(torch.zeros(1, num_groups, 1))
        if estimate:
            self.bn = EstBN(num_channels)
        else:
            self.bn = nn.BatchNorm3d(num_channels)

    def forward(self, inp):
        out = self.bn(inp)
        out = out.view(1, inp.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, self.eps, True)
        out = out.view(inp.size(0), self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.view_as(inp)
        return out


class EstBN(nn.Module):

    def __init__(self, num_features):
        super(EstBN, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('estbn_moving_speed', torch.zeros(1))

    def forward(self, inp):
        ms = self.estbn_moving_speed.item()
        if self.training:
            with torch.no_grad():
                inp_t = inp.transpose(0, 1).contiguous().view(self.num_features, -1)
                running_mean = inp_t.mean(dim=1)
                inp_t = inp_t - self.running_mean.view(-1, 1)
                running_var = torch.mean(inp_t * inp_t, dim=1)
                self.running_mean.data.mul_(1 - ms).add_(ms * running_mean.data)
                self.running_var.data.mul_(1 - ms).add_(ms * running_var.data)
        out = inp - self.running_mean.view(1, -1, 1, 1, 1)
        out = out / torch.sqrt(self.running_var + 1e-5).view(1, -1, 1, 1, 1)
        weight = self.weight.view(1, -1, 1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1, 1)
        out = weight * out + bias
        return out


def get_norm_layer(norm_type="group", dim="3d"):
    print(norm_type)
    if norm_type == "group":
        return lambda x: nn.GroupNorm(8, x, affine=True)
    elif norm_type == "none":
        return None
    elif norm_type == "batch":
        return lambda x: nn.BatchNorm3d(x, affine=True) if dim == "3d" else nn.BatchNorm2d(x, affine=True)
    elif norm_type == "instance":
        return lambda x: nn.InstanceNorm3d(x, affine=True) if dim == "3d" else nn.InstanceNorm2d(x, affine=True)
    elif norm_type == "bcn":
        return lambda x: BCNorm(x, 8, estimate=True)
    else:
        raise ValueError('Norm type is not correct')


def get_act(act_type="relu"):
    if act_type in ["elu", "relu", "leakyrelu"]:
        kwargs = {"inplace": True}
        return Act[act_type](**kwargs)
    else:
        return Act[act_type]()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0.0, mode='fan_out')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # if hasattr(m, 'bias') and m.bias is not None:
            #     nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1 or classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
