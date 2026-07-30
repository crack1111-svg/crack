'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch.nn as nn


class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


def _safe_group_count(channels, requested_groups):
    requested_groups = max(1, min(int(requested_groups), int(channels)))
    for groups in range(requested_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def get_norm_layer(norm_type, channels, num_groups):
    norm_type = norm_type.upper()
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=_safe_group_count(channels, num_groups), num_channels=channels)
    if norm_type == 'BN':
        return nn.BatchNorm2d(channels)
    if norm_type == 'IN':
        return nn.InstanceNorm2d(channels)
    raise ValueError(f'Unsupported norm_type: {norm_type}')


class GBC(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()
        mid_channels = max(1, in_channels // 8)
        groups_main = max(1, in_channels // 16)

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, groups_main),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, groups_main),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, groups_main),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, mid_channels, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, groups_main),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)

        return x + residual
