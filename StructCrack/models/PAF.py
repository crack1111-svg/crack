'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GBC import BottConv, get_norm_layer


class PAF(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 after_relu: bool = False,
                 norm_type: str = 'GN'):
        super().__init__()
        self.after_relu = after_relu
        mid_channels = max(1, int(mid_channels))

        self.feature_transform = nn.Sequential(
            BottConv(in_channels, mid_channels, mid_channels=max(1, mid_channels // 2), kernel_size=1),
            get_norm_layer(norm_type, mid_channels, max(1, mid_channels // 16))
        )

        self.channel_adapter = nn.Sequential(
            BottConv(mid_channels, in_channels, mid_channels=max(1, mid_channels // 2), kernel_size=1),
            get_norm_layer(norm_type, in_channels, max(1, in_channels // 16))
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, base_feat: torch.Tensor, guidance_feat: torch.Tensor) -> torch.Tensor:
        base_shape = base_feat.size()

        if self.after_relu:
            base_feat = self.relu(base_feat)
            guidance_feat = self.relu(guidance_feat)

        guidance_query = self.feature_transform(guidance_feat)
        base_key = self.feature_transform(base_feat)
        guidance_query = F.interpolate(guidance_query, size=[base_shape[2], base_shape[3]], mode='bilinear', align_corners=False)
        similarity_map = torch.sigmoid(self.channel_adapter(base_key * guidance_query))
        resized_guidance = F.interpolate(guidance_feat, size=[base_shape[2], base_shape[3]], mode='bilinear', align_corners=False)

        fused_feature = (1 - similarity_map) * base_feat + similarity_map * resized_guidance
        return fused_feature
