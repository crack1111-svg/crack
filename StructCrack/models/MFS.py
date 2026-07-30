'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DySample import DySample
from models.GBC import BottConv, GBC, get_norm_layer


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w


class FuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='GN', use_attention=True):
        super().__init__()
        mid = max(4, out_channels)
        self.gbc = GBC(in_channels, norm_type=norm_type)
        self.norm = get_norm_layer(norm_type, in_channels, max(1, in_channels // 16))
        self.proj = BottConv(in_channels, out_channels, mid_channels=mid, kernel_size=1, stride=1, padding=0)
        self.attn = ChannelAttention(out_channels) if use_attention else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.gbc(x)
        x = self.norm(x)
        x = self.proj(x)
        x = self.attn(x)
        x = self.act(x)
        return x


class MFS(nn.Module):
    def __init__(self,
                 input_dims,
                 embedding_dim,
                 norm_type='GN',
                 fusion_mode='hierarchical',
                 use_feature_attention=True,
                 detail_channels=0,
                 upsample_mode='dysample'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fusion_mode = fusion_mode
        self.detail_channels = detail_channels
        self.upsample_mode = upsample_mode

        self.projectors = nn.ModuleList([MLP(input_dim=d, embed_dim=embedding_dim) for d in input_dims])
        self.linear_fuse = BottConv(embedding_dim * 4, embedding_dim, max(1, embedding_dim), kernel_size=1, padding=0, stride=1)
        self.final_fuse = FuseBlock(embedding_dim + (detail_channels if detail_channels > 0 else 0),
                                    embedding_dim, norm_type=norm_type,
                                    use_attention=use_feature_attention)

        self.fuse43 = FuseBlock(embedding_dim * 2, embedding_dim, norm_type=norm_type, use_attention=use_feature_attention)
        self.fuse432 = FuseBlock(embedding_dim * 2, embedding_dim, norm_type=norm_type, use_attention=use_feature_attention)
        self.fuse4321 = FuseBlock(embedding_dim * 2, embedding_dim, norm_type=norm_type, use_attention=use_feature_attention)

        self.dropout = nn.Dropout(p=0.1)

        if self.upsample_mode == 'dysample':
            self.DySample_C_2 = DySample(embedding_dim, scale=2)
            self.DySample_C_4 = DySample(embedding_dim, scale=4)
            self.DySample_C_8 = DySample(embedding_dim, scale=8)
        else:
            self.DySample_C_2 = self.DySample_C_4 = self.DySample_C_8 = None

    def _project(self, feat, projector):
        b, c, h, w = feat.shape
        return projector(feat.reshape(b, c, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)

    def _upsample(self, x, scale_factor):
        if scale_factor == 1:
            return x
        if self.upsample_mode == 'dysample':
            if scale_factor == 2:
                return self.DySample_C_2(x)
            if scale_factor == 4:
                return self.DySample_C_4(x)
            if scale_factor == 8:
                return self.DySample_C_8(x)
        return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, inputs, detail_feat=None):
        c4, c3, c2, c1 = inputs

        out_c4 = self._upsample(self._project(c4, self.projectors[0]), 8)
        out_c3 = self._upsample(self._project(c3, self.projectors[1]), 4)
        out_c2 = self._upsample(self._project(c2, self.projectors[2]), 2)
        out_c1 = self._project(c1, self.projectors[3])

        if self.fusion_mode == 'concat':
            fused = self.linear_fuse(torch.cat([out_c4, out_c3, out_c2, out_c1], dim=1))
        else:
            fused = self.fuse43(torch.cat([out_c4, out_c3], dim=1))
            fused = self.fuse432(torch.cat([fused, out_c2], dim=1))
            fused = self.fuse4321(torch.cat([fused, out_c1], dim=1))

        if detail_feat is not None:
            if detail_feat.shape[-2:] != fused.shape[-2:]:
                detail_feat = F.interpolate(detail_feat, size=fused.shape[-2:], mode='bilinear', align_corners=False)
            fused = torch.cat([fused, detail_feat], dim=1)

        fused = self.final_fuse(fused)
        fused = self.dropout(fused)

        return fused, [out_c4, out_c3, out_c2, out_c1]
