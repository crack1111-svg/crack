'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from typing import Sequence
import copy

import numpy as np
import torch
import torch.nn as nn

from mmcls.SAVSS_dev.models.modules.patch_embed import ConvPatchEmbed
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES
from models.GBC import BottConv


def _safe_groups(channels: int) -> int:
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def build_norm_layer(norm_cfg, num_features, postfix=''):
    cfg = dict(norm_cfg) if norm_cfg is not None else {'type': 'LN', 'eps': 1e-6}
    layer_type = cfg.pop('type', 'LN').upper()
    eps = cfg.pop('eps', 1e-6)
    if layer_type == 'LN':
        layer = nn.LayerNorm(num_features, eps=eps)
    elif layer_type == 'BN':
        layer = nn.BatchNorm1d(num_features, eps=eps)
    elif layer_type == 'GN':
        groups = _safe_groups(num_features)
        layer = nn.GroupNorm(groups, num_features, eps=eps)
    else:
        raise ValueError(f'Unsupported norm layer type: {layer_type}')
    return f'norm{postfix}', layer


def resize_pos_embed(pos_embed, old_hw, new_hw, mode='bicubic', num_extra_tokens=0):
    if old_hw == new_hw:
        return pos_embed
    B, L, C = pos_embed.shape
    token = pos_embed[:, num_extra_tokens:, :]
    token = token.reshape(B, old_hw[0], old_hw[1], C).permute(0, 3, 1, 2)
    token = nn.functional.interpolate(token, size=new_hw, mode='bilinear', align_corners=False)
    token = token.permute(0, 2, 3, 1).reshape(B, new_hw[0] * new_hw[1], C)
    if num_extra_tokens > 0:
        extra = pos_embed[:, :num_extra_tokens, :]
        token = torch.cat([extra, token], dim=1)
    return token


@BACKBONES.register_module()
class SAVSS(BaseBackbone):
    arch_zoo = {
        'savss_tiny': {
            'patch_size': 8,
            'embed_dims': 128,
            'num_layers': 4,
            'num_convs_patch_embed': 1,
            'layers_with_dwconv': [],
            'with_pos_embed': False,
            'stage_out_channels': [96, 48, 24, 16],
            'layer_cfgs': {
                'use_rms_norm': False,
                'gbc_repeats': 1,
                'use_paf': True,
                'paf_reduction': 2,
                'mamba_cfg': {
                    'd_state': 8,
                    'expand': 1.5,
                    'conv_size': 5,
                    'dt_init': 'random',
                    'conv_bias': True,
                    'bias': True,
                }
            }
        },
        'savss_light': {
            'patch_size': 8,
            'embed_dims': 160,
            'num_layers': 4,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'with_pos_embed': False,
            'stage_out_channels': [128, 64, 32, 16],
            'layer_cfgs': {
                'use_rms_norm': False,
                'gbc_repeats': 1,
                'use_paf': True,
                'paf_reduction': 2,
                'mamba_cfg': {
                    'd_state': 8,
                    'expand': 1.5,
                    'conv_size': 5,
                    'dt_init': 'random',
                    'conv_bias': True,
                    'bias': True,
                }
            }
        },
        'savss_base': {
            'patch_size': 8,
            'embed_dims': 256,
            'num_layers': 4,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'with_pos_embed': True,
            'stage_out_channels': [128, 64, 32, 16],
            'layer_cfgs': {
                'use_rms_norm': False,
                'gbc_repeats': 2,
                'use_paf': True,
                'paf_reduction': 2,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2.0,
                    'conv_size': 7,
                    'dt_init': 'random',
                    'conv_bias': True,
                    'bias': True,
                }
            }
        },
        'savss_highacc': {
            'patch_size': 4,
            'embed_dims': 224,
            'num_layers': 4,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'with_pos_embed': False,
            'stage_out_channels': [128, 64, 32, 16],
            'layer_cfgs': {
                'use_rms_norm': False,
                'gbc_repeats': 1,
                'use_paf': True,
                'paf_reduction': 2,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2.0,
                    'conv_size': 7,
                    'dt_init': 'random',
                    'conv_bias': True,
                    'bias': True,
                }
            }
        },
        'Crack': {
            'patch_size': 8,
            'embed_dims': 256,
            'num_layers': 4,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'with_pos_embed': True,
            'stage_out_channels': [128, 64, 32, 16],
            'layer_cfgs': {
                'use_rms_norm': False,
                'gbc_repeats': 2,
                'use_paf': True,
                'paf_reduction': 2,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2.0,
                    'conv_size': 7,
                    'dt_init': 'random',
                    'conv_bias': True,
                    'bias': True,
                }
            }
        },
    }

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 arch='savss_base',
                 patch_size=16,
                 embed_dims=192,
                 num_layers=20,
                 num_convs_patch_embed=1,
                 with_pos_embed=True,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 interpolate_mode='bicubic',
                 layer_cfgs=dict(),
                 layers_with_dwconv=None,
                 init_cfg=None,
                 test_cfg=dict(),
                 convert_syncbn=False,
                 freeze_patch_embed=False,
                 norm_type='GN',
                 stage_out_channels=None,
                 gbc_repeats=None,
                 use_paf=None,
                 paf_reduction=None,
                 d_state=None,
                 expand=None,
                 conv_size=None,
                 **kwargs):
        super().__init__(init_cfg)

        self.test_cfg = test_cfg
        self.img_size = to_2tuple(img_size)
        self.convert_syncbn = convert_syncbn
        self.arch = arch
        self.interpolate_mode = interpolate_mode
        self.freeze_patch_embed = freeze_patch_embed
        layers_with_dwconv = layers_with_dwconv or []

        if arch is not None:
            assert arch in self.arch_zoo, f'Unsupported arch: {arch}'
            cfg = copy.deepcopy(self.arch_zoo[arch])
            self.embed_dims = int(embed_dims) if embed_dims != 192 else cfg['embed_dims']
            self.num_layers = int(num_layers) if num_layers != 20 else cfg['num_layers']
            self.patch_size = int(patch_size) if patch_size != 16 else cfg['patch_size']
            self.num_convs_patch_embed = int(num_convs_patch_embed) if num_convs_patch_embed != 1 else cfg['num_convs_patch_embed']
            self.layers_with_dwconv = cfg['layers_with_dwconv'] if not layers_with_dwconv else layers_with_dwconv
            self.with_pos_embed = bool(with_pos_embed) if with_pos_embed != True else cfg.get('with_pos_embed', True)
            self.stage_out_channels = list(stage_out_channels or cfg.get('stage_out_channels', [128, 64, 32, 16]))
            _layer_cfgs = copy.deepcopy(cfg['layer_cfgs'])
        else:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
            self.with_pos_embed = with_pos_embed
            self.stage_out_channels = list(stage_out_channels or [128, 64, 32, 16])
            _layer_cfgs = copy.deepcopy(layer_cfgs)

        _layer_cfgs.setdefault('use_rms_norm', False)
        _layer_cfgs.setdefault('gbc_repeats', 2)
        _layer_cfgs.setdefault('use_paf', True)
        _layer_cfgs.setdefault('paf_reduction', 2)
        _layer_cfgs.setdefault('mamba_cfg', {})
        _layer_cfgs['norm_type'] = norm_type

        if gbc_repeats is not None:
            _layer_cfgs['gbc_repeats'] = gbc_repeats
        if use_paf is not None:
            _layer_cfgs['use_paf'] = use_paf
        if paf_reduction is not None:
            _layer_cfgs['paf_reduction'] = paf_reduction
        if d_state is not None:
            _layer_cfgs['mamba_cfg']['d_state'] = d_state
        if expand is not None:
            _layer_cfgs['mamba_cfg']['expand'] = expand
        if conv_size is not None:
            _layer_cfgs['mamba_cfg']['conv_size'] = conv_size

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size
        )
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        if self.with_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), f'"out_indices" must be sequence or int, got {type(out_indices)}'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.drop_path_rate = drop_path_rate

        self.layers = nn.ModuleList()
        layer_cfgs_for_all = [copy.deepcopy(_layer_cfgs) for _ in range(self.num_layers)]
        for i in range(self.num_layers):
            layer_cfg_i = layer_cfgs_for_all[i]
            layer_cfg_i.update({
                'embed_dims': self.embed_dims,
                'drop_path_rate': float(dpr[i]),
                'with_dwconv': i in self.layers_with_dwconv,
            })
            self.layers.append(SAVSS_Layer(**layer_cfg_i))

        self.final_norm = final_norm
        if final_norm:
            self.final_norm_layer = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)[1]
        else:
            self.final_norm_layer = nn.Identity()

        for i in out_indices:
            if i != self.num_layers - 1:
                norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1] if norm_cfg is not None else nn.Identity()
                self.add_module(f'norm_layer{i}', norm_layer)

        assert len(self.stage_out_channels) >= len(self.out_indices), 'stage_out_channels 的长度必须不少于输出尺度数'
        self.stage_projs = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        for out_ch in self.stage_out_channels[:len(self.out_indices)]:
            mid_ch = max(4, out_ch // 4)
            self.stage_projs.append(
                BottConv(in_channels=self.embed_dims, out_channels=out_ch, mid_channels=mid_ch,
                         kernel_size=1, stride=1, padding=0)
            )
            self.stage_norms.append(nn.GroupNorm(num_channels=out_ch, num_groups=_safe_groups(out_ch)))


    def init_weights(self):
        if self.with_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def _target_size(self, patch_resolution, level_idx):
        base_h, base_w = self.img_size
        pyramid_scales = [8, 4, 2, 1]
        divisor = pyramid_scales[level_idx] if level_idx < len(pyramid_scales) else 1
        return max(1, base_h // divisor), max(1, base_w // divisor)

    def _apply_norm_token(self, patch_token, norm_layer):
        if isinstance(norm_layer, nn.LayerNorm):
            return norm_layer(patch_token)
        if isinstance(norm_layer, nn.BatchNorm1d):
            b, h, w, c = patch_token.shape
            x = patch_token.reshape(b * h * w, c)
            x = norm_layer(x)
            return x.reshape(b, h, w, c)
        return patch_token

    def forward(self, x):
        x, patch_resolution = self.patch_embed(x)
        if self.with_pos_embed:
            pos_embed = resize_pos_embed(self.pos_embed, self.patch_resolution, patch_resolution,
                                         mode=self.interpolate_mode, num_extra_tokens=0)
            x = x + pos_embed
        x = self.drop_after_pos(x)

        outs = []
        proj_idx = 0
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.final_norm_layer(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm_layer{i}')
                    patch_token = self._apply_norm_token(patch_token, norm_layer)
                patch_token = patch_token.permute(0, 3, 1, 2)

                patch_token_mid = self.stage_norms[proj_idx](self.stage_projs[proj_idx](patch_token))
                target_size = self._target_size(patch_resolution, proj_idx)
                if patch_token_mid.shape[-2:] != target_size:
                    patch_token_mid = nn.functional.interpolate(
                        patch_token_mid,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False,
                    )
                outs.append(patch_token_mid)
                proj_idx += 1

        return outs
