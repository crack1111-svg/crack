"""解析式模型统计工具。

说明：
- 参数量：若提供 model，则精确统计模型参数；
- FLOPs：基于当前架构的解析近似公式估计，不依赖前向推理，避免大模型/无 CUDA 环境下统计过慢。
"""

from __future__ import annotations

import math
from typing import Dict, Optional


def _conv2d_flops(h, w, cin, cout, k, stride=1, groups=1):
    out_h = math.ceil(h / stride)
    out_w = math.ceil(w / stride)
    return 2.0 * out_h * out_w * cout * (cin / groups) * k * k


def _linear_flops(tokens, din, dout):
    return 2.0 * tokens * din * dout


def _bottconv_params(cin, cout, mid, k, bias=True):
    return cin * mid + mid * k * k + mid * cout + (mid if bias else 0)


def _bottconv_flops(h, w, cin, cout, mid, k):
    pw1 = 2.0 * h * w * cin * mid
    dw = 2.0 * h * w * mid * k * k
    pw2 = 2.0 * h * w * mid * cout
    return pw1 + dw + pw2


def _gbc_params_flops(h, w, channels):
    mid = max(1, channels // 8)
    params = 0.0
    flops = 0.0
    for k in [3, 3, 1, 1]:
        params += _bottconv_params(channels, channels, mid, k, bias=True)
        flops += _bottconv_flops(h, w, channels, channels, mid, k)
    flops += h * w * channels * 3.0
    return params, flops


def _dy_upsample_flops(h, w, c, scale):
    # 近似估计：offset 1x1 + grid/sample 开销
    offset_channels = 2 * 4 * (scale ** 2)
    conv = 2.0 * h * w * c * offset_channels
    sample = 8.0 * (h * scale) * (w * scale) * c
    return conv + sample


def profile_model(model=None, args=None, input_shape=(1, 3, 512, 512), device='cpu') -> Dict[str, float]:
    batch, in_ch, h, w = input_shape
    params = float(sum(p.numel() for p in model.parameters())) if model is not None else 0.0
    trainable_params = float(sum(p.numel() for p in model.parameters() if p.requires_grad)) if model is not None else 0.0

    if args is None:
        return {
            'params': params,
            'trainable_params': trainable_params,
            'param_size_mb': params * 4 / (1024 ** 2),
            'flops': 0.0,
            'gflops': 0.0,
        }

    patch_size = int(getattr(args, 'backbone_patch_size', 8))
    embed_dims = int(getattr(args, 'backbone_embed_dims', 256))
    num_layers = int(getattr(args, 'backbone_num_layers', 4))
    num_patch_convs = int(getattr(args, 'backbone_num_convs_patch_embed', 2))
    d_state = int(getattr(args, 'backbone_d_state', 16))
    expand = float(getattr(args, 'backbone_expand', 2.0))
    conv_size = int(getattr(args, 'backbone_conv_size', 7))
    gbc_repeats = int(getattr(args, 'backbone_gbc_repeats', 1))
    use_paf = bool(getattr(args, 'backbone_use_paf', True))
    paf_reduction = int(getattr(args, 'backbone_paf_reduction', 2))
    stage_out_channels = list(getattr(args, 'backbone_stage_out_channels', [128, 64, 32, 16]))
    decoder_dim = int(getattr(args, 'decoder_embedding_dim', 12))
    use_detail = bool(getattr(args, 'use_detail_branch', False))
    detail_channels = int(getattr(args, 'detail_channels', 24)) if use_detail else 0
    use_aux = bool(getattr(args, 'use_deep_supervision', False))
    use_boundary = bool(getattr(args, 'use_boundary_head', False))
    fusion_mode = getattr(args, 'fusion_mode', 'hierarchical')

    total_flops = 0.0

    # Patch embedding
    stem_h, stem_w = h // 2, w // 2
    total_flops += _conv2d_flops(h, w, 3, 64, 7, stride=2)
    for _ in range(num_patch_convs):
        total_flops += _conv2d_flops(stem_h, stem_w, 64, 64, 3, stride=1)

    proj_k = patch_size // 2
    proj_stride = patch_size // 2
    patch_h, patch_w = stem_h // proj_stride, stem_w // proj_stride
    total_flops += _conv2d_flops(stem_h, stem_w, 64, embed_dims, proj_k, stride=proj_stride)

    # Backbone layers
    tokens = patch_h * patch_w
    d_inner = int(embed_dims * expand)
    dt_rank = math.ceil(embed_dims / 16)
    paf_mid = max(1, embed_dims // max(1, paf_reduction))
    for _ in range(num_layers):
        gbc_p, gbc_f = _gbc_params_flops(patch_h, patch_w, embed_dims)
        total_flops += gbc_repeats * gbc_f

        total_flops += _linear_flops(tokens, embed_dims, d_inner * 2)
        total_flops += _bottconv_flops(patch_h, patch_w, d_inner, d_inner, max(1, d_inner // 16), conv_size)
        total_flops += _linear_flops(tokens, d_inner, dt_rank + 2 * d_state)
        total_flops += _linear_flops(tokens, dt_rank, d_inner)
        total_flops += 8.0 * 4.0 * tokens * d_inner * d_state
        total_flops += _linear_flops(tokens, d_inner, embed_dims)

        if use_paf:
            total_flops += 2 * _bottconv_flops(patch_h, patch_w, embed_dims, paf_mid, max(1, paf_mid // 2), 1)
            total_flops += _bottconv_flops(patch_h, patch_w, paf_mid, embed_dims, max(1, paf_mid // 2), 1)
            total_flops += 3.0 * patch_h * patch_w * embed_dims
        total_flops += _linear_flops(tokens, embed_dims, embed_dims)

    # Stage projection heads in backbone
    for out_ch in stage_out_channels[:4]:
        total_flops += _bottconv_flops(patch_h, patch_w, embed_dims, out_ch, max(4, out_ch // 4), 1)

    # Spatial sizes of 4 pyramid outputs: 1/8, 1/4, 1/2, 1
    pyramid_sizes = [(max(1, h // 8), max(1, w // 8)), (max(1, h // 4), max(1, w // 4)),
                     (max(1, h // 2), max(1, w // 2)), (h, w)]

    # Decoder projector MLPs
    for (sh, sw), in_dim in zip(pyramid_sizes, stage_out_channels[:4]):
        total_flops += _linear_flops(sh * sw, in_dim, decoder_dim)

    # DySample / upsample
    for (sh, sw), scale in zip(pyramid_sizes[:3], [8, 4, 2]):
        total_flops += _dy_upsample_flops(sh, sw, decoder_dim, scale)

    # Fuse blocks
    if fusion_mode == 'concat':
        total_flops += _bottconv_flops(h, w, decoder_dim * 4, decoder_dim, max(1, decoder_dim), 1)
    else:
        total_flops += _gbc_params_flops(h, w, decoder_dim * 2)[1] + _bottconv_flops(h, w, decoder_dim * 2, decoder_dim, max(4, decoder_dim), 1)
        total_flops += _gbc_params_flops(h, w, decoder_dim * 2)[1] + _bottconv_flops(h, w, decoder_dim * 2, decoder_dim, max(4, decoder_dim), 1)
        total_flops += _gbc_params_flops(h, w, decoder_dim * 2)[1] + _bottconv_flops(h, w, decoder_dim * 2, decoder_dim, max(4, decoder_dim), 1)

    final_in = decoder_dim + detail_channels
    total_flops += _gbc_params_flops(h, w, final_in)[1] + _bottconv_flops(h, w, final_in, decoder_dim, max(4, decoder_dim), 1)

    # Detail branch
    if use_detail:
        mid = max(8, detail_channels // 2)
        total_flops += _conv2d_flops(h, w, 3, mid, 3, stride=2)
        total_flops += _bottconv_flops(h // 2, w // 2, mid, detail_channels, max(4, mid // 2), 3)
        total_flops += _bottconv_flops(h // 2, w // 2, detail_channels, detail_channels, max(4, detail_channels // 2), 3)

    # Heads
    total_flops += _conv2d_flops(h, w, decoder_dim, 1, 1, stride=1)
    total_flops += _conv2d_flops(h, w, 1, 1, 1, stride=1)
    if use_aux:
        for sh, sw in pyramid_sizes:
            total_flops += _conv2d_flops(sh, sw, decoder_dim, 1, 1, stride=1)
    if use_boundary:
        total_flops += _conv2d_flops(h, w, decoder_dim, 1, 1, stride=1)

    param_size_mb = params * 4 / (1024 ** 2)
    return {
        'params': float(params),
        'trainable_params': float(trainable_params),
        'param_size_mb': float(param_size_mb),
        'flops': float(total_flops * batch),
        'gflops': float(total_flops * batch / 1e9),
    }
