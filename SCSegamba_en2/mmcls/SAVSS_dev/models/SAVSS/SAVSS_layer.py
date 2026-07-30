'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch.nn.init import trunc_normal_

from models.GBC import BottConv, GBC
from models.PAF import PAF


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def build_dropout(cfg):
    if cfg is None:
        return nn.Identity()
    drop_prob = cfg.get('drop_prob', 0.0)
    if cfg.get('type', 'DropPath') == 'DropPath':
        return DropPath(drop_prob)
    return nn.Identity()


def selective_scan_fn(x, dt, A, B, C, D, z=None, delta_bias=None, delta_softplus=True, return_last_state=False):
    """纯 PyTorch 的快速近似版 selective scan。

    为了摆脱 mamba_ssm 依赖，这里使用向量化近似，保证前向可运行且速度可接受。
    该实现用于工程复现与消融入口，不等价于官方 CUDA/Triton 版本。
    """
    if delta_bias is not None:
        dt = dt + delta_bias.view(1, -1, 1)
    if delta_softplus:
        dt = F.softplus(dt)

    b_gain = B.mean(dim=1, keepdim=True)
    c_gain = C.mean(dim=1, keepdim=True)
    a_gain = torch.tanh(A.mean(dim=-1)).view(1, -1, 1)
    state_term = torch.tanh(dt) * x
    context = (b_gain * c_gain).expand(-1, x.shape[1], -1)
    y = (1 + D.view(1, -1, 1) + a_gain) * x + state_term + 0.1 * context
    last_state = (torch.tanh(dt.mean(dim=-1, keepdim=True)) * x.mean(dim=-1, keepdim=True)).unsqueeze(-1)
    if return_last_state:
        return y, last_state
    return y


class SAVSS_2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_size=7,
            bias=False,
            conv_bias=False,
            init_layer_scale=None,
            default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.n_directions = 4

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        conv_padding = conv_size // 2
        mid_channels = max(1, self.d_inner // 16)
        self.conv2d = BottConv(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            mid_channels=mid_channels,
            kernel_size=conv_size,
            padding=conv_padding,
            stride=1,
            bias=conv_bias,
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    def sass(self, hw_shape):
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        d1, d2, d3, d4 = [], [], [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d2.append(4)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d2.append(3)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "down"
        d2 = [0] + d2[:-1]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o3) - 1
                        d3.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o3) - 1
                        d3.append(4 if i == diag else 1)
        d3 = [0] + d3[:-1]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(4 if i == diag else 1)
        d4 = [0] + d4[:-1]

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse)), \
            (tuple(d1), tuple(d2), tuple(d3), tuple(d4))

    def _reorder_ssm_inputs(self, x_conv, dt, B, C, order, direction_bias):
        x_ordered = x_conv[:, order, :].permute(0, 2, 1).contiguous()
        dt_ordered = dt[:, :, order].contiguous()
        B_ordered = (B[:, :, order] + direction_bias).contiguous()
        C_ordered = C[:, :, order].contiguous()
        return x_ordered, dt_ordered, B_ordered, C_ordered

    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        ssm_state = None
        xz = self.in_proj(x)
        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=-1)
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        assert self.activation in ["silu", "swish"]

        orders, inverse_orders, directions = self.sass(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        direction_Bs = [
            dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype, device=B.device)
            for dB in direction_Bs
        ]

        y_scan = []
        for order, inv_order, dB in zip(orders, inverse_orders, direction_Bs):
            x_ordered, dt_ordered, B_ordered, C_ordered = self._reorder_ssm_inputs(x_conv, dt, B, C, order, dB)
            y_dir = selective_scan_fn(
                x_ordered,
                dt_ordered,
                A,
                B_ordered,
                C_ordered,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            y_scan.append(y_dir.permute(0, 2, 1)[:, inv_order, :])

        y = sum(y_scan) * self.act(z)
        out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out


class SAVSS_Layer(nn.Module):
    def __init__(
            self,
            embed_dims,
            use_rms_norm,
            with_dwconv,
            drop_path_rate,
            mamba_cfg,
            norm_type='GN',
            gbc_repeats=2,
            use_paf=True,
            paf_reduction=2,
    ):

        super(SAVSS_Layer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})
        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        self.with_dwconv = with_dwconv
        self.use_paf = use_paf
        self.gbc_repeats = max(0, int(gbc_repeats))

        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                    groups=embed_dims
                ),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )

        self.SAVSS_2D = SAVSS_2D(**mamba_cfg)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        gn_groups = max(1, embed_dims // 16)
        self.linear_proj = nn.Linear(in_features=embed_dims, out_features=embed_dims, bias=True)
        self.channel_norm = nn.GroupNorm(num_channels=embed_dims, num_groups=gn_groups)
        self.GBC_C = GBC(embed_dims, norm_type=norm_type)
        paf_mid_channels = max(1, embed_dims // max(1, int(paf_reduction)))
        self.PAF = PAF(embed_dims, paf_mid_channels, norm_type=norm_type) if self.use_paf else None

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        if H * W != L:
            raise ValueError(f'hw_shape {hw_shape} is inconsistent with sequence length {L}')
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        for _ in range(self.gbc_repeats):
            x = self.GBC_C(x)

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        mixed_x = self.drop_path(self.SAVSS_2D(self.norm(x), hw_shape))
        b, _, c = mixed_x.shape
        h, w = hw_shape

        base_feat = x.permute(0, 2, 1).reshape(b, c, h, w)
        guidance_feat = mixed_x.permute(0, 2, 1).reshape(b, c, h, w)
        if self.PAF is not None:
            mixed_x = self.PAF(base_feat, guidance_feat)
        else:
            mixed_x = guidance_feat + base_feat

        mixed_x = self.channel_norm(mixed_x).reshape(b, c, h * w).permute(0, 2, 1)

        if self.with_dwconv:
            mixed_x = mixed_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            mixed_x = self.GBC_C(mixed_x)
            mixed_x = mixed_x.reshape(b, c, h * w).permute(0, 2, 1)

        mixed_x_res = self.linear_proj(self.channel_norm(mixed_x.permute(0, 2, 1)).permute(0, 2, 1))
        return mixed_x + mixed_x_res
