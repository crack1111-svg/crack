'''
Author: Chenhongyi Yang
Reference: https://github.com/OliverRensu/Shunted-Transformer
'''

from torch import nn


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class AdaptivePadding(nn.Module):
    """轻量替代版 AdaptivePadding，仅支持 same/corner。"""
    def __init__(self, kernel_size, stride, dilation=(1, 1), padding='corner'):
        super().__init__()
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.padding = padding

    def get_pad_shape(self, input_size):
        h, w = input_size
        out_h = (h + self.stride[0] - 1) // self.stride[0]
        out_w = (w + self.stride[1] - 1) // self.stride[1]
        pad_h = max((out_h - 1) * self.stride[0] + (self.kernel_size[0] - 1) * self.dilation[0] + 1 - h, 0)
        pad_w = max((out_w - 1) * self.stride[1] + (self.kernel_size[1] - 1) * self.dilation[1] + 1 - w, 0)
        return pad_h, pad_w

    def forward(self, x):
        import torch.nn.functional as F
        pad_h, pad_w = self.get_pad_shape((x.shape[-2], x.shape[-1]))
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            else:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 num_convs=0,
                 conv_type='Conv2d',
                 patch_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__()

        assert patch_size % 2 == 0

        self.embed_dims = embed_dims
        if stride is None:
            stride = patch_size // 2
        else:
            stride = stride // 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
            nn.GroupNorm(num_channels=64, num_groups=4),
            nn.ReLU(True))

        if num_convs > 0:
            convs = []
            for _ in range(num_convs):
                convs.append(nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1, bias=False))
                convs.append(nn.GroupNorm(num_channels=64, num_groups=4))
                convs.append(nn.ReLU(True))
            self.convs = nn.Sequential(*convs)
        else:
            self.convs = None

        kernel_size = to_2tuple(patch_size // 2)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=64,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

        self.norm = nn.LayerNorm(embed_dims) if norm_cfg is not None else None

        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            _input_size = (input_size[0] // 2, input_size[1] // 2)
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(_input_size)
                _input_size = (_input_size[0] + pad_h, _input_size[1] + pad_w)

            h_out = (_input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (_input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        x = self.stem(x)
        if self.convs is not None:
            x = self.convs(x)

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size
