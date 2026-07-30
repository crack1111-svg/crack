import torch
import torch.nn as nn

from models.GBC import BottConv, get_norm_layer


class DetailBranch(nn.Module):
    """轻量高分辨率细节分支，用于补偿裂缝边界与连续性。"""

    def __init__(self, in_channels=3, out_channels=24, norm_type='GN'):
        super().__init__()
        mid = max(8, out_channels // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm_layer(norm_type, mid, max(1, mid // 8)),
            nn.ReLU(inplace=True),
            BottConv(mid, out_channels, mid_channels=max(4, mid // 2), kernel_size=3, stride=1, padding=1),
            get_norm_layer(norm_type, out_channels, max(1, out_channels // 8)),
            nn.ReLU(inplace=True),
            BottConv(out_channels, out_channels, mid_channels=max(4, out_channels // 2), kernel_size=3, stride=1, padding=1),
            get_norm_layer(norm_type, out_channels, max(1, out_channels // 8)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
