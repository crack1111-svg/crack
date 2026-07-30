'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
from models.MFS import MFS
from models.detail_branch import DetailBranch


class Decoder(nn.Module):
    def __init__(self, backbone, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.use_detail_branch = bool(getattr(args, 'use_detail_branch', False))
        self.use_deep_supervision = bool(getattr(args, 'use_deep_supervision', False))
        self.use_boundary_head = bool(getattr(args, 'use_boundary_head', False))
        stage_dims = list(getattr(args, 'backbone_stage_out_channels', [128, 64, 32, 16]))

        detail_channels = int(getattr(args, 'detail_channels', 24)) if self.use_detail_branch else 0
        if self.use_detail_branch:
            self.detail_branch = DetailBranch(in_channels=3, out_channels=detail_channels,
                                              norm_type=getattr(args, 'Norm_Type', 'GN'))
        else:
            self.detail_branch = None

        self.MFS = MFS(
            input_dims=stage_dims[:4],
            embedding_dim=int(getattr(args, 'decoder_embedding_dim', 12)),
            norm_type=getattr(args, 'Norm_Type', 'GN'),
            fusion_mode=getattr(args, 'fusion_mode', 'hierarchical'),
            use_feature_attention=bool(getattr(args, 'use_feature_attention', True)),
            detail_channels=detail_channels,
            upsample_mode='dysample',
        )

        decoder_dim = int(getattr(args, 'decoder_embedding_dim', 12))
        self.linear_pred = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.out_refine = nn.Conv2d(1, 1, kernel_size=1)

        if self.use_deep_supervision:
            self.aux_heads = nn.ModuleList([nn.Conv2d(decoder_dim, 1, kernel_size=1) for _ in range(4)])
        else:
            self.aux_heads = nn.ModuleList()

        self.boundary_head = nn.Conv2d(decoder_dim, 1, kernel_size=1) if self.use_boundary_head else None

    def forward(self, samples):
        outs_savss = self.backbone(samples)
        detail_feat = self.detail_branch(samples) if self.detail_branch is not None else None
        fused_feat, aux_feats = self.MFS(outs_savss, detail_feat=detail_feat)

        logits = self.out_refine(self.linear_pred(fused_feat))

        output = {'logits': logits}

        if self.use_deep_supervision:
            target_size = logits.shape[-2:]
            aux_logits = []
            for feat, head in zip(aux_feats, self.aux_heads):
                pred = head(feat)
                if pred.shape[-2:] != target_size:
                    pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
                aux_logits.append(pred)
            output['aux_logits'] = aux_logits

        if self.boundary_head is not None:
            boundary_logits = self.boundary_head(fused_feat)
            if boundary_logits.shape[-2:] != logits.shape[-2:]:
                boundary_logits = F.interpolate(boundary_logits, size=logits.shape[-2:], mode='bilinear', align_corners=False)
            output['boundary_logits'] = boundary_logits

        return output


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, dims=(-2, -1)):
        super().__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, dims=(-2, -1)):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating = (1 - p_t) ** self.gamma
        loss = alpha_factor * modulating * bce
        return loss.mean()


class CompositeSegLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pos_weight = torch.tensor([float(getattr(args, 'bce_pos_weight', 3.0))]) if getattr(args, 'use_weighted_bce', True) else None
        self.register_buffer('bce_pos_weight', pos_weight if pos_weight is not None else torch.tensor([1.0]), persistent=False)
        self.dice_fn = DiceLoss()
        self.tversky_fn = TverskyLoss(alpha=float(getattr(args, 'tversky_alpha', 0.3)),
                                      beta=float(getattr(args, 'tversky_beta', 0.7)))
        self.focal_fn = FocalLoss(alpha=float(getattr(args, 'focal_alpha', 0.25)),
                                  gamma=float(getattr(args, 'focal_gamma', 2.0)))
        self.aux_weights = list(getattr(args, 'aux_loss_weights', [1.0, 0.8, 0.6, 0.4]))

    def _bce(self, logits, targets):
        pos_weight = None
        if bool(getattr(self.args, 'use_weighted_bce', True)):
            pos_weight = self.bce_pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    def _boundary_target(self, targets):
        k = int(getattr(self.args, 'boundary_kernel_size', 3))
        if k < 3:
            k = 3
        pad = k // 2
        dilated = F.max_pool2d(targets, kernel_size=k, stride=1, padding=pad)
        eroded = -F.max_pool2d(-targets, kernel_size=k, stride=1, padding=pad)
        boundary = torch.clamp(dilated - eroded, 0.0, 1.0)
        return boundary

    def _main_loss(self, logits, targets):
        loss = 0.0
        bce = self._bce(logits, targets)
        dice = self.dice_fn(torch.sigmoid(logits), targets)
        loss = float(getattr(self.args, 'BCELoss_ratio', 0.6)) * bce + float(getattr(self.args, 'DiceLoss_ratio', 0.4)) * dice

        if bool(getattr(self.args, 'use_tversky_loss', False)):
            loss = loss + float(getattr(self.args, 'tversky_loss_weight', 0.2)) * self.tversky_fn(torch.sigmoid(logits), targets)
        if bool(getattr(self.args, 'use_focal_loss', False)):
            loss = loss + float(getattr(self.args, 'focal_loss_weight', 0.0)) * self.focal_fn(logits, targets)
        return loss

    def forward(self, output, targets):
        if isinstance(output, torch.Tensor):
            return self._main_loss(output, targets)

        logits = output['logits']
        total_loss = self._main_loss(logits, targets)

        if bool(getattr(self.args, 'use_deep_supervision', False)) and 'aux_logits' in output:
            aux_total = 0.0
            weight_sum = 0.0
            for idx, aux in enumerate(output['aux_logits']):
                w = self.aux_weights[idx] if idx < len(self.aux_weights) else self.aux_weights[-1]
                aux_total = aux_total + w * self._main_loss(aux, targets)
                weight_sum += w
            if weight_sum > 0:
                total_loss = total_loss + float(getattr(self.args, 'aux_loss_weight', 0.25)) * (aux_total / weight_sum)

        if bool(getattr(self.args, 'use_boundary_head', False)) and 'boundary_logits' in output:
            boundary_target = self._boundary_target(targets)
            boundary_loss = self._bce(output['boundary_logits'], boundary_target) + \
                            self.dice_fn(torch.sigmoid(output['boundary_logits']), boundary_target)
            total_loss = total_loss + float(getattr(self.args, 'boundary_loss_weight', 0.2)) * boundary_loss

        return total_loss


def build(args):
    device = torch.device(args.device)

    backbone = SAVSS(
        arch=getattr(args, 'backbone_name', 'savss_base'),
        img_size=args.load_size,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=float(getattr(args, 'backbone_drop_path_rate', 0.2)),
        final_norm=True,
        convert_syncbn=True,
        norm_type=getattr(args, 'Norm_Type', 'GN'),
        with_pos_embed=bool(getattr(args, 'backbone_with_pos_embed', False)),
        embed_dims=int(getattr(args, 'backbone_embed_dims', 224)),
        num_layers=int(getattr(args, 'backbone_num_layers', 4)),
        patch_size=int(getattr(args, 'backbone_patch_size', 4)),
        num_convs_patch_embed=int(getattr(args, 'backbone_num_convs_patch_embed', 2)),
        stage_out_channels=list(getattr(args, 'backbone_stage_out_channels', [128, 64, 32, 16])),
        gbc_repeats=int(getattr(args, 'backbone_gbc_repeats', 1)),
        use_paf=bool(getattr(args, 'backbone_use_paf', True)),
        paf_reduction=int(getattr(args, 'backbone_paf_reduction', 2)),
        d_state=int(getattr(args, 'backbone_d_state', 16)),
        expand=float(getattr(args, 'backbone_expand', 2.0)),
        conv_size=int(getattr(args, 'backbone_conv_size', 7)),
    )
    model = Decoder(backbone, args)
    criterion = CompositeSegLoss(args)
    criterion.to(device)

    return model, criterion
