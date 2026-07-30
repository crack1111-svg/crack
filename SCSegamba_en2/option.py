"""统一参数入口。

本文件的目标：
1. 将训练、测试、消融实验、轻量化实验、backbone 切换等参数全部集中到一个地方；
2. 所有关键参数都带中文注释，便于做论文复现与消融；
3. 通过 preset 一键切换 baseline / 高精度 / 轻量化 / 消融配置。
"""

from __future__ import annotations

import argparse
from typing import List


# -----------------------------
# 基础解析工具
# -----------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if v in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'无法解析布尔值: {v}')


class ParseIntList(argparse.Action):
    """将形如 1,2,3 的参数解析为 int 列表。"""
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            parsed = [int(v) for v in values]
        else:
            parsed = [int(v.strip()) for v in str(values).split(',') if v.strip()]
        setattr(namespace, self.dest, parsed)


class ParseFloatList(argparse.Action):
    """将形如 1.0,0.5,0.25 的参数解析为 float 列表。"""
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            parsed = [float(v) for v in values]
        else:
            parsed = [float(v.strip()) for v in str(values).split(',') if v.strip()]
        setattr(namespace, self.dest, parsed)


# -----------------------------
# 参数预设：方便做消融/参数实验
# -----------------------------

def apply_experiment_preset(args):
    """根据实验预设覆盖部分参数。

    说明：
    - baseline：尽量接近原始实现，便于公平对比；
    - high_acc：默认推荐配置，偏向提升裂缝分割精度；
    - lightweight：轻量部署配置，优先降低参数量与 FLOPs；
    - ablation_*：用于模块消融，尽量只关闭单个模块。
    """
    preset = getattr(args, 'exp_preset', 'high_acc')

    if preset == 'baseline':
        args.backbone_name = 'savss_base'
        args.backbone_with_pos_embed = True
        args.backbone_embed_dims = 256
        args.backbone_num_layers = 4
        args.backbone_patch_size = 8
        args.backbone_d_state = 16
        args.backbone_expand = 2.0
        args.backbone_conv_size = 7
        args.backbone_gbc_repeats = 2
        args.backbone_use_paf = True
        args.use_detail_branch = False
        args.use_deep_supervision = False
        args.use_boundary_head = False
        args.fusion_mode = 'concat'
        args.use_feature_attention = False
        args.use_weighted_bce = False
        args.use_focal_loss = False
        args.use_tversky_loss = False
        args.boundary_loss_weight = 0.0
        args.aux_loss_weight = 0.0
        args.train_use_augmentation = False

    elif preset == 'high_acc':
        args.backbone_name = 'savss_highacc'
        args.backbone_with_pos_embed = False
        args.backbone_embed_dims = 224
        args.backbone_num_layers = 4
        args.backbone_patch_size = 4
        args.backbone_d_state = 16
        args.backbone_expand = 2.0
        args.backbone_conv_size = 7
        args.backbone_gbc_repeats = 1
        args.backbone_use_paf = True
        args.use_detail_branch = True
        args.use_deep_supervision = True
        args.use_boundary_head = True
        args.fusion_mode = 'hierarchical'
        args.use_feature_attention = True
        args.use_weighted_bce = True
        args.use_focal_loss = False
        args.use_tversky_loss = True
        args.boundary_loss_weight = 0.20
        args.aux_loss_weight = 0.25
        args.train_use_augmentation = True
        args.positive_crop_prob = 0.75
        args.min_fg_pixels = 48

    elif preset == 'lightweight':
        args.backbone_name = 'savss_light'
        args.backbone_with_pos_embed = False
        args.backbone_embed_dims = 160
        args.backbone_num_layers = 4
        args.backbone_patch_size = 8
        args.backbone_d_state = 8
        args.backbone_expand = 1.5
        args.backbone_conv_size = 5
        args.backbone_gbc_repeats = 1
        args.backbone_use_paf = True
        args.use_detail_branch = True
        args.detail_channels = 16
        args.decoder_embedding_dim = 8
        args.use_deep_supervision = True
        args.use_boundary_head = False
        args.fusion_mode = 'hierarchical'
        args.use_feature_attention = True
        args.use_weighted_bce = True
        args.use_focal_loss = False
        args.use_tversky_loss = False
        args.boundary_loss_weight = 0.0
        args.aux_loss_weight = 0.20
        args.train_use_augmentation = True

    elif preset == 'ablation_detail_off':
        args.use_detail_branch = False
    elif preset == 'ablation_boundary_off':
        args.use_boundary_head = False
        args.boundary_loss_weight = 0.0
    elif preset == 'ablation_aux_off':
        args.use_deep_supervision = False
        args.aux_loss_weight = 0.0
    elif preset == 'ablation_pos_embed_on':
        args.backbone_with_pos_embed = True
    elif preset == 'ablation_paf_off':
        args.backbone_use_paf = False
    elif preset == 'ablation_gbc_repeat2':
        args.backbone_gbc_repeats = 2
    elif preset == 'custom':
        pass
    else:
        raise ValueError(f'不支持的 exp_preset: {preset}')

    return args


# -----------------------------
# 主解析器
# -----------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('SCSegamba Crack Segmentation', add_help=False)

    # ========== 实验控制 ==========
    parser.add_argument('--exp_preset', default='high_acc', type=str,
                        choices=['baseline', 'high_acc', 'lightweight', 'ablation_detail_off',
                                 'ablation_boundary_off', 'ablation_aux_off', 'ablation_pos_embed_on',
                                 'ablation_paf_off', 'ablation_gbc_repeat2', 'custom'],
                        help='实验预设：baseline/高精度/轻量化/消融实验')
    parser.add_argument('--experiment_name', default='scsegamba_refactor', type=str,
                        help='实验名称，用于日志与结果目录命名')

    # ========== 数据集与运行环境 ==========
    parser.add_argument('--dataset_path', default='/root/ykj/crack/data/CrackMap', type=str,
                        help='数据集根目录')
    parser.add_argument('--dataset_mode', default='crack', type=str,
                        help='数据集模式，默认 crack')
    parser.add_argument('--device', default='cuda', type=str,
                        help='运行设备，cuda 或 cpu')
    parser.add_argument('--seed', default=42, type=int,
                        help='随机种子')
    parser.add_argument('--num_threads', default=1, type=int,
                        help='DataLoader 线程数')
    parser.add_argument('--serial_batches', action='store_true',
                        help='若启用则不打乱训练数据')
    parser.add_argument('--phase', default='train', type=str,
                        help='运行阶段：train/val/test')

    # ========== 输入尺寸 ==========
    parser.add_argument('--load_size', default=512, type=int,
                        help='验证/测试统一缩放尺寸；训练增强时作为基础尺寸')
    parser.add_argument('--train_crop_size', default=512, type=int,
                        help='训练随机裁剪尺寸')
    parser.add_argument('--train_use_crop', default=True, type=str2bool,
                        help='是否启用训练裁剪')
    parser.add_argument('--train_keep_aspect_ratio', default=True, type=str2bool,
                        help='训练时是否尽量保持原图长宽比后再裁剪')
    parser.add_argument('--scale_jitter_min', default=0.75, type=float,
                        help='训练尺度抖动下界')
    parser.add_argument('--scale_jitter_max', default=1.50, type=float,
                        help='训练尺度抖动上界')

    # ========== 数据增强 ==========
    parser.add_argument('--train_use_augmentation', default=True, type=str2bool,
                        help='是否启用训练增强')
    parser.add_argument('--aug_hflip', default=True, type=str2bool,
                        help='随机水平翻转')
    parser.add_argument('--aug_vflip', default=False, type=str2bool,
                        help='随机垂直翻转')
    parser.add_argument('--aug_rotate90', default=True, type=str2bool,
                        help='随机 90/180/270 度旋转')
    parser.add_argument('--aug_brightness_contrast', default=True, type=str2bool,
                        help='随机亮度/对比度增强')
    parser.add_argument('--aug_blur', default=True, type=str2bool,
                        help='随机模糊')
    parser.add_argument('--aug_noise', default=True, type=str2bool,
                        help='随机高斯噪声')
    parser.add_argument('--positive_crop_prob', default=0.70, type=float,
                        help='正样本感知裁剪概率，提升细裂缝采样占比')
    parser.add_argument('--min_fg_pixels', default=32, type=int,
                        help='正样本裁剪时最少前景像素数')
    parser.add_argument('--crop_max_retry', default=10, type=int,
                        help='正样本裁剪最大重试次数')

    # ========== 训练超参数 ==========
    parser.add_argument('--batch_size_train', default=8, type=int,
                        help='训练 batch size')
    parser.add_argument('--batch_size_test', default=16, type=int,
                        help='验证/测试 batch size')
    parser.add_argument('--lr_scheduler', default='PolyLR', type=str,
                        help='学习率调度器：PolyLR / StepLR / CosLR')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='初始学习率')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='最小学习率')
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='权重衰减')
    parser.add_argument('--epochs', default=80, type=int,
                        help='训练总 epoch')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='起始 epoch')
    parser.add_argument('--lr_drop', default=30, type=int,
                        help='StepLR 时的衰减周期')
    parser.add_argument('--sgd', action='store_true',
                        help='是否改用 SGD；默认使用 AdamW')
    parser.add_argument('--grad_clip_norm', default=1.0, type=float,
                        help='梯度裁剪阈值，<=0 表示关闭')
    parser.add_argument('--output_dir', default='./logs/checkpoints/', type=str,
                        help='模型与日志输出目录')

    # ========== Backbone 选择与结构参数 ==========
    parser.add_argument('--backbone_name', default='savss_highacc', type=str,
                        choices=['savss_tiny', 'savss_light', 'savss_base', 'savss_highacc'],
                        help='选择哪一种 backbone 配置')
    parser.add_argument('--backbone_with_pos_embed', default=False, type=str2bool,
                        help='是否启用可学习绝对位置编码')
    parser.add_argument('--backbone_embed_dims', default=224, type=int,
                        help='backbone token 通道数')
    parser.add_argument('--backbone_num_layers', default=4, type=int,
                        help='backbone 层数')
    parser.add_argument('--backbone_patch_size', default=4, type=int,
                        help='patch size，越小越利于细裂缝细节保留')
    parser.add_argument('--backbone_num_convs_patch_embed', default=2, type=int,
                        help='patch embedding 前的卷积层数')
    parser.add_argument('--backbone_drop_path_rate', default=0.2, type=float,
                        help='DropPath 比例')
    parser.add_argument('--backbone_d_state', default=16, type=int,
                        help='Mamba 状态维度 d_state')
    parser.add_argument('--backbone_expand', default=2.0, type=float,
                        help='Mamba 内部扩张倍数 expand')
    parser.add_argument('--backbone_conv_size', default=7, type=int,
                        help='Mamba 内部局部卷积核大小')
    parser.add_argument('--backbone_gbc_repeats', default=1, type=int,
                        help='每个 SAVSS layer 前置 GBC 重复次数，用于结构消融')
    parser.add_argument('--backbone_use_paf', default=True, type=str2bool,
                        help='是否使用 PAF 引导融合')
    parser.add_argument('--backbone_paf_reduction', default=2, type=int,
                        help='PAF 中间通道压缩比例，越大越轻量')
    parser.add_argument('--backbone_stage_out_channels', default=[128, 64, 32, 16],
                        action=ParseIntList,
                        help='backbone 四个输出层的通道数，用于解码器融合与消融')

    # ========== 解码器 / 融合模块 ==========
    parser.add_argument('--Norm_Type', default='GN', type=str,
                        help='归一化类型：GN / BN / IN')
    parser.add_argument('--decoder_embedding_dim', default=12, type=int,
                        help='解码器统一投影维度，越大越强，越小越轻')
    parser.add_argument('--fusion_mode', default='hierarchical', type=str,
                        choices=['concat', 'hierarchical'],
                        help='多尺度融合方式：直接拼接 / 分层递进融合')
    parser.add_argument('--use_feature_attention', default=True, type=str2bool,
                        help='是否使用通道注意力进行多尺度加权')
    parser.add_argument('--use_detail_branch', default=True, type=str2bool,
                        help='是否启用高分辨率细节分支')
    parser.add_argument('--detail_channels', default=24, type=int,
                        help='细节分支通道数')
    parser.add_argument('--use_deep_supervision', default=True, type=str2bool,
                        help='是否启用深监督')
    parser.add_argument('--aux_loss_weight', default=0.25, type=float,
                        help='深监督总损失权重')
    parser.add_argument('--aux_loss_weights', default=[1.0, 0.8, 0.6, 0.4],
                        action=ParseFloatList,
                        help='四个尺度辅助监督权重')
    parser.add_argument('--use_boundary_head', default=True, type=str2bool,
                        help='是否启用边界分支')
    parser.add_argument('--boundary_kernel_size', default=3, type=int,
                        help='边界真值提取时的形态学核大小')

    # ========== 损失函数配置 ==========
    # 兼容原始命名，便于老脚本继续使用
    parser.add_argument('--BCELoss_ratio', default=0.60, type=float,
                        help='原始 BCE 权重兼容参数')
    parser.add_argument('--DiceLoss_ratio', default=0.40, type=float,
                        help='原始 Dice 权重兼容参数')

    parser.add_argument('--use_weighted_bce', default=True, type=str2bool,
                        help='是否对 BCE 设置前景正样本权重')
    parser.add_argument('--bce_pos_weight', default=3.0, type=float,
                        help='BCE 中前景权重，适合裂缝前景稀疏场景')
    parser.add_argument('--use_focal_loss', default=False, type=str2bool,
                        help='是否启用 focal loss')
    parser.add_argument('--focal_loss_weight', default=0.0, type=float,
                        help='focal loss 权重')
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='focal gamma')
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help='focal alpha')
    parser.add_argument('--use_tversky_loss', default=True, type=str2bool,
                        help='是否启用 Tversky loss，适合细结构召回优化')
    parser.add_argument('--tversky_loss_weight', default=0.20, type=float,
                        help='Tversky loss 权重')
    parser.add_argument('--tversky_alpha', default=0.30, type=float,
                        help='Tversky alpha（控制 FP 惩罚）')
    parser.add_argument('--tversky_beta', default=0.70, type=float,
                        help='Tversky beta（控制 FN 惩罚，建议裂缝任务偏大）')
    parser.add_argument('--boundary_loss_weight', default=0.20, type=float,
                        help='边界损失权重')

    # ========== 测试 / 评估 ==========
    parser.add_argument('--save_root', default='', type=str,
                        help='测试时保存预测结果的目录')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='测试时加载的 checkpoint 路径')

    # ========== 模型统计 ==========
    parser.add_argument('--report_model_profile', default=True, type=str2bool,
                        help='是否在程序启动时统计参数量与 FLOPs')
    parser.add_argument('--profile_batch_size', default=1, type=int,
                        help='统计 Params/FLOPs 时的 batch size')

    return parser


# -----------------------------
# 参数后处理
# -----------------------------

def finalize_args(args):
    args = apply_experiment_preset(args)

    # 保证权重列表长度和输出尺度一致
    if len(args.aux_loss_weights) < len(args.backbone_stage_out_channels):
        last = args.aux_loss_weights[-1] if args.aux_loss_weights else 1.0
        args.aux_loss_weights = args.aux_loss_weights + [last] * (
            len(args.backbone_stage_out_channels) - len(args.aux_loss_weights)
        )
    args.aux_loss_weights = args.aux_loss_weights[:len(args.backbone_stage_out_channels)]

    # 与原始代码兼容：create_dataset 读取 batch_size
    if not hasattr(args, 'batch_size') or args.batch_size is None:
        args.batch_size = args.batch_size_train if args.phase == 'train' else args.batch_size_test

    return args
