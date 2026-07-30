import os
import random
from typing import Tuple

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from .image_folder import make_dataset


class CrackDataset(BaseDataset):
    """裂缝分割数据集。

    目录结构：
        dataset_path/train/image/*.png
        dataset_path/train/seg_gt/*.png
        dataset_path/val/image/*.png
        dataset_path/val/seg_gt/*.png
        dataset_path/test/image/*.png
        dataset_path/test/seg_gt/*.png
    """

    def __init__(self, args):
        super().__init__(args)
        self.phase = args.phase
        self.img_dir = os.path.join(args.dataset_path, self.phase, 'image')
        self.lab_dir = os.path.join(args.dataset_path, self.phase, 'seg_gt')

        self.img_paths = make_dataset(self.img_dir)
        self.img_paths.sort()

        self.load_size = int(args.load_size)
        self.train_crop_size = int(args.train_crop_size)
        self.train_use_crop = bool(getattr(args, 'train_use_crop', True))
        self.keep_aspect = bool(getattr(args, 'train_keep_aspect_ratio', True))
        self.train_use_augmentation = bool(getattr(args, 'train_use_augmentation', True))

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _normalize_image(img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float()

    @staticmethod
    def _mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
        mask = (mask > 0).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)

    @staticmethod
    def _pad_to_min_size(img: np.ndarray, lab: np.ndarray, min_h: int, min_w: int):
        h, w = img.shape[:2]
        pad_h = max(0, min_h - h)
        pad_w = max(0, min_w - w)
        if pad_h == 0 and pad_w == 0:
            return img, lab
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
        lab = cv2.copyMakeBorder(lab, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)
        return img, lab

    def _resize_train(self, img: np.ndarray, lab: np.ndarray):
        base = self.load_size
        scale = random.uniform(self.args.scale_jitter_min, self.args.scale_jitter_max)
        target = max(128, int(round(base * scale)))
        h, w = img.shape[:2]

        if self.keep_aspect:
            if h >= w:
                new_w = target
                new_h = int(round(h * target / max(1, w)))
            else:
                new_h = target
                new_w = int(round(w * target / max(1, h)))
        else:
            new_h = target
            new_w = target

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        lab = cv2.resize(lab, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return img, lab

    def _random_crop(self, img: np.ndarray, lab: np.ndarray):
        crop = self.train_crop_size
        img, lab = self._pad_to_min_size(img, lab, crop, crop)
        h, w = lab.shape[:2]

        positive_crop_prob = float(getattr(self.args, 'positive_crop_prob', 0.7))
        min_fg_pixels = int(getattr(self.args, 'min_fg_pixels', 32))
        crop_max_retry = int(getattr(self.args, 'crop_max_retry', 10))

        use_positive = random.random() < positive_crop_prob and lab.sum() >= min_fg_pixels
        if use_positive:
            ys, xs = np.where(lab > 0)
            if len(xs) > 0:
                for _ in range(crop_max_retry):
                    idx = random.randint(0, len(xs) - 1)
                    cx, cy = xs[idx], ys[idx]
                    x1 = max(0, min(w - crop, cx - random.randint(0, crop - 1)))
                    y1 = max(0, min(h - crop, cy - random.randint(0, crop - 1)))
                    patch_lab = lab[y1:y1 + crop, x1:x1 + crop]
                    if patch_lab.sum() >= min_fg_pixels:
                        return img[y1:y1 + crop, x1:x1 + crop], patch_lab

        x1 = random.randint(0, max(0, w - crop))
        y1 = random.randint(0, max(0, h - crop))
        return img[y1:y1 + crop, x1:x1 + crop], lab[y1:y1 + crop, x1:x1 + crop]

    def _augment(self, img: np.ndarray, lab: np.ndarray):
        if not self.train_use_augmentation or self.phase != 'train':
            return img, lab

        if getattr(self.args, 'aug_hflip', True) and random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            lab = np.ascontiguousarray(lab[:, ::-1])

        if getattr(self.args, 'aug_vflip', False) and random.random() < 0.3:
            img = np.ascontiguousarray(img[::-1, :])
            lab = np.ascontiguousarray(lab[::-1, :])

        if getattr(self.args, 'aug_rotate90', True) and random.random() < 0.5:
            k = random.randint(0, 3)
            img = np.ascontiguousarray(np.rot90(img, k))
            lab = np.ascontiguousarray(np.rot90(lab, k))

        if getattr(self.args, 'aug_brightness_contrast', True) and random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if getattr(self.args, 'aug_blur', True) and random.random() < 0.2:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), sigmaX=0)

        if getattr(self.args, 'aug_noise', True) and random.random() < 0.2:
            noise = np.random.normal(0, 8.0, size=img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img, lab

    def _load_pair(self, img_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
        img_name = os.path.basename(img_path)
        lab_name = img_name.replace('image', 'target', 1)
        lab_path = os.path.join(self.lab_dir, lab_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        if lab is None:
            raise FileNotFoundError(f'[Err] Label not found: {lab_path}')

        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)
        return img, lab, lab_path

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img, lab, lab_path = self._load_pair(img_path)

        if self.phase == 'train':
            img, lab = self._resize_train(img, lab)
            if self.train_use_crop:
                img, lab = self._random_crop(img, lab)
            else:
                img = cv2.resize(img, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
                lab = cv2.resize(lab, (self.load_size, self.load_size), interpolation=cv2.INTER_NEAREST)
            img, lab = self._augment(img, lab)
        else:
            img = cv2.resize(img, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (self.load_size, self.load_size), interpolation=cv2.INTER_NEAREST)

        img_tensor = self._normalize_image(img)
        lab_tensor = self._mask_to_tensor(lab)

        return {
            'image': img_tensor,
            'label': lab_tensor,
            'A_paths': img_path,
            'B_paths': lab_path,
        }
