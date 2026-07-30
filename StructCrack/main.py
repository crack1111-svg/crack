'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import argparse
import copy
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import util.misc as utils
from datasets import create_dataset
from engine import train_one_epoch
from eval.evaluate import eval
from models import build_model
from option import finalize_args, get_args_parser
from util.logger import get_logger
from util.model_output import get_main_logits
from util.model_profile import profile_model


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, eta_min=1e-6, begin=0, end=100, power=0.9, last_epoch=-1):
        self.eta_min = eta_min
        self.begin = begin
        self.end = max(end, begin + 1)
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cur_epoch = max(self.last_epoch, self.begin)
        factor = (1 - min(cur_epoch - self.begin, self.end - self.begin) / float(self.end - self.begin)) ** self.power
        return [self.eta_min + (base_lr - self.eta_min) * factor for base_lr in self.base_lrs]



def make_data_loader(args, phase, batch_size):
    loader_args = copy.deepcopy(args)
    loader_args.phase = phase
    loader_args.batch_size = batch_size
    return create_dataset(loader_args)


def tensor_mask_to_uint8(mask_tensor):
    mask = mask_tensor.detach().float().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def tensor_prob_to_uint8(logit_tensor):
    prob = torch.sigmoid(logit_tensor.detach()).float().cpu().numpy()
    if prob.ndim == 3:
        prob = prob[0]
    prob = np.clip(prob, 0.0, 1.0)
    return (prob * 255.0).round().astype(np.uint8)


def main(args):
    checkpoints_path = './logs/checkpoints'
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    process_folder_path = os.path.join(checkpoints_path, f'{cur_time}_{args.experiment_name}_Dataset->{dataset_name}')
    os.makedirs(process_folder_path, exist_ok=False)

    log_train = get_logger(process_folder_path, 'train')
    log_test = get_logger(process_folder_path, 'test')
    log_eval = get_logger(process_folder_path, 'eval')

    log_train.info('args -> %s', str(args))

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model, criterion = build_model(args)
    model.to(device)

    if bool(getattr(args, 'report_model_profile', True)):
        profile = profile_model(
            model=model,
            args=args,
            input_shape=(int(args.profile_batch_size), 3, int(args.load_size), int(args.load_size)),
            device='cpu',
        )
        profile_msg = (
            f"Model Profile | Params: {profile['params']:.0f} | Trainable: {profile['trainable_params']:.0f} | "
            f"Param Size(MB): {profile['param_size_mb']:.2f} | FLOPs: {profile['gflops']:.3f} GFLOPs"
        )
        print(profile_msg)
        log_train.info(profile_msg)
        with open(os.path.join(process_folder_path, 'model_profile.txt'), 'w', encoding='utf-8') as f:
            for k, v in profile.items():
                f.write(f'{k}: {v}\n')

    train_loader = make_data_loader(args, phase='train', batch_size=args.batch_size_train)
    test_dir = os.path.join(args.dataset_path, 'test')
    if os.path.exists(test_dir):
        test_loader = make_data_loader(args, phase='test', batch_size=args.batch_size_test)
    else:
        test_loader = make_data_loader(args, phase='val', batch_size=args.batch_size_test)

    dataset_size = len(train_loader)
    print('The number of training images = %d' % dataset_size)
    log_train.info('The number of training images = %d' % dataset_size)

    param_dicts = [{'params': [p for _, p in model.named_parameters()], 'lr': args.lr}]
    if args.sgd:
        print('use SGD!')
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        print('use AdamW!')
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)
    else:
        raise ValueError(f'Unsupported lr_scheduler: {args.lr_scheduler}')

    output_dir = Path(args.output_dir) / f'{cur_time}_{args.experiment_name}_Dataset->{dataset_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Start processing!')
    log_train.info('Start processing!')
    start_time = time.time()
    max_metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0, 'Best_thresh': 0}

    for epoch in range(args.start_epoch, args.epochs):
        print('---------------------------------------------------------------------------------------')
        print('training epoch start -> ', epoch)
        train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, log_train)
        lr_scheduler.step()

        checkpoint_paths = [output_dir / 'checkpoint.pth']
        checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        print('training epoch finish -> ', epoch)
        print('---------------------------------------------------------------------------------------')

        print('testing epoch start -> ', epoch)
        results_path = f'{cur_time}_{args.experiment_name}_Dataset->{dataset_name}'
        save_root = Path('./results') / results_path / f'results_{epoch}'
        save_root.mkdir(parents=True, exist_ok=True)
        pbar = tqdm(total=len(test_loader.dataloader), desc='Initial Loss: Pending')

        with torch.no_grad():
            model.eval()
            for _, data in enumerate(test_loader):
                x = data['image'].to(device)
                target = data['label'].to(device=device, dtype=torch.float32)
                output = model(x)
                out = get_main_logits(output)
                loss = criterion(output, target)

                batch_size = x.shape[0]
                for sample_idx in range(batch_size):
                    root_name = Path(data['A_paths'][sample_idx]).stem
                    target_img = tensor_mask_to_uint8(target[sample_idx])
                    pred_img = tensor_prob_to_uint8(out[sample_idx])

                    lab_path = save_root / f'{root_name}_lab.png'
                    pre_path = save_root / f'{root_name}_pre.png'
                    cv2.imwrite(str(lab_path), target_img)
                    cv2.imwrite(str(pre_path), pred_img)
                    log_test.info(f'image -> {root_name} | loss -> {loss.item():.4f}')
                pbar.set_description(f'Test Loss: {loss.item():.4f}')
                pbar.update(1)
        pbar.close()

        metrics = eval(log_eval, str(save_root), epoch)
        if metrics['mIoU'] >= max_metrics['mIoU']:
            max_metrics = metrics
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_metrics': metrics,
            }, output_dir / 'best_checkpoint.pth')

        log_eval.info(f'Current best -> {max_metrics}')

    total_time = time.time() - start_time
    total_time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
    print('Training time {}'.format(total_time_str))
    log_train.info('Training time {}'.format(total_time_str))
    log_eval.info(f'Best metrics -> {max_metrics}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
    args = parser.parse_args()
    args = finalize_args(args)
    main(args)
