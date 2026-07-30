'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from typing import Iterable
import time

import torch
from tqdm import tqdm

from util.model_output import get_main_logits


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, args=None, logger=None):
    model.train()
    criterion.train()

    device = torch.device(args.device)
    pbar = tqdm(total=len(data_loader.dataloader), desc='Initial Loss: Pending')
    for _, data in enumerate(data_loader):
        samples = data['image'].to(device)
        targets = data['label'].to(device=device, dtype=torch.float32)

        output = model(samples)
        logits = get_main_logits(output)
        loss_final = criterion(output, targets)
        cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))

        loss_final_str = '{:.4f}'.format(loss_final.item())
        l = optimizer.param_groups[0]['lr']
        if logger is not None:
            logger.info(
                f"time -> {cur_time} | Epoch -> {epoch} | image_num -> {data['A_paths']} | "
                f"loss final -> {loss_final_str} | lr -> {l} | pred_shape -> {tuple(logits.shape)}"
            )

        pbar.set_description(f'Loss: {loss_final.item():.4f}')
        pbar.update(1)
        optimizer.zero_grad()
        loss_final.backward()

        clip_norm = float(getattr(args, 'grad_clip_norm', 0.0)) if args is not None else 0.0
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

    pbar.close()
