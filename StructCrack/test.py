'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets import create_dataset
from models import build_model
from option import finalize_args, get_args_parser
from util.model_output import get_main_logits
from util.model_profile import profile_model


def tensor_mask_to_uint8(mask_tensor):
    mask = mask_tensor.detach().float().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return ((mask > 0.5).astype(np.uint8) * 255)


def tensor_prob_to_uint8(logit_tensor):
    prob = torch.sigmoid(logit_tensor.detach()).float().cpu().numpy()
    if prob.ndim == 3:
        prob = prob[0]
    prob = np.clip(prob, 0.0, 1.0)
    return (prob * 255.0).round().astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/weights/checkpoint_TUT/checkpoint_TUT.pth',
                        help='Path to the checkpoint file to evaluate.')
    parser.add_argument('--save_root', type=str, default='',
                        help='Directory to save prediction images. Defaults to ./results/results_test/<checkpoint_parent>.')
    return finalize_args(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    args.phase = 'test'
    args.batch_size = args.batch_size_test
    device = torch.device(args.device)

    test_dl = create_dataset(args)
    load_model_file = args.checkpoint
    model, _ = build_model(args)
    state_dict = torch.load(load_model_file, map_location=device)
    model.load_state_dict(state_dict['model'], strict=False)
    model.to(device)
    print('Load Model Successful!')

    if bool(getattr(args, 'report_model_profile', True)):
        profile = profile_model(model=model, args=args, input_shape=(1, 3, int(args.load_size), int(args.load_size)), device='cpu')
        print(f"Params: {profile['params']:.0f} | Param Size(MB): {profile['param_size_mb']:.2f} | FLOPs: {profile['gflops']:.3f} GFLOPs")

    suffix = Path(load_model_file).parent.name
    save_root = Path(args.save_root) if args.save_root else Path('./results/results_test') / suffix
    save_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model.eval()
        pbar = tqdm(total=len(test_dl.dataloader), desc='Inference: Pending')
        for _, data in enumerate(test_dl):
            x = data['image'].to(device)
            target = data['label'].to(device=device, dtype=torch.float32)
            output = model(x)
            out = get_main_logits(output)

            for sample_idx in range(x.shape[0]):
                root_name = Path(data['A_paths'][sample_idx]).stem
                target_img = tensor_mask_to_uint8(target[sample_idx])
                pred_img = tensor_prob_to_uint8(out[sample_idx])

                cv2.imwrite(str(save_root / f'{root_name}_lab.png'), target_img)
                cv2.imwrite(str(save_root / f'{root_name}_pre.png'), pred_img)
            pbar.update(1)
        pbar.close()

    print('Finished!')
