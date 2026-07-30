import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model
from option import finalize_args, get_args_parser
from util.model_profile import profile_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Profile SCSegamba model', parents=[get_args_parser()])
    args = finalize_args(parser.parse_args())
    args.device = 'cpu'
    model, _ = build_model(args)
    profile = profile_model(model=model, args=args, input_shape=(1, 3, args.load_size, args.load_size), device='cpu')
    for k, v in profile.items():
        print(f'{k}: {v}')
