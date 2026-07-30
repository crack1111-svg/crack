# StructCrack Code

This directory contains the implementation of the StructCrack crack segmentation model.

## Components

- `mmcls/SAVSS_dev/`: structure-aware Mamba encoder;
- `models/detail_branch.py`: high-resolution detail branch;
- `models/decoder.py`: detail-aware hierarchical decoder;
- `models/GBC.py`, `models/MFS.py`, `models/PAF.py`: feature processing and fusion;
- `datasets/crack_dataset.py`: paired image-mask loader and augmentation;
- `main.py`: training;
- `test.py`: checkpoint inference;
- `eval_compute.py` and `eval/`: metric computation.

The model preview is available at [`docs/figures/model.png`](docs/figures/model.png).

## Installation

```bash
conda create -n structcrack python=3.10 -y
conda activate structcrack
pip install -r requirements.txt
```

Reference versions are PyTorch 1.13.1, torchvision 0.14.1, CUDA 11.6, and Mamba-SSM 1.2.0. Use versions compatible with the local CUDA driver for compiled packages.

## Dataset layout

```text
DATASET_ROOT/
|-- train/image/
|-- train/seg_gt/
|-- val/image/
|-- val/seg_gt/
|-- test/image/
`-- test/seg_gt/
```

The loader reads paired images and masks by filename and binarizes masks at threshold 127.

## Commands

```bash
python main.py --dataset_path /path/to/DATASET_ROOT \
  --exp_preset high_acc \
  --output_dir ./logs/checkpoints/structcrack_highacc
```

```bash
python test.py --help
python test.py
python eval_compute.py
python eval/evaluate.py
```

Available experiment presets are defined in `option.py`, including `baseline`, `high_acc`, `lightweight`, and ablation configurations.

## Code Availability

<https://github.com/crack1111-svg/crack>
