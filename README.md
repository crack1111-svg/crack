# StructCrack

Lightweight structure-aware Mamba network for fine-grained crack segmentation.

StructCrack performs binary pixel-level segmentation of cracks in structural images. The implementation includes a structure-aware Mamba encoder, detail-aware hierarchical decoder, output refinement, training, inference, profiling, and evaluation.

## Model

![StructCrack model architecture](StructCrack/docs/figures/model.png)

The model figure is available as [`model.png`](StructCrack/docs/figures/model.png). PDF files are not included in this GitHub release.

## Repository

```text
.
|-- StructCrack/
|   |-- datasets/       Dataset loading and augmentation
|   |-- eval/           Segmentation evaluation
|   |-- models/         Decoder and feature modules
|   |-- mmcls/          SAVSS and supporting modules
|   |-- main.py         Training entry point
|   |-- test.py         Inference entry point
|   |-- option.py       Experiment configuration
|   `-- requirements.txt
|-- README.md
`-- .gitignore
```

## Environment

Reference environment: Python 3.10, PyTorch 1.13.1, torchvision 0.14.1, CUDA 11.6, MMCV, Mamba-SSM 1.2.0, OpenCV, NumPy, and the packages in [`StructCrack/requirements.txt`](StructCrack/requirements.txt).

```bash
cd StructCrack
conda create -n structcrack python=3.10 -y
conda activate structcrack
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch and torchvision build before compiled dependencies such as MMCV and Mamba-SSM.

## Dataset format

```text
DATASET_ROOT/
|-- train/image/
|-- train/seg_gt/
|-- val/image/
|-- val/seg_gt/
|-- test/image/
`-- test/seg_gt/
```

Images and masks are paired by filename. Dataset images and annotations are not included in this repository.

## Usage

Train:

```bash
cd StructCrack
python main.py --dataset_path /path/to/DATASET_ROOT \
  --exp_preset high_acc \
  --output_dir ./logs/checkpoints/structcrack_highacc
```

Inference:

```bash
cd StructCrack
python test.py --help
python test.py
```

Evaluation:

```bash
cd StructCrack
python eval_compute.py
python eval/evaluate.py
```

## Code Availability

Source code: <https://github.com/crack1111-svg/crack>
