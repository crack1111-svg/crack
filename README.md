# StructCrack / SCSegamba Crack Segmentation Code Release

This repository provides the research code for the manuscript **“StructCrack: A Lightweight Structure-Aware MambaNetwork for Fine-Grained Crack Segmentation.”** The implementation is contained in [`SCSegamba_en2/`](SCSegamba_en2/).

The model is designed for binary pixel-level segmentation of cracks in structural images. It contains a structure-aware visual state-space backbone, gated bottleneck convolution, structure-aware feature processing, multi-scale fusion, a detail branch, and a decoder. Training, inference, model profiling, and metric computation are provided as executable scripts.

## Visual method summary

```mermaid
flowchart LR
    I[Structural RGB image] --> P[Resize / normalize]
    P --> B[SAVSS backbone]
    B --> S[Structure-aware scanning]
    B --> G[Gated bottleneck convolution]
    S --> F[Multi-scale feature fusion]
    G --> F
    F --> D[Detail branch + decoder]
    D --> O[Binary crack mask]
```

The supplied directory did not contain raster figures, dataset images, prediction examples, or checkpoint files. To keep this release academically accurate, no unverified result image or numerical claim is embedded in this README. Qualitative figures can be added under `docs/figures/` only when the image source, dataset split, checkpoint, threshold, and license are recorded.

## Main contents

| Path | Description |
| --- | --- |
| [`SCSegamba_en2/main.py`](SCSegamba_en2/main.py) | Training entry point |
| [`SCSegamba_en2/test.py`](SCSegamba_en2/test.py) | Checkpoint-based inference |
| [`SCSegamba_en2/eval_compute.py`](SCSegamba_en2/eval_compute.py) | Prediction metric computation |
| [`SCSegamba_en2/option.py`](SCSegamba_en2/option.py) | Experiment presets and hyperparameters |
| [`SCSegamba_en2/models/`](SCSegamba_en2/models/) | Model modules, fusion, detail branch, and decoder |
| [`SCSegamba_en2/mmcls/SAVSS_dev/`](SCSegamba_en2/mmcls/SAVSS_dev/) | SAVSS implementation |
| [`SCSegamba_en2/datasets/`](SCSegamba_en2/datasets/) | Image-mask loading and training augmentation |
| [`SCSegamba_en2/eval/`](SCSegamba_en2/eval/) | F1, precision, recall, mIoU, ODS, and OIS evaluation |
| [`SCSegamba_en2/README.md`](SCSegamba_en2/README.md) | Full installation and reproducibility guide |
| [`SCSegamba_en2/CODE_AVAILABILITY.md`](SCSegamba_en2/CODE_AVAILABILITY.md) | Manuscript-facing code availability statement |

## Environment

The reference environment is Python 3.10, PyTorch 1.13.1, torchvision 0.14.1, CUDA 11.6, `mmcv-full`, Mamba-SSM 1.2.0, OpenCV, NumPy, and the packages listed in [`SCSegamba_en2/requirements.txt`](SCSegamba_en2/requirements.txt). Linux with an NVIDIA GPU is recommended because MMCV and Mamba-SSM include compiled components. Windows users may need to select compatible pre-built wheels or build tools.

```bash
cd SCSegamba_en2
conda create -n scsegamba python=3.10 -y
conda activate scsegamba
pip install -r requirements.txt
```

Install a PyTorch and torchvision wheel compatible with the local CUDA driver before installing compiled dependencies. Record the exact package versions and GPU used for any reported experiment.

## Dataset access and format

The loader expects paired images and binary masks in this form:

```text
DATASET_ROOT/
├── train/image/      training images
├── train/seg_gt/     training masks
├── val/image/        validation images
├── val/seg_gt/       validation masks
├── test/image/       test images
└── test/seg_gt/      test masks
```

The code supports a dataset root supplied through `--dataset_path`. Dataset images and annotations are not included in this repository. Users should obtain them from the official providers, comply with their licenses, and document the dataset version, access conditions, and split used.

Potential benchmark sources mentioned by the underlying project include [TUT](https://github.com/Karl1109/TUT), [DeepCrack](https://github.com/yhlleo/DeepCrack), [Crack500](https://github.com/fyangneil/pavement-crack-detection), and [CrackMap](https://github.com/niuchuangnn/CrackMap). These links are provided for dataset discovery only; users must verify the current official source and redistribution terms before use.

## Usage

From `SCSegamba_en2/`:

```bash
python main.py --dataset_path /path/to/DATASET_ROOT \
  --exp_preset high_acc \
  --output_dir ./logs/checkpoints/scsegamba_highacc
```

Available presets include `baseline`, `high_acc`, `lightweight`, and the ablation presets defined in `option.py`.

For inference, provide a compatible checkpoint through the arguments accepted by `test.py`:

```bash
python test.py --help
python test.py
```

For evaluation of generated predictions:

```bash
python eval_compute.py
python eval/evaluate.py
```

The evaluation implementation calculates threshold-dependent precision, recall, and F1, as well as mIoU, ODS, and OIS. Results are only comparable when preprocessing, mask thresholding, split, checkpoint, and evaluation protocol are held constant.

## Reproducibility and code availability

The public repository URL is:

<https://github.com/crack1111-svg/crack>

Before submitting the manuscript revision, create a versioned GitHub release and archive it with Zenodo or another recognised DOI-assigning repository. Insert the permanent DOI and release tag into [`SCSegamba_en2/CODE_AVAILABILITY.md`](SCSegamba_en2/CODE_AVAILABILITY.md) and the manuscript’s Code Availability section. The manuscript-matching release should include the source code, configuration, split/index files where permitted, environment description, and evaluation instructions.

## Limitations

This code release does not by itself establish generalization to all structures, materials, lighting conditions, sensors, or crack morphologies. Results may be affected by dataset bias, annotation quality, image resizing, threshold selection, and checkpoint choice. The model is a research tool and should not be used as the sole basis for structural safety decisions without independent engineering inspection and validation.

## Citation

Please cite the associated manuscript and the archived version of this code release using its permanent DOI. The underlying project provides the following citation record:

```bibtex
@inproceedings{liu2025scsegamba,
  title={SCSegamba: Lightweight Structure-Aware Vision Mamba for Crack Segmentation in Structures},
  author={Liu, Hui and Jia, Chen and Shi, Fan and Cheng, Xu and Chen, Shengyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License and third-party notices

The source tree contains or adapts third-party components, including MMCV/MMClassification-related utilities and Mamba-based components. Before publication, retain all upstream license and attribution notices and verify that redistribution is permitted. The final repository should include an explicit license file matching the rights granted for the complete source tree.
