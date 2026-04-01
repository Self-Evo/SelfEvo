# SelfEvo: Self-Improving 4D Perception via Self-Distillation

[**Arxiv**](#citation) | [**Website**](https://self-evo.github.io/) | [**SelfEvo Model**](https://huggingface.co/Changearthmore/SelfEvoVGGT)

**[Nan Huang](https://nnanhuang.github.io/)<sup>1,2,\*</sup>, [Pengcheng Yu](https://github.com/yupengchengg147)<sup>2,4,\*</sup>, [Weijia Zeng](https://fantasticoven2.github.io/)<sup>6</sup>, [James M. Rehg](https://rehg.org/)<sup>1</sup>, [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)<sup>5</sup>, [Haiwen Feng](https://havenfeng.github.io/)<sup>2,5,†</sup>, [Qianqian Wang](https://qianqianwang68.github.io/)<sup>3,†</sup>**

<sup>1</sup>UIUC &nbsp; <sup>2</sup>Impossible Inc. &nbsp; <sup>3</sup>Harvard &nbsp; <sup>4</sup>Max Planck Institute &nbsp; <sup>5</sup>UC Berkeley &nbsp; <sup>6</sup>UBC

<sup>\*</sup>Equal contribution &nbsp; <sup>†</sup>Equal advising

---

SelfEvo is a self-improving framework that continually improves pretrained multi-view reconstruction models (e.g., [VGGT](https://github.com/facebookresearch/vggt)) using unlabeled videos — **no 3D annotations required**. It introduces a self-distillation scheme based on spatiotemporal context asymmetry, where a richer-context teacher provides pseudo supervision to a student operating on a dropped subset of frames. The teacher is updated as an EMA of the student, forming a fully online self-improving loop.

SelfEvo achieves up to **36.5% relative improvement** in video depth estimation and **20.1%** in camera estimation over pretrained baselines, without using any labeled data.

## Updates

- **[2026-04]** Paper and code released.

## Installation

**1. Create a Python >= 3.10 environment:**

```bash
conda create -n selfevo python=3.10
conda activate selfevo
```

**2. Install training dependencies:**

```bash
pip install -r requirements.txt
pip install -e .
```

**3. (Optional) Install demo dependencies:**

```bash
pip install -r requirements_demo.txt
```

## Pretrained Models

**Base VGGT-1B** (pretrained):

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="facebook/VGGT-1B", filename="model.pt", local_dir="ckpt/")
```

Or via CLI:
```bash
huggingface-cli download facebook/VGGT-1B model.pt --local-dir ckpt/
```

**SelfEvo (VGGT)** — self-improved checkpoint:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Changearthmore/SelfEvoVGGT", filename="model.pt", local_dir="ckpt/selfevo/")
```

Or via CLI:
```bash
huggingface-cli download Changearthmore/SelfEvoVGGT model.pt --local-dir ckpt/selfevo/
```

Update `checkpoint.resume_checkpoint_path` in your config to point to the downloaded checkpoint.

## Interactive Demo

Try the interactive Gradio demo on your own images or videos.

**Pretrained VGGT-1B:**
```bash
pip install -r requirements_demo.txt
python demo_gradio.py --model facebook/VGGT-1B
```

**SelfEvo (self-improved):**
```bash
python demo_gradio.py --model Changearthmore/SelfEvoVGGT
```

For 3D visualization with [Viser](https://viser.studio/):

**Pretrained VGGT-1B:**
```bash
python demo_viser.py --model facebook/VGGT-1B --image_folder examples/drift-chicane/
```

**SelfEvo (self-improved):**
```bash
python demo_viser.py --model Changearthmore/SelfEvoVGGT --image_folder examples/drift-chicane/
```

## Data Preparation

The training dataset should be organized as a directory of video sequences, each containing frame images (`.jpg` or `.png`). **No ground-truth annotations are used.**

**Expected structure:**

```
/path/to/dataset/
├── sequence_001/
│   ├── frame_000.jpg
│   ├── frame_001.jpg
│   └── ...
├── sequence_002/
│   └── ...
└── ...
```

**To extract frames from videos using ffmpeg:**

```bash
ffmpeg -i video.mp4 -qscale:v 2 /path/to/output/%05d.jpg
```

**Datasets used in the paper:**
- [OmniWorld-Game](https://arxiv.org/abs/2509.12201) (primary training set)
- [BEDLAM2.0](https://bedlam.is.tue.mpg.de/) (synthetic human-centric video)
- [DROID](https://droid-dataset.github.io/) (real-world robot manipulation)

Evaluation benchmarks: Sintel, KITTI, Bonn (video depth); RealEstate10K (camera); DROID, HOI4D (OOD generalization).

## Training

Launch self-distillation training with `torchrun` (DDP):

```bash
cd training
torchrun --nproc_per_node=<NUM_GPUS> launch.py \
    --config default
```

**Key configuration flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `drop` | `True` | Enable frame dropping (core asymmetry mechanism) |
| `drop_max` | `12` | Max frames to drop from the teacher's sequence |
| `optim.ema.enabled` | `True` | Enable EMA teacher |
| `optim.ema.decay` | `0.995` | EMA decay rate |
| `optim.frozen_module_names` | `["*camera_head*"]` | Modules to freeze during training |
| `optim.optimizer.lr` | `1e-5` | Learning rate |
| `max_epochs` | `20` | Total training epochs |
| `aug` | `False` | Apply color jitter augmentation to student |
| `attn_drop_enabled` | `False` | Use attention-guided frame selection (random by default) |
| `rep_enabled` | `False` | Add representation-level distillation loss |
| `accum_steps` | `1` | Gradient accumulation steps (increase if OOM) |

 See `training/config/default.yaml` for the full configuration.

## Configuration Reference

The primary config is at `training/config/default.yaml`. It uses [Hydra](https://hydra.cc/) for configuration management. Key config groups:

- **`data`** — train/val dataset paths and dataloader settings
- **`model`** — VGGT model with enabled tasks (camera, depth, point, track)
- **`optim`** — EMA, optimizer (AdamW), LR scheduler (cosine with warmup), frozen modules, AMP (bfloat16)
- **`loss`** — Per-task loss weights and types (camera: L1 × 5.0; depth: gradient loss × 1.0)
- **`checkpoint`** — Save directory, save frequency, resume path
- **`logging`** — TensorBoard logger, log directory, metric keys
- **`distributed`** — DDP backend (NCCL), timeout, bucket size

## Evaluation

Evaluation scripts for video depth and camera estimation will be released soon.


## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{selfevo2026,
  title     = {Self-Improving 4D Perception via Self-Distillation},
  year      = {2026},
}
```

We build on [VGGT](https://github.com/facebookresearch/vggt); please also cite:

```bibtex
@inproceedings{wang2025vggt,
  title     = {VGGT: Visual Geometry Grounded Transformer},
  author    = {Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle = {CVPR},
  year      = {2025},
}
```

## License and Acknowledgments

This project is released under the VGGT License — see [LICENSE.txt](LICENSE.txt) for details.

We thank the authors of [VGGT](https://github.com/facebookresearch/vggt) for their excellent codebase. We also thank the maintainers of [OmniWorld](https://arxiv.org/abs/2509.12201), [BEDLAM2.0](https://bedlam.is.tue.mpg.de/), and [DROID](https://droid-dataset.github.io/) for providing the datasets used in this work.
