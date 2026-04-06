# Eval

Evaluation framework for **VGGT** and **Pi3** vision models across multiple computer vision benchmarks. Supports relative pose estimation, depth estimation, and multi-view reconstruction tasks.

## Setup

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU

### Install Dependencies

```bash
pip install -r requirements.txt
```

### VGGT Symbolic Link

The project requires a symbolic link to the VGGT package. Create it from the repo root:

```bash
ln -s your-root/vggt/vggt vggt
```

This links `eval/vggt/` to the VGGT model source, enabling direct imports like:

```python
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
```

### Environment

```bash
conda activate selfevo
export PYTHONPATH=you-root/vggt:$PYTHONPATH
cd you-root/eval
```

## Usage

All evaluation scripts use [Hydra](https://hydra.cc/) for configuration. Override any parameter from the command line.

### Relative Pose Estimation

**Angular metrics** (rotation/translation angle error, AUC):

```bash
python relpose/eval_angle.py \
  eval_datasets="[Re10K, CO3Dv2, OmniGeo, OmniVideo]" \
  name="my_experiment"
```

### Depth Estimation

**Inference** (generate depth predictions):

```bash
python videodepth/infer.py \
  name="my_experiment" \
  model=vggt \
  pi3.pretrained_model_name_or_path="/path/to/model.pt" \
  eval_datasets="[sintel, bonn, kitti]"
```

**Evaluation** (compute metrics against ground truth):

```bash
python videodepth/eval.py \
  name="my_experiment" \
  model=vggt \
  pi3.pretrained_model_name_or_path="/path/to/model.pt" \
  eval_datasets="[sintel, bonn, kitti]" \
  align="scale&shift"
```

Alignment options: `scale`, `scale&shift`, `metric`

### SLURM Batch Submission

```bash
bash scripts_extended_eval/submit.sh
```

Submits jobs for depth and camera evaluation on pretrained and baseline models.


## Visualization

Generate comparison plots from evaluation results:

```bash
python vis.py        # Depth evaluation plots
python vis_cam.py    # Camera evaluation plots
```

## Outputs

Results are written to `outputs/<experiment_name>/` with:
- Per-dataset CSV metric summaries
- Hydra run configs (for reproducibility)
- Trajectory files (for distance-based pose evaluation)
