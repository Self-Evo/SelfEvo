# Relative Camera Pose Estimation

## 1. Angular Metrics

See configs in `configs/evaluation/relpose-angular.yaml`.

```bash
# python relpose/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python relpose/eval_angle.py
```

### Eval World-to-Camera Camera Poses

`eval-angle.py` will generate files like:

```
recons-eval
├── ...
├── outputs
|   └── relpose-angular
|       ├── hydra (runtime configs)
|       ├── CO3Dv2-metric.csv
|       └── Re10K-metric.csv
└── ...
```

### Use Your Own Sampling Strategy

seq-id-map records the mapping from sequence name to image ids of this sequence. Follow [VGGT](https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/test_co3d.py#L281), We randomly choose 10 images in each sequence as model input. To ensure reproducible evaluation results, we precompute seq-id-maps with given random seed.

You can edit [configs/data/relpose-angular.yaml](configs/data/relpose-angular.yaml) and [relpose/sampling.py](relpose/sampling.py) to generate new seq-id-maps under [datasets/seq-id-maps](datasets/seq-id-maps). We have provided all necessary seq-id-maps for our evaluation settings.

If you want to use your own sampling strategy, you need to generate seq-id-maps before running `eval-angle.py`, and specify it in `configs/data/relpose-angular.yaml`.

## 2. Distance Metrics

```bash
python relpose/eval_dist.py
```

Assume `eval_models` only have `pi3`, `eval-dist.py` will generate folders like:

```
recons-eval
├── ...
├── outputs
|   └── relpose-distance
|       ├── hydra (runtime configs)
|       ├── scannetv2
|       |   ├── sequence_1
|       |   |   ├── eval_metric.txt
|       |   |   ├── pred_intrinsics.json
|       |   |   ├── pred_poses.npy
|       |   |   ├── pred_traj.txt
|       |   |   └── vis_traj_error.png
|       |   ├── sequence_2
|       |   ├── ...
|       |   └── _seq_metrics.csv
|       ├── sintel
|       └── tum
|       ├── scannetv2-metric.csv
|       ├── sintel-metric.csv
|       └── tum-metric.csv
└── ...
```