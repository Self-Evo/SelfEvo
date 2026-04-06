"""
Analyze the distribution of ground truth pairwise relative camera rotation
angles (degrees) across all BEDLAM2 evaluation sequences.

Usage:
    cd eval
    python relpose/analyze_gt_rotation.py [--save-csv] [--output-dir DIR]
"""

import os
import sys
import argparse
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

# ── path setup (mirrors eval_angle.py) ────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_eval_dir = os.path.dirname(_this_dir)
_pi3_root = os.path.join(os.path.dirname(_eval_dir), "uncharted4d-Pi3")
if os.path.isdir(_pi3_root):
    sys.path.insert(0, _pi3_root)

# allow importing sibling modules (bedlam2, metric, geometry)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from bedlam2 import BEDLAM2Dataset
from metric import build_pair_index
from geometry import closed_form_inverse_se3


# ── same sparse sampling as eval_angle.py ─────────────────────────────
def maybe_sparse_sample_ids(
    ids: np.ndarray,
    *,
    sparse: bool,
    divisor: int = 10,
    min_frames: int = 2,
) -> np.ndarray:
    if not sparse:
        return ids
    ids = np.asarray(ids, dtype=int)
    num = len(ids)
    if num < min_frames:
        return ids
    target = max(min_frames, num // max(1, int(divisor)))
    target = min(target, num)
    picked = np.linspace(0, num - 1, target, dtype=int)
    out = ids[picked]
    out = np.unique(out)
    if len(out) < min_frames and num >= min_frames:
        out = np.unique(np.array([ids[0], ids[-1]], dtype=int))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GT pairwise relative rotation distribution"
    )
    parser.add_argument(
        "--bedlam2-dir",
        default="/is/cluster/fast/groups/ps-pbrgaussians/Uncharted4d/data/Bedlamv2",
        help="Root directory of BEDLAM2 dataset",
    )
    parser.add_argument(
        "--test-selection",
        default=os.path.join(_eval_dir, "bedlam2_test_selection_20seq_seed42.json"),
        help="Test selection JSON file",
    )
    parser.add_argument(
        "--sparse", action="store_true", default=True,
        help="Apply 1/10 sparse sampling (default: True)",
    )
    parser.add_argument(
        "--no-sparse", action="store_true", default=False,
        help="Disable sparse sampling",
    )
    parser.add_argument(
        "--save-csv", action="store_true", default=False,
        help="Save per-pair rotation angles to CSV",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for output files (default: cwd)",
    )
    args = parser.parse_args()

    sparse = args.sparse and (not args.no_sparse)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load dataset ───────────────────────────────────────────────
    dataset = BEDLAM2Dataset(
        BEDLAM2_DIR=args.bedlam2_dir,
        test_selection_file=args.test_selection,
    )
    print(f"\nTotal sequences: {len(dataset.sequence_list)}")

    # ── 2. Iterate over sequences and collect pairwise rotation angles ─
    all_angles = []      # flat list of all rotation angles (degrees)
    per_seq_stats = []   # (seq_name, num_frames, num_pairs, mean, median)
    skipped = 0

    for seq_name in tqdm(dataset.sequence_list, desc="Processing sequences"):
        num_frames = dataset.get_seq_framenum(sequence_name=seq_name)
        ids = np.arange(num_frames, dtype=int)

        if sparse:
            ids = maybe_sparse_sample_ids(ids, sparse=True, divisor=10, min_frames=2)

        if len(ids) < 2:
            skipped += 1
            continue

        # Load GT extrinsics (W2C)
        batch = dataset.get_data(sequence_name=seq_name, ids=ids)
        gt_w2c = batch["extrs"].double()  # (L, 4, 4)

        N = len(ids)
        pair_i1, pair_i2 = build_pair_index(N, B=1)

        # relative_pose = W2C_j @ inv(W2C_i)
        relative_pose = gt_w2c[pair_i2].bmm(
            closed_form_inverse_se3(gt_w2c[pair_i1])
        )

        # Rotation angle from trace: angle = arccos((trace(R) - 1) / 2)
        R_rel = relative_pose[:, :3, :3]
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        angles_rad = torch.acos(((trace - 1) / 2).clamp(-1, 1))
        angles_deg = (angles_rad * 180 / np.pi).numpy()

        all_angles.append(angles_deg)
        per_seq_stats.append((seq_name, N, len(angles_deg), angles_deg.mean(), np.median(angles_deg)))

    if skipped:
        print(f"Skipped {skipped} sequences (< 2 frames after sampling)")

    all_angles = np.concatenate(all_angles)
    print(f"\nTotal pairwise rotations collected: {len(all_angles)}")
    print(f"From {len(per_seq_stats)} sequences\n")

    # ── 3. Summary statistics ─────────────────────────────────────────
    percentiles = [5, 25, 50, 75, 95]
    pcts = np.percentile(all_angles, percentiles)

    print("=" * 55)
    print("  GT Pairwise Relative Rotation Angle (degrees)")
    print("=" * 55)
    print(f"  Count  : {len(all_angles)}")
    print(f"  Min    : {all_angles.min():.6f}")
    print(f"  Max    : {all_angles.max():.6f}")
    print(f"  Mean   : {all_angles.mean():.6f}")
    print(f"  Median : {np.median(all_angles):.6f}")
    print(f"  Std    : {all_angles.std():.6f}")
    for p, v in zip(percentiles, pcts):
        print(f"  P{p:<3d}   : {v:.6f}")
    print("=" * 55)

    # ── 4. Histogram ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_angles, bins=100, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax.set_xlabel("Relative rotation (degrees)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"GT Pairwise Relative Rotation Distribution\n"
            f"({len(per_seq_stats)} seqs, {len(all_angles)} pairs, "
            f"sparse={'yes' if sparse else 'no'})"
        )
        # mark percentiles
        colors = ["red", "orange", "green", "orange", "red"]
        for p, v, c in zip(percentiles, pcts, colors):
            ax.axvline(v, color=c, linestyle="--", linewidth=1, label=f"P{p}={v:.3f}")
        ax.legend(fontsize=8)
        plt.tight_layout()

        hist_path = os.path.join(output_dir, "gt_rotation_distribution.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"\nHistogram saved to: {hist_path}")
    except ImportError:
        print("\n[WARN] matplotlib not available – skipping histogram.")

    # ── 5. Optional CSV ───────────────────────────────────────────────
    if args.save_csv:
        csv_path = os.path.join(output_dir, "gt_rotation_stats.csv")
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seq_name", "pair_i", "pair_j", "rotation_angle_deg"])
            offset = 0
            for seq_name, N, num_pairs, _, _ in per_seq_stats:
                pair_i1, pair_i2 = build_pair_index(N, B=1)
                for k in range(num_pairs):
                    writer.writerow([
                        seq_name,
                        pair_i1[k].item(),
                        pair_i2[k].item(),
                        f"{all_angles[offset + k]:.8f}",
                    ])
                offset += num_pairs
        print(f"Per-pair CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
