"""
Analyze the distribution of ground truth pairwise relative camera translation
magnitudes (||t_rel||, meters) and rotation angles (degrees) across all BEDLAM2
evaluation sequences.

Usage:
    cd eval
    python relpose/analyze_gt_distribution.py [--save-csv] [--output-dir DIR]
    python relpose/analyze_gt_distribution.py --fixed-stride 40 --num-test-per-seq 5
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


def generate_strided_subsequences(num_frames, stride, num_subseqs, min_len=3):
    max_offsets = min(num_subseqs, stride)
    offsets = np.round(np.linspace(0, stride - 1, max_offsets)).astype(int)
    offsets = np.unique(offsets)
    subseqs = []
    for o in offsets:
        ids = np.arange(o, num_frames, stride)
        if len(ids) >= min_len:
            subseqs.append(ids)
    return subseqs


def print_stats(name, unit, data, percentiles):
    pcts = np.percentile(data, percentiles)
    print("=" * 55)
    print(f"  {name} ({unit})")
    print("=" * 55)
    print(f"  Count  : {len(data)}")
    print(f"  Min    : {data.min():.6f}")
    print(f"  Max    : {data.max():.6f}")
    print(f"  Mean   : {data.mean():.6f}")
    print(f"  Median : {np.median(data):.6f}")
    print(f"  Std    : {data.std():.6f}")
    for p, v in zip(percentiles, pcts):
        print(f"  P{p:<3d}   : {v:.6f}")
    print("=" * 55)
    return pcts


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GT pairwise relative translation & rotation distribution"
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
        "--fixed-stride", type=int, default=0,
        help="Use strided subsequence sampling with this stride (0=disabled, use sparse instead)",
    )
    parser.add_argument(
        "--num-test-per-seq", type=int, default=5,
        help="Number of strided subsequences per sequence (only used with --fixed-stride)",
    )
    parser.add_argument(
        "--split", choices=["test", "train"], default="test",
        help="Which split to analyze: 'test' (default) or 'train' (all minus test)",
    )
    parser.add_argument(
        "--exclude-session-suffix",
        action="append",
        default=[],
        help="Exclude sessions whose name ends with this suffix (repeatable, e.g. --exclude-session-suffix _vcam --exclude-session-suffix _hdri)",
    )
    parser.add_argument(
        "--save-csv", action="store_true", default=False,
        help="Save per-pair stats to CSV",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for output files (default: cwd)",
    )
    args = parser.parse_args()

    sparse = args.sparse and (not args.no_sparse)
    use_strided = args.fixed_stride > 0
    if use_strided:
        sparse = False  # strided mode overrides sparse
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load dataset ───────────────────────────────────────────────
    if args.split == "train":
        # Load ALL sequences, then exclude the test ones
        dataset_all = BEDLAM2Dataset(
            BEDLAM2_DIR=args.bedlam2_dir,
            test_selection_file=None,  # scan everything
        )
        dataset_test = BEDLAM2Dataset(
            BEDLAM2_DIR=args.bedlam2_dir,
            test_selection_file=args.test_selection,
        )
        test_keys = set(dataset_test.sequence_list)
        train_keys = [k for k in dataset_all.sequence_list if k not in test_keys]
        # Patch dataset_all to only iterate train sequences
        dataset_all.sequence_list = train_keys
        dataset = dataset_all
        print(f"\nTrain split: {len(train_keys)} sequences (excluded {len(test_keys)} test sequences)")
    else:
        dataset = BEDLAM2Dataset(
            BEDLAM2_DIR=args.bedlam2_dir,
            test_selection_file=args.test_selection,
        )
        print(f"\nTest split: {len(dataset.sequence_list)} sequences")

    if args.exclude_session_suffix:
        suffixes = tuple(args.exclude_session_suffix)
        before = len(dataset.sequence_list)
        dataset.sequence_list = [
            k for k in dataset.sequence_list
            if not k.split("/")[0].endswith(suffixes)
        ]
        print(f"Excluded {before - len(dataset.sequence_list)} sequences "
              f"(session suffix filter: {suffixes}), "
              f"{len(dataset.sequence_list)} remaining")

    # ── 2. Iterate over sequences and collect both metrics ────────────
    all_norms = []       # ||t_rel|| in meters
    all_angles = []      # rotation angle in degrees
    per_seq_stats = []   # (seq_name, N, num_pairs)
    skipped = 0

    def _collect_pairs(gt_w2c, N):
        """Compute pairwise translation norms and rotation angles."""
        pair_i1, pair_i2 = build_pair_index(N, B=1)
        relative_pose = gt_w2c[pair_i2].bmm(
            closed_form_inverse_se3(gt_w2c[pair_i1])
        )
        t_rel = relative_pose[:, :3, 3]
        norms = torch.norm(t_rel, dim=1).numpy()
        R_rel = relative_pose[:, :3, :3]
        trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        angles_rad = torch.acos(((trace - 1) / 2).clamp(-1, 1))
        angles_deg = (angles_rad * 180 / np.pi).numpy()
        return norms, angles_deg

    for seq_name in tqdm(dataset.sequence_list, desc="Processing sequences"):
        num_frames = dataset.get_seq_framenum(sequence_name=seq_name)

        if use_strided:
            subseq_list = generate_strided_subsequences(
                num_frames, args.fixed_stride, args.num_test_per_seq, min_len=3,
            )
            if not subseq_list:
                skipped += 1
                continue
            seq_norms, seq_angles = [], []
            for ids in subseq_list:
                batch = dataset.get_data(sequence_name=seq_name, ids=ids)
                if batch is None:
                    continue
                gt_w2c = batch["extrs"].double()
                n, a = _collect_pairs(gt_w2c, len(batch["ind"]))
                seq_norms.append(n)
                seq_angles.append(a)
            if not seq_norms:
                skipped += 1
                continue
            norms = np.concatenate(seq_norms)
            angles_deg = np.concatenate(seq_angles)
            N = sum(len(ids) for ids in subseq_list)
        else:
            ids = np.arange(num_frames, dtype=int)
            if sparse:
                ids = maybe_sparse_sample_ids(ids, sparse=True, divisor=10, min_frames=2)
            if len(ids) < 2:
                skipped += 1
                continue
            batch = dataset.get_data(sequence_name=seq_name, ids=ids)
            if batch is None:
                skipped += 1
                continue
            gt_w2c = batch["extrs"].double()
            N = len(batch["ind"])
            norms, angles_deg = _collect_pairs(gt_w2c, N)

        all_norms.append(norms)
        all_angles.append(angles_deg)
        per_seq_stats.append((seq_name, N, len(norms)))

    if skipped:
        print(f"Skipped {skipped} sequences (< 2 frames after sampling)")

    all_norms = np.concatenate(all_norms)
    all_angles = np.concatenate(all_angles)
    print(f"\nTotal pairwise measurements collected: {len(all_norms)}")
    print(f"From {len(per_seq_stats)} sequences\n")

    # ── 3. Summary statistics ─────────────────────────────────────────
    percentiles = [5, 25, 50, 75, 95]
    pcts_t = print_stats("GT Pairwise Relative Translation ||t_rel||", "meters", all_norms, percentiles)
    print()
    pcts_r = print_stats("GT Pairwise Relative Rotation Angle", "degrees", all_angles, percentiles)

    # ── 4. Histograms ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if use_strided:
            sampling_str = f"stride={args.fixed_stride}, subseqs={args.num_test_per_seq}"
        else:
            sampling_str = f"sparse={'yes' if sparse else 'no'}"
        subtitle = (
            f"({len(per_seq_stats)} seqs, {len(all_norms)} pairs, "
            f"split={args.split}, {sampling_str})"
        )
        colors = ["red", "orange", "green", "orange", "red"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # ── Translation histogram ──
        ax1.hist(all_norms, bins=100, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax1.set_xlabel("||t_rel|| (meters)")
        ax1.set_ylabel("Count")
        ax1.set_title(f"GT Pairwise Relative Translation Distribution\n{subtitle}")
        for p, v, c in zip(percentiles, pcts_t, colors):
            ax1.axvline(v, color=c, linestyle="--", linewidth=1, label=f"P{p}={v:.3f}")
        ax1.legend(fontsize=8)

        # ── Rotation histogram ──
        ax2.hist(all_angles, bins=100, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax2.set_xlabel("Relative rotation (degrees)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"GT Pairwise Relative Rotation Distribution\n{subtitle}")
        for p, v, c in zip(percentiles, pcts_r, colors):
            ax2.axvline(v, color=c, linestyle="--", linewidth=1, label=f"P{p}={v:.3f}")
        ax2.legend(fontsize=8)

        plt.tight_layout()

        suffix = f"_stride{args.fixed_stride}_x{args.num_test_per_seq}" if use_strided else ""
        split_tag = f"_{args.split}" if args.split != "test" else ""
        hist_path = os.path.join(output_dir, f"gt_distribution{split_tag}{suffix}.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"\nHistogram saved to: {hist_path}")
    except ImportError:
        print("\n[WARN] matplotlib not available – skipping histogram.")

    # ── 5. Optional CSV ───────────────────────────────────────────────
    if args.save_csv:
        csv_path = os.path.join(output_dir, "gt_distribution_stats.csv")
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seq_name", "pair_i", "pair_j", "t_rel_norm", "rotation_angle_deg"])
            offset = 0
            for seq_name, N, num_pairs in per_seq_stats:
                pair_i1, pair_i2 = build_pair_index(N, B=1)
                for k in range(num_pairs):
                    writer.writerow([
                        seq_name,
                        pair_i1[k].item(),
                        pair_i2[k].item(),
                        f"{all_norms[offset + k]:.8f}",
                        f"{all_angles[offset + k]:.8f}",
                    ])
                offset += num_pairs
        print(f"Per-pair CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
