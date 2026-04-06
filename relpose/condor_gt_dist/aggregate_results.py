"""
Aggregate partial GT distribution results from all 24 condor workers.

Loads all *_partial.npy files, concatenates norms and angles, prints
summary statistics (same format as analyze_gt_distribution.py), and
generates a combined histogram PNG.

Usage:
    cd eval
    python relpose/condor_gt_dist/aggregate_results.py
    python relpose/condor_gt_dist/aggregate_results.py --results-dir relpose/condor_gt_dist/results
"""

import os
import argparse
import numpy as np
from pathlib import Path


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
        description="Aggregate condor GT distribution partial results"
    )
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--results-dir",
        default=os.path.join(_this_dir, "results"),
        help="Directory containing *_partial.npy files",
    )
    parser.add_argument(
        "--output-dir",
        default=_this_dir,
        help="Directory for output histogram (default: condor_gt_dist/)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load all partial results ──────────────────────────────────────
    partial_files = sorted(results_dir.glob("*_partial.npy"))
    if not partial_files:
        print(f"No *_partial.npy files found in {results_dir}")
        return

    print(f"Found {len(partial_files)} partial result files\n")

    all_norms = []
    all_angles = []
    all_seq_stats = []

    for pf in partial_files:
        data = np.load(pf, allow_pickle=True).item()
        norms = data["norms"]
        angles = data["angles"]
        seq_stats = data["seq_stats"]

        session_name = pf.stem.replace("_partial", "")
        n_seqs = len(seq_stats)
        n_pairs = len(norms)
        print(f"  {session_name}: {n_seqs} sequences, {n_pairs} pairs")

        if len(norms) > 0:
            all_norms.append(norms)
            all_angles.append(angles)
        all_seq_stats.extend(seq_stats)

    if not all_norms:
        print("\nNo pairwise data found across all sessions.")
        return

    all_norms = np.concatenate(all_norms)
    all_angles = np.concatenate(all_angles)

    print(f"\nTotal pairwise measurements collected: {len(all_norms)}")
    print(f"From {len(all_seq_stats)} sequences\n")

    # ── Summary statistics ────────────────────────────────────────────
    percentiles = [5, 25, 50, 75, 95]
    pcts_t = print_stats(
        "GT Pairwise Relative Translation ||t_rel||", "meters",
        all_norms, percentiles,
    )
    print()
    pcts_r = print_stats(
        "GT Pairwise Relative Rotation Angle", "degrees",
        all_angles, percentiles,
    )

    # ── Histogram ─────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        subtitle = (
            f"({len(all_seq_stats)} seqs, {len(all_norms)} pairs, "
            f"split=train, sparse=yes)"
        )
        colors = ["red", "orange", "green", "orange", "red"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # Translation histogram
        ax1.hist(all_norms, bins=100, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax1.set_xlabel("||t_rel|| (meters)")
        ax1.set_ylabel("Count")
        ax1.set_title(f"GT Pairwise Relative Translation Distribution\n{subtitle}")
        for p, v, c in zip(percentiles, pcts_t, colors):
            ax1.axvline(v, color=c, linestyle="--", linewidth=1, label=f"P{p}={v:.3f}")
        ax1.legend(fontsize=8)

        # Rotation histogram
        ax2.hist(all_angles, bins=100, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax2.set_xlabel("Relative rotation (degrees)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"GT Pairwise Relative Rotation Distribution\n{subtitle}")
        for p, v, c in zip(percentiles, pcts_r, colors):
            ax2.axvline(v, color=c, linestyle="--", linewidth=1, label=f"P{p}={v:.3f}")
        ax2.legend(fontsize=8)

        plt.tight_layout()
        hist_path = os.path.join(output_dir, "gt_distribution_train.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"\nHistogram saved to: {hist_path}")
    except ImportError:
        print("\n[WARN] matplotlib not available - skipping histogram.")


if __name__ == "__main__":
    main()
