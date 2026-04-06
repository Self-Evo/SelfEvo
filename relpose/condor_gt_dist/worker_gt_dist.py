"""
Per-session worker for GT distribution analysis on BEDLAM2 train split.

Processes a single session: scans all seq_* directories, excludes test
sequences, applies sparse sampling, computes pairwise translation norms
and rotation angles, then saves partial results as .npy.

Usage (condor or manual):
    cd eval
    python relpose/condor_gt_dist/worker_gt_dist.py --session 20240806_1_250_ai1101_vcam
"""

import os
import sys
import json
import argparse
import re
import numpy as np
import torch

from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_relpose_dir = os.path.dirname(_this_dir)
_eval_dir = os.path.dirname(_relpose_dir)
_pi3_root = os.path.join(os.path.dirname(_eval_dir), "uncharted4d-Pi3")
if os.path.isdir(_pi3_root):
    sys.path.insert(0, _pi3_root)
if _relpose_dir not in sys.path:
    sys.path.insert(0, _relpose_dir)

from bedlam2 import parse_camera_params
from metric import build_pair_index
from geometry import closed_form_inverse_se3
from pi3.utils.geometry import se3_inverse

# ── natural sort helper ───────────────────────────────────────────────
_num_pat = re.compile(r"(\d+)")


def _natural_key(path: str):
    s = os.path.basename(path)
    parts = _num_pat.split(s)
    return [int(p) if p.isdigit() else p for p in parts]


# ── sparse sampling (mirrors analyze_gt_distribution.py) ─────────────
def maybe_sparse_sample_ids(ids, *, divisor=10, min_frames=2):
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


def collect_frames(png_dir, seq_id):
    """Collect valid (filename, frame_id) pairs from a sequence PNG dir."""
    frames = []
    for f in sorted(Path(png_dir).glob("*.png"), key=lambda p: _natural_key(str(p))):
        parts = f.stem.split("_")
        if len(parts) >= 3:
            try:
                frame_id = int(parts[-1])
                frames.append((f.name, frame_id))
            except ValueError:
                continue
    return frames


def load_w2c_for_ids(meta_dir, seq_id, frames, ids, img_width=1280, img_height=720):
    """Load W2C extrinsics for selected frame indices."""
    c2w_list = []
    for i in ids:
        _, fid = frames[i]
        meta_path = os.path.join(meta_dir, f"{seq_id}_{fid:04d}_meta.json")
        _, c2w = parse_camera_params(meta_path, img_width, img_height)
        c2w_list.append(c2w)
    c2w = torch.from_numpy(np.stack(c2w_list, axis=0)).float()
    w2c = se3_inverse(c2w)
    return w2c


def collect_pairs(gt_w2c, N):
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


def main():
    parser = argparse.ArgumentParser(
        description="Per-session GT distribution worker"
    )
    parser.add_argument(
        "--session", required=True,
        help="Session name (e.g. 20240806_1_250_ai1101_vcam)",
    )
    parser.add_argument(
        "--bedlam2-dir",
        default="/is/cluster/fast/groups/ps-pbrgaussians/Uncharted4d/data/Bedlamv2",
    )
    parser.add_argument(
        "--test-selection",
        default=os.path.join(_eval_dir, "bedlam2_test_selection_20seq_seed42.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_eval_dir, "relpose", "condor_gt_dist", "results"),
    )
    args = parser.parse_args()

    session = args.session
    root = Path(args.bedlam2_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load test selection to exclude test sequences ─────────────────
    with open(args.test_selection, "r") as f:
        test_selection = json.load(f)
    test_seq_ids = set(test_selection.get(session, []))
    print(f"[{session}] Excluding {len(test_seq_ids)} test sequences")

    # ── Scan session directory directly (bypass full BEDLAM2Dataset) ──
    png_base = root / session / "png" / session / "png"
    meta_base = root / session / "groundtruth" / session / "ground_truth" / "meta_exr_depth"

    if not png_base.exists():
        print(f"[{session}] PNG base not found: {png_base}")
        sys.exit(1)
    if not meta_base.exists():
        print(f"[{session}] Meta base not found: {meta_base}")
        sys.exit(1)

    # Discover all seq_* directories
    seq_dirs = sorted(
        [d for d in png_base.iterdir() if d.is_dir() and d.name.startswith("seq_")],
        key=lambda d: d.name,
    )
    print(f"[{session}] Found {len(seq_dirs)} total sequences")

    # Filter to train only
    train_dirs = [d for d in seq_dirs if d.name not in test_seq_ids]
    print(f"[{session}] Training sequences: {len(train_dirs)}")

    # ── Process each training sequence ────────────────────────────────
    all_norms = []
    all_angles = []
    seq_stats = []
    skipped = 0

    for seq_dir in train_dirs:
        seq_id = seq_dir.name
        meta_dir = str(meta_base / seq_id)

        if not os.path.isdir(meta_dir):
            skipped += 1
            continue

        frames = collect_frames(str(seq_dir), seq_id)
        if len(frames) < 2:
            skipped += 1
            continue

        # Validate metadata exists for frames
        valid_frames = []
        for fname, fid in frames:
            meta_path = os.path.join(meta_dir, f"{seq_id}_{fid:04d}_meta.json")
            if os.path.exists(meta_path):
                valid_frames.append((fname, fid))
        if len(valid_frames) < 2:
            skipped += 1
            continue

        # Sparse sampling
        ids = np.arange(len(valid_frames), dtype=int)
        ids = maybe_sparse_sample_ids(ids, divisor=10, min_frames=2)
        if len(ids) < 2:
            skipped += 1
            continue

        # Load extrinsics and compute pairs
        try:
            gt_w2c = load_w2c_for_ids(meta_dir, seq_id, valid_frames, ids).double()
            N = len(ids)
            norms, angles_deg = collect_pairs(gt_w2c, N)
        except Exception as e:
            print(f"[{session}] Error processing {seq_id}: {e}")
            skipped += 1
            continue

        seq_key = f"{session}/{seq_id}"
        all_norms.append(norms)
        all_angles.append(angles_deg)
        seq_stats.append((seq_key, N, len(norms)))

    if skipped:
        print(f"[{session}] Skipped {skipped} sequences")

    # ── Save partial results ──────────────────────────────────────────
    if all_norms:
        all_norms = np.concatenate(all_norms)
        all_angles = np.concatenate(all_angles)
    else:
        all_norms = np.array([], dtype=np.float64)
        all_angles = np.array([], dtype=np.float64)

    out_path = os.path.join(args.output_dir, f"{session}_partial.npy")
    np.save(out_path, {
        "norms": all_norms,
        "angles": all_angles,
        "seq_stats": seq_stats,
    })

    print(f"[{session}] Done: {len(seq_stats)} sequences, "
          f"{len(all_norms)} pairs saved to {out_path}")


if __name__ == "__main__":
    main()
