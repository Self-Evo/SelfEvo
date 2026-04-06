# Drop-in replacement.
# - Keeps your original API:
#     get_all_sequences(dataset_cfg)
#     list_imgs_a_sequence(dataset_cfg, seq)
#     list_depths_a_sequence(dataset_cfg, seq)
# - Adds OmniWorld-DROID support (multi-subset + only ext/fixed third-view cameras).
#
# Notes:
# - DROID sequences returned as "<scene_rel>/<cam_id>"
#   where scene_rel is relative to anno_root (so it includes subset prefix automatically).
# - Images are located via metadata_csv mapping from "Annotation Path" to "Video Path":
#     <frames_root>/<video_rel>/recordings/PNG/<cam_id>/*.png
# - GT depths:
#     <anno_root>/<scene_rel>/foundation_stereo/<cam_id>/*.png

import os
import os.path as osp
import glob
import re
import csv
import json
from functools import lru_cache
from typing import Optional, List, Tuple, Dict, Any

from omegaconf import DictConfig, ListConfig
import random
import csv


# -------------------------
# Helpers
# -------------------------
def _load_seq_cache(cache_path: str) -> Optional[dict]:
    if (not cache_path) or (not osp.isfile(cache_path)):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_seq_cache(cache_path: str, payload: dict):
    if not cache_path:
        return
    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, cache_path)

def _natural_frame_key(p: str):
    """Sort frames naturally by numeric filename (0.png, 1.png, 10.png)."""
    base = osp.splitext(osp.basename(p))[0]
    m = re.search(r"\d+", base)
    return int(m.group()) if m else base


def _droid_seq_to_scene_and_cam(seq: str) -> Tuple[str, str]:
    scene_rel, cam_id = seq.rsplit("/", 1)
    return scene_rel, cam_id

def droid_wrist_cam_ids(scene_dir: str) -> Optional[List[str]]:
    """
    Returns wrist-view camera serial ids.
    Reads <scene_dir>/meta_info.json.

    Keeps keys like:
      wrist_cam_serial
      wrist_left_cam_serial / wrist_right_cam_serial (if exists)
      wrist_*_cam_serial

    Returns list (unique, stable order by key name).
    """
    meta_path = osp.join(scene_dir, "meta_info.json")
    if not osp.isfile(meta_path):
        return None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return None

    items: List[Tuple[str, str]] = []
    for k, v in meta.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, (str, int)):
            continue

        kk = k.strip().lower()
        if not kk.endswith("_cam_serial"):
            continue

        # accept wrist_cam_serial / wrist_*_cam_serial
        if kk == "wrist_cam_serial" or kk.startswith("wrist_"):
            items.append((kk, str(v)))

    if not items:
        return None

    # stable
    items.sort(key=lambda x: x[0])

    out: List[str] = []
    for _, s in items:
        if s not in out:
            out.append(s)
    return out

@lru_cache(maxsize=1)
def _load_droid_meta_map(meta_csv: str) -> Dict[str, str]:
    """
    Read omniworld_droid_metadata.csv and build:
      annotation_path -> video_path
    Robust to header variants and case.
    """
    mp: Dict[str, str] = {}
    if (not meta_csv) or (not osp.isfile(meta_csv)):
        return mp

    with open(meta_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_l = {(k or "").strip().lower(): v for k, v in row.items()}
            ap = row_l.get("annotation path") or row_l.get("annotation_path")
            vp = row_l.get("video path") or row_l.get("video_path")
            if ap and vp:
                mp[ap.rstrip("/")] = vp.rstrip("/")
    return mp


def _iter_droid_processed_roots(dataset_cfg: DictConfig) -> List[str]:
    """
    dataset_cfg.anno_root is the parent folder containing all subsets, e.g.:
      /.../OmniWorld-DROID_benchmark/depths

    Supports:
      subset_list: [omniworld_droid_5000_5049, ...]
      subset_glob: omniworld_droid_*
    """
    base = dataset_cfg.anno_root

    subset_roots: List[str] = []
    if getattr(dataset_cfg, "subset_list", None) is not None:
        subset_roots = [osp.join(base, s) for s in list(dataset_cfg.subset_list)]
    elif getattr(dataset_cfg, "subset_glob", None) is not None:
        subset_roots = sorted([p for p in glob.glob(osp.join(base, dataset_cfg.subset_glob)) if osp.isdir(p)])
    else:
        # default: try match all omniworld_droid_*
        subset_roots = sorted([p for p in glob.glob(osp.join(base, "omniworld_droid_*")) if osp.isdir(p)])
        if not subset_roots:
            subset_roots = [base]

    processed_roots: List[str] = []
    for r in subset_roots:
        cand = osp.join(r, "droid_processed")
        processed_roots.append(cand if osp.isdir(cand) else r)

    return processed_roots


def droid_fixed_cam_ids(scene_dir: str, keep_ext_indices: Optional[List[int]] = None) -> Optional[List[str]]:
    """
    Returns fixed-view camera serial ids (external cameras), e.g. ext1/ext2.
    Reads <scene_dir>/meta_info.json.

    - Keeps: ext1_cam_serial, ext2_cam_serial, ...
    - Drops: wrist_cam_serial
    - keep_ext_indices:
        None -> keep all ext*
        [1]  -> keep only ext1
        [2]  -> keep only ext2
        [1,2]-> keep ext1+ext2
    """
    meta_path = osp.join(scene_dir, "meta_info.json")
    if not osp.isfile(meta_path):
        return None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return None

    keep_set = set(keep_ext_indices) if keep_ext_indices is not None else None

    ext_items: List[Tuple[int, str]] = []
    for k, v in meta.items():
        if not isinstance(k, str):
            continue
        if not k.endswith("_cam_serial"):
            continue
        if not isinstance(v, (str, int)):
            continue

        kk = k.lower()
        if kk.startswith("wrist_"):
            continue

        m = re.match(r"ext(\d+)_cam_serial$", kk)
        if m:
            idx = int(m.group(1))
            if (keep_set is None) or (idx in keep_set):
                ext_items.append((idx, str(v)))

    ext_items.sort(key=lambda x: x[0])

    out: List[str] = []
    for _, s in ext_items:
        if s not in out:
            out.append(s)

    return out


# -------------------------
# Public APIs (used by your scripts)
# -------------------------

def get_all_sequences(dataset_cfg: DictConfig, sort_by_seq_name: bool = True):
    """
    Generic datasets:
      - ls_all_seqs is str -> list subfolders
      - ls_all_seqs is ListConfig -> use list directly

    DROID dataset (dataset_cfg.format == "droid"):
      - anno_root points to depths parent containing all subsets
      - enumerates <scene>/foundation_stereo/<cam_id> under all subsets
      - if only_fixed_views: keep only ext cameras from meta_info.json
    """
    # ---------- DROID ----------
    if getattr(dataset_cfg, "format", None) == "droid":
        processed_roots = _iter_droid_processed_roots(dataset_cfg)

        only_fixed = bool(getattr(dataset_cfg, "only_fixed_views", False))
        only_wrist = bool(getattr(dataset_cfg, "only_wrist_views", False))
        if only_fixed and only_wrist:
            raise ValueError("[DROID] only_fixed_views and only_wrist_views cannot both be True.")

        keep_ext = getattr(dataset_cfg, "ext_keep_indices", None)
        if keep_ext is not None and isinstance(keep_ext, ListConfig):
            keep_ext = [int(x) for x in list(keep_ext)]

        # ---- cache (store FULL seqs only) ----
        cache_path = getattr(dataset_cfg, "seq_cache_file", None)
        refresh = bool(getattr(dataset_cfg, "refresh_seq_cache", False))

        cache_key = {
            "anno_root": str(getattr(dataset_cfg, "anno_root", "")),
            "subset_glob": str(getattr(dataset_cfg, "subset_glob", "")),
            "subset_list": list(getattr(dataset_cfg, "subset_list", [])) if getattr(dataset_cfg, "subset_list", None) is not None else None,
            "processed_roots": [str(p) for p in processed_roots],
            "only_fixed_views": only_fixed,
            "ext_keep_indices": keep_ext,
            "only_wrist_views": only_wrist,
            # 注意：不包含 max_seqs / seed
        }

        full_seqs = None
        if cache_path and (not refresh):
            cached = _load_seq_cache(cache_path)
            if cached and cached.get("cache_key", None) == cache_key:
                fs = cached.get("full_seqs", None)
                if isinstance(fs, list) and len(fs) > 0:
                    full_seqs = fs

        # ---- compute FULL seqs if cache miss ----
        if full_seqs is None:
            seqs_tmp: List[str] = []

            for processed_root in processed_roots:
                for dirpath, dirnames, _ in os.walk(processed_root):
                    if osp.basename(dirpath) != "foundation_stereo":
                        continue

                    scene_dir = osp.dirname(dirpath)  # .../<scene>
                    scene_rel = osp.relpath(scene_dir, dataset_cfg.anno_root)  # includes subset prefix

                    cam_ids = sorted([d for d in dirnames if d and (not d.startswith("."))])

                    if only_wrist:
                        allowed = droid_wrist_cam_ids(scene_dir)
                        if not allowed:
                            continue
                        allowed_set = set(allowed)
                        cam_ids = [c for c in cam_ids if c in allowed_set]
                        if not cam_ids:
                            continue

                    elif only_fixed:
                        allowed = droid_fixed_cam_ids(scene_dir, keep_ext_indices=keep_ext)
                        if not allowed:
                            continue
                        allowed_set = set(allowed)
                        cam_ids = [c for c in cam_ids if c in allowed_set]
                        if not cam_ids:
                            continue

                    for cam_id in cam_ids:
                        seqs_tmp.append(f"{scene_rel}/{cam_id}")

            # full list should be stable for deterministic sampling
            full_seqs = sorted(set(seqs_tmp))

            if not full_seqs:
                raise RuntimeError(
                    f"[DROID] No sequences found under anno_root={dataset_cfg.anno_root} "
                    f"(only_fixed_views={only_fixed}). "
                    f"Check subset_glob/subset_list and meta_info.json."
                )

            if cache_path:
                _save_seq_cache(cache_path, {"cache_key": cache_key, "full_seqs": full_seqs})

        # ---- deterministic sampling on RETURN (does NOT affect cache) ----
        max_seqs = int(getattr(dataset_cfg, "max_seqs", 0) or 0)
        seed = int(getattr(dataset_cfg, "seq_sample_seed", 0) or 0)

        if max_seqs > 0 and len(full_seqs) > max_seqs:
            rng = random.Random(seed)  # local RNG, deterministic
            seqs = rng.sample(full_seqs, k=max_seqs)  # deterministic given full_seqs order + seed
            # keep output stable/readable
            if sort_by_seq_name:
                seqs = sorted(seqs)
            return seqs

        # no sampling
        return full_seqs

    # ---------- Generic ----------
    if isinstance(dataset_cfg.ls_all_seqs, str):
        p = dataset_cfg.ls_all_seqs

        # NEW: if ls_all_seqs is a CSV file, read sequences from it
        if osp.isfile(p) and p.lower().endswith(".csv"):
            seq_col = getattr(dataset_cfg, "seq_col", "Annotation Path")

            # optional filter: e.g. require_prior_depth: true -> only keep has_prior_depth==True rows
            require_prior = bool(getattr(dataset_cfg, "require_prior_depth", False))
            filter_col = getattr(dataset_cfg, "csv_filter_col", None)
            filter_val = getattr(dataset_cfg, "csv_filter_val", None)

            if require_prior:
                filter_col = "has_prior_depth"
                filter_val = True

            seq_list = []
            with open(p, "r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or seq_col not in reader.fieldnames:
                    raise ValueError(f"CSV {p} missing column '{seq_col}', have {reader.fieldnames}")

                for row in reader:
                    # filtering (optional)
                    if filter_col is not None:
                        rv = row.get(filter_col, None)
                        if rv is None:
                            continue
                        # normalize bool-ish strings
                        if isinstance(filter_val, bool):
                            rv_norm = str(rv).strip().lower()
                            ok = (rv_norm in ("1", "true", "yes", "y")) if filter_val else (rv_norm in ("0", "false", "no", "n"))
                        else:
                            ok = (str(rv).strip() == str(filter_val))
                        if not ok:
                            continue

                    s = (row.get(seq_col) or "").strip()
                    if s:
                        seq_list.append(s)

            # de-dup keep order
            seen = set()
            uniq = []
            for s in seq_list:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            seq_list = uniq

            keep_order = bool(getattr(dataset_cfg, "keep_csv_order", False))
            if sort_by_seq_name and (not keep_order):
                return sorted(seq_list)
            return seq_list

        # OLD: directory listing behavior
        seq_list = [d for d in os.listdir(p) if osp.isdir(osp.join(p, d))]

    elif isinstance(dataset_cfg.ls_all_seqs, ListConfig):
        seq_list = list(dataset_cfg.ls_all_seqs)
    else:
        raise ValueError(
            f"Unknown ls_all_seqs type: {type(dataset_cfg.ls_all_seqs)}, "
            f"ls_all_seqs is {dataset_cfg.ls_all_seqs}, which should be a string or a ListConfig"
        )

    return sorted(seq_list) if sort_by_seq_name else seq_list


def list_imgs_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None):
    """
    Generic:
      subdir = dataset_cfg.img.path.format(seq=seq)

    DROID:
      scene_rel is relative to anno_root (depths parent) and includes subset prefix.
      metadata csv keys ("Annotation Path") usually start from "droid_processed/..." (no subset prefix),
      so we strip subset prefix before lookup.
    """
    # ---------- DROID ----------
    if getattr(dataset_cfg, "format", None) == "droid":
        assert seq is not None, "DROID requires seq"
        scene_rel, cam_id = _droid_seq_to_scene_and_cam(seq)

        # strip subset prefix for metadata lookup
        scene_key = scene_rel.rstrip("/")
        idx = scene_key.find("droid_processed/")
        scene_key_for_meta = scene_key[idx:] if idx >= 0 else scene_key

        meta_map = _load_droid_meta_map(getattr(dataset_cfg, "metadata_csv", ""))
        video_rel = meta_map.get(scene_key_for_meta, None)
        if video_rel is None:
            # fallback assumes raw/processed share structure
            video_rel = scene_key_for_meta.replace("droid_processed", "droid_raw", 1)

        img_dir = osp.join(dataset_cfg.frames_root, video_rel, "recordings", "PNG", cam_id)
        if not osp.isdir(img_dir):
            raise FileNotFoundError(f"[DROID] image dir not found: {img_dir}")

        ext = getattr(dataset_cfg.img, "ext", "png")
        paths = glob.glob(osp.join(img_dir, f"*.{ext}"))
        return sorted(paths, key=_natural_frame_key)

    # ---------- Generic ----------
    subdir = dataset_cfg.img.path.format(seq=seq)  # string include {seq}
    ext = dataset_cfg.img.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"))
    return filelist


def list_depths_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None):
    """
    Generic:
      subdir = dataset_cfg.depth.path.format(seq=seq)

    DROID:
      <anno_root>/<scene_rel>/foundation_stereo/<cam_id>/*.png
    """
    # ---------- DROID ----------
    if getattr(dataset_cfg, "format", None) == "droid":
        assert seq is not None, "DROID requires seq"
        scene_rel, cam_id = _droid_seq_to_scene_and_cam(seq)

        depth_dir = osp.join(dataset_cfg.anno_root, scene_rel, "foundation_stereo", cam_id)
        if not osp.isdir(depth_dir):
            raise FileNotFoundError(f"[DROID] depth dir not found: {depth_dir}")

        ext = getattr(dataset_cfg.depth, "ext", "png")
        paths = glob.glob(osp.join(depth_dir, f"*.{ext}"))
        return sorted(paths, key=_natural_frame_key)

    # ---------- Generic ----------
    subdir = dataset_cfg.depth.path.format(seq=seq)  # string include {seq}
    ext = dataset_cfg.depth.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"))
    return filelist
