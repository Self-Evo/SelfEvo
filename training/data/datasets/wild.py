# data/datasets/wild_images.py
import os, os.path as osp, glob, random, logging, re
from typing import List, Tuple, Optional

import cv2
import numpy as np

from data.base_dataset import BaseDataset
import math
from pathlib import Path
from torch.utils.data import get_worker_info
import imageio
import csv
import json
from functools import lru_cache

IMG_EXTS = (".png", ".jpg")

_num_pat = re.compile(r"(\d+)")

def _strip_to_seq_base(rel_seq_dir: str) -> str | None:
    """
    rel_seq_dir: relative to ROOT_DIR, e.g.
    GuptaLab/failure/.../Thu_Apr_20_13:11:11_2023/recordings/PNG/16291792
    return:
    GuptaLab/failure/.../Thu_Apr_20_13:11:11_2023
    """
    # Handle trailing subdir_name or cam_dir
    m = re.search(r"(.*)/recordings/PNG/[^/]+/?$", rel_seq_dir.replace("\\", "/"))
    if not m:
        return None
    return m.group(1).rstrip("/")

@lru_cache(maxsize=4096)
def _load_json(p: str) -> dict | None:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _walk(obj):
    """Yield (key_lower, value) over nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k.strip().lower(), v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)

def _extract_wrist_and_ext_serials(meta: dict) -> tuple[str | None, list[str]]:
    wrist = None
    ext_items = []  # (idx, serial)

    for k, v in _walk(meta):
        if not isinstance(v, (str, int)):
            continue
        sv = str(v)

        # wrist
        if (wrist is None) and (k == "wrist_cam_serial" or k.endswith("/wrist_cam_serial")):
            wrist = sv

        # extN
        mm = re.match(r"ext(\d+)_cam_serial$", k)
        if mm:
            idx = int(mm.group(1))
            ext_items.append((idx, sv))

    # stable + dedup
    ext_items.sort(key=lambda x: x[0])
    ext_serials = []
    for _, s in ext_items:
        if s not in ext_serials:
            ext_serials.append(s)

    return wrist, ext_serials

@lru_cache(maxsize=8192)
def _meta_json_for_rel_base(droid_meta_root, rel_base: str) -> str | None:
    """
    rel_base:
    GuptaLab/failure/.../Thu_Apr_20_13:11:11_2023
    map to:
    <droid_meta_root>/<rel_base>/metadata_*.json
    """
    if not droid_meta_root:
        return None
    meta_dir = osp.join(droid_meta_root, rel_base)
    if not osp.isdir(meta_dir):
        return None

    # prefer metadata_*.json
    cands = sorted(glob.glob(osp.join(meta_dir, "metadata_*.json")))
    if cands:
        return cands[0]

    # fallback: any json in that folder
    cands = sorted(glob.glob(osp.join(meta_dir, "*.json")))
    return cands[0] if cands else None

def _keep_by_view_filter(only_fixed, only_wrist, ROOT_DIR, droid_meta_root, seq_dir_abs: str) -> bool:
    if not (only_fixed or only_wrist):
        return True

    # cam_id is the basename of .../recordings/PNG/<cam_id>
    cam_id = osp.basename(seq_dir_abs.rstrip("/"))

    rel_seq_dir = osp.relpath(seq_dir_abs, ROOT_DIR)
    rel_base = _strip_to_seq_base(rel_seq_dir)
    if rel_base is None:
        return False

    meta_path = _meta_json_for_rel_base(droid_meta_root, rel_base)
    if meta_path is None:
        return False

    meta = _load_json(meta_path)
    if not meta:
        return False

    wrist, ext_serials = _extract_wrist_and_ext_serials(meta)

    if only_fixed:
        return (cam_id in set(ext_serials)) and (len(ext_serials) > 0)

    # only_wrist
    return (wrist is not None) and (cam_id == wrist)

def make_dummy_intrinsic(image_shape, fov_deg=60.0):
    """
    image_shape: (H, W) or (H, W, C)
    fov_deg: assumed horizontal field of view in degrees
    """
    if len(image_shape) == 3:
        H, W = image_shape[:2]
    else:
        H, W = image_shape

    # principal point at image center
    cx = W / 2.0
    cy = H / 2.0

    # estimate focal length (pixels) from FOV
    fov_rad = np.deg2rad(fov_deg)
    f = 0.5 * W / np.tan(fov_rad / 2.0)

    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return K

def _natural_key(path: str):
    """
    Natural sort key: frame_2.png < frame_10.png
    """
    s = osp.basename(path)
    parts = _num_pat.split(s)
    return [int(p) if p.isdigit() else p for p in parts]

def _sample_indices(
    base_indices: List[int],
    img_per_seq: int,
    *,
    rng: random.Random = random,
) -> List[int]:
    if img_per_seq <= 0 or not base_indices:
        return []

    n = len(base_indices)

    # 1) Randomly choose fixed_interval, reduce until all frames fit
    fixed_interval = rng.choice([0, 1, 2, 3, 4, 5])
    while True:
        stride = fixed_interval + 1
        need = 1 + (img_per_seq - 1) * stride  # frames needed to fit k samples
        if need <= n or fixed_interval == 0:
            break
        fixed_interval -= 1

    stride = fixed_interval + 1
    # 2) Compute valid start range and sample a random start
    start_hi = n - need
    start = rng.randint(0, max(0, start_hi))

    # 3) Sample positions then map back to original frame indices
    pos = [start + t * stride for t in range(img_per_seq)]
    return [base_indices[p] for p in pos]

def load_depth(depthpath):
    """
    Returns
    -------
    depthmap : (H, W) float32
    valid   : (H, W) bool      True for reliable pixels
    """

    depthmap = imageio.v2.imread(depthpath).astype(np.float32) / 65535.0
    near_mask = depthmap < 0.0015   # 1. too close
    far_mask = depthmap > (65500.0 / 65535.0) # 2. filter sky
    # far_mask = depthmap > np.percentile(depthmap[~far_mask], 95) # 3. filter far area (optional)
    near, far = 1., 1000.
    depthmap = depthmap / (far - depthmap * (far - near)) / 0.004

    valid = ~(near_mask | far_mask)
    depthmap[~valid] = -1

    return depthmap, valid

class WildFramesFolderDataset(BaseDataset):
    “””
    Reads sequences from a pre-extracted frame directory:
    - Each subdirectory under ROOT_DIR is treated as one video sequence.
    - Each subdirectory contains frame images (png/jpg/...) and an optional fps.txt.
    - Sampling strategy and return structure are compatible with WildVideoFolderDataset
      for seamless drop-in replacement.

    Sampling parameters:
    - fix_img_num: -1 means variable; >0 fixes the number of sampled frames (uniform/random).
    “””
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ROOT_DIR: str = None,
        seq_glob: str = "*",
        len_train: int = 100000,
        len_test: int = 10000,
        robo: bool = False,
        game: bool = False,
        subdir_name: str = "color",
        metadata_csv: str | None = None,
        test_csv: str | None = None,
        split_key: str = "Index",                 # NEW
        seq_col: str = "Annotation Path",         # NEW
        require_seq_dir: bool = True,             # NEW
        val_json: str | None = None,   # NEW
        save_mask_dir: str | None = None,
        only_fixed: bool = False,
        only_wrist: bool = False,
        droid_meta_root: str | None = None,

        # sampling
        fix_img_num: int = -1,
        sampling: str = "uniform",
        frame_stride: int = 1,
        two_frame_strategy: str = "first_last",
    ):
        super().__init__(common_conf=common_conf)

        if ROOT_DIR is None:
            raise ValueError("ROOT_DIR must be specified for WildFramesFolderDataset.")

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = getattr(common_conf, "inside_random", True)
        self.img_nums = common_conf.img_nums

        assert two_frame_strategy in ("first_last", "ends_uniform")
        assert sampling in ("uniform", "random")
        self.two_frame_strategy = two_frame_strategy
        self.fix_img_num = fix_img_num
        self.sampling = sampling
        self.frame_stride = max(1, int(frame_stride))

        self.len_train = len_train if split == "train" else len_test

        self.ROOT_DIR = ROOT_DIR
        self.seq_glob = seq_glob
        self.robo = robo
        self.game = game
        self.subdir_name = subdir_name

        # save split parameters
        self.metadata_csv = metadata_csv
        self.test_csv = test_csv
        self.split_key = split_key
        self.seq_col = seq_col
        self.require_seq_dir = require_seq_dir
        self.val_json = val_json

        self.sequence_list: List[str] = []

        # ===== NEW: val_json split =====
        self.val_uids = set()
        if self.val_json is not None:
            self.val_uids = self._load_val_uids_from_json(self.val_json)

            if split in ("test", "val", "validation"):
                j = json.load(open(self.val_json, "r", encoding="utf-8"))
                eval_samples = []
                for it in j.get("items", []):
                    uid = (it.get("UID") or it.get("seq") or "").strip()
                    rel = self._resolve_rel_dir_for_uid(uid)
                    if rel is None:
                        continue
                    for fs in it.get("frame_sets", []):
                        ids = fs.get("ids", [])
                        if ids:
                            eval_samples.append((rel, ids))
                self.eval_samples = eval_samples
                self.sequence_list = [s[0] for s in eval_samples]
                self.len_train = len(self.eval_samples)  # make __len__ return correct value
                logging.info(f"[WildFrames][val_json] split={split}: eval_samples={len(self.eval_samples)}")
                return

        # ===== NEW: HOI4D meta/test split mode =====
        if self.metadata_csv is not None and self.test_csv is not None:
            seqs = self._build_seq_list_from_meta_minus_test(
                metadata_csv=self.metadata_csv,
                test_csv=self.test_csv,
                split=split,
                split_key=self.split_key,
                seq_col=self.seq_col,
            )
            min_frames, high = self.img_nums
            for seq in seqs:
                # images folder: ROOT_DIR/<Annotation Path>/[subdir_name]/...
                cand_dir = osp.join(ROOT_DIR, seq, subdir_name) if subdir_name else osp.join(ROOT_DIR, seq)
                if not osp.isdir(cand_dir):
                    # allow fallback
                    cand_dir2 = osp.join(ROOT_DIR, seq)
                    if not osp.isdir(cand_dir2):
                        if require_seq_dir:
                            continue
                        cand_dir = cand_dir2
                    else:
                        cand_dir = cand_dir2

                # count frames quickly (same as your glob logic)
                frame_count = 0
                for ext in IMG_EXTS:
                    frame_count += len(glob.glob(osp.join(cand_dir, f"*{ext}")))
                    if frame_count > high:
                        break

                if frame_count > high:
                    self.sequence_list.append(osp.relpath(cand_dir, ROOT_DIR))

                if self.debug and len(self.sequence_list) > 1:
                    break

        elif self.robo is True:
            seq_txt = "/home/nan.huang/code/vggt/training/data/sequence_list.txt"
            self._load_sequences_from_txt(seq_txt)
        else:
            min_frames, high = self.img_nums

            if only_fixed and only_wrist:
                raise ValueError("only_fixed and only_wrist cannot both be True.")

            candidates = sorted(glob.glob(osp.join(ROOT_DIR, seq_glob), recursive=True))
            for p in candidates:
                if not osp.isdir(p):
                    continue

                # seq_dir: actual image directory (handles optional subdir_name)
                seq_dir = osp.join(p, subdir_name) if (subdir_name and osp.isdir(osp.join(p, subdir_name))) else p

                # NEW: only keep fixed / wrist view
                if not _keep_by_view_filter(only_fixed, only_wrist, ROOT_DIR, droid_meta_root, seq_dir):
                    continue

                frame_count = 0
                for ext in IMG_EXTS:
                    frame_count += len(glob.glob(osp.join(seq_dir, f"*{ext}")))
                    if frame_count > high:
                        break

                if frame_count > high:
                    self.sequence_list.append(osp.relpath(seq_dir, ROOT_DIR))

                if self.debug and len(self.sequence_list) > 1:
                    break

        logging.info(f"[WildFrames] Found {len(self.sequence_list)} sequences under {ROOT_DIR}")
        # If training and val_json is given, exclude validation UIDs
        if split == "train" and self.val_uids:
            before = len(self.sequence_list)
            self.sequence_list = [
                s for s in self.sequence_list
                if self._uid_from_rel_dir(s) not in self.val_uids
            ]
            after = len(self.sequence_list)
            logging.info(f"[WildFrames][val_json] excluded {before-after} val sequences (remain {after})")

    def _read_csv_rows(self, path: str):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"[WildFrames] empty csv or no header: {path}")
            rows = []
            for r in reader:
                rows.append(r)
        return rows

    def _load_val_uids_from_json(self, val_json: str) -> set[str]:
        with open(val_json, "r", encoding="utf-8") as f:
            j = json.load(f)
        items = j.get("items", [])
        uids = set()
        for it in items:
            uid = (it.get("UID") or it.get("seq") or "").strip()
            if uid:
                uids.add(uid)
        return uids

    def _uid_from_rel_dir(self, rel_dir: str) -> str:
        # rel_dir could be ".../<UID>/color" or ".../<UID>"
        p = Path(rel_dir)
        if self.subdir_name and p.name == self.subdir_name:
            return p.parent.name
        return p.name

    def _resolve_rel_dir_for_uid(self, uid: str) -> str | None:
        """
        Given UID, find the actual sequence directory under ROOT_DIR.
        Prefer ROOT_DIR/<uid>/<subdir_name> if exists.
        Fallback to a few common layouts; if still not found, try a light glob search.
        Returns relpath to ROOT_DIR (same style as self.sequence_list elements).
        """
        candidates = [
            uid,
            osp.join("videos", "OmniWorld-Game", uid),
            osp.join("annotations", "OmniWorld-Game", uid),
            osp.join("OmniWorld-Game", uid),
        ]

        # try direct candidates first
        for c in candidates:
            if self.subdir_name and osp.isdir(osp.join(self.ROOT_DIR, c, self.subdir_name)):
                return osp.relpath(osp.join(self.ROOT_DIR, c, self.subdir_name), self.ROOT_DIR)
            if osp.isdir(osp.join(self.ROOT_DIR, c)):
                return osp.relpath(osp.join(self.ROOT_DIR, c), self.ROOT_DIR)

        # fallback: search by uid directory name (only 20 uids -> acceptable)
        hits = glob.glob(osp.join(self.ROOT_DIR, "**", uid), recursive=True)
        for h in hits:
            if not osp.isdir(h):
                continue
            if self.subdir_name and osp.isdir(osp.join(h, self.subdir_name)):
                return osp.relpath(osp.join(h, self.subdir_name), self.ROOT_DIR)
            # or directory itself contains images
            frame_count = 0
            for ext in IMG_EXTS:
                frame_count += len(glob.glob(osp.join(h, f"*{ext}")))
                if frame_count > 0:
                    return osp.relpath(h, self.ROOT_DIR)

        return None
    def _build_seq_list_from_meta_minus_test(
        self,
        metadata_csv: str,
        test_csv: str,
        split: str,
        *,
        split_key: str = "Index",
        seq_col: str = "Annotation Path",
    ) -> List[str]:
        meta_rows = self._read_csv_rows(metadata_csv)
        test_rows = self._read_csv_rows(test_csv)

        def get_key(r):
            if split_key == "Index":
                v = (r.get("Index") or "").strip()
                if v == "":
                    return None
                return int(float(v))  # robust: "23" / "23.0"
            else:
                v = (r.get("Annotation Path") or "").strip()
                return v if v else None

        def get_seq(r):
            v = (r.get(seq_col) or "").strip()
            return v if v else None

        meta_map = {}  # key -> seq
        for r in meta_rows:
            k = get_key(r)
            s = get_seq(r)
            if k is None or s is None:
                continue
            if k not in meta_map:
                meta_map[k] = s

        test_keys = set()
        for r in test_rows:
            k = get_key(r)
            if k is not None:
                test_keys.add(k)

        if split == "test":
            keys = [k for k in test_keys if k in meta_map]
        else:
            keys = [k for k in meta_map.keys() if k not in test_keys]

        # keys sort for deterministic order
        keys = sorted(keys)
        seqs = [meta_map[k] for k in keys]

        # de-dup preserve order
        seen = set()
        out = []
        for s in seqs:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # --------- required by BaseDataset style loaders ---------
    def __len__(self):
        if (not self.training) and hasattr(self, "eval_samples") and self.eval_samples:
            return len(self.eval_samples)
        return self.len_train

    def _uniform_indices(self, n_total: int, k: int) -> List[int]:
        if k <= 0:
            return []
        if k >= n_total:
            return list(range(n_total))
        # uniformly sample k indices
        return [int(round(i * (n_total - 1) / (k - 1))) for i in range(k)]

    def _read_frame_rgb(self, img_path: str) -> Optional[np.ndarray]:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)
    
    def _load_sequences_from_txt(self, seq_txt: str):
        if not osp.exists(seq_txt):
            raise FileNotFoundError(f"Sequence list file not found: {seq_txt}")

        with open(seq_txt, "r") as f:
            for line in f:
                seq = line.strip()
                if not seq:
                    continue
                self.sequence_list.append(seq)
                if self.debug and len(self.sequence_list) > 1:
                    break

    def _resize_letterbox(self, img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        """Proportional resize + center pad to target size (consistent with wild-videos/simple version)."""
        th, tw = target_hw
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, 3), dtype=resized.dtype)
        y0 = (th - nh) // 2
        x0 = (tw - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    def _collect_images(self, seq_dir: str) -> List[str]:
        paths = []
        abs_dir = osp.join(self.ROOT_DIR, seq_dir)
        for ext in IMG_EXTS:
            paths.extend(glob.glob(osp.join(abs_dir, f"*{ext}")))
        # Natural sort ensures frame_0001.png < frame_0010.png ordering
        paths = sorted(paths, key=_natural_key)
        return paths
    
    def _pick_other_seq(self, cur_rel_dir: str, rng=random) -> str:
        if len(self.sequence_list) <= 1:
            return cur_rel_dir
        for _ in range(10):
            cand = self.sequence_list[rng.randint(0, len(self.sequence_list) - 1)]
            if cand != cur_rel_dir:
                return cand
        return self.sequence_list[(self.sequence_list.index(cur_rel_dir) + 1) % len(self.sequence_list)]

    # --------- main API ---------
    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,   # relative directory name if caller specifies a sequence
        ids: list = None,       # optional: pass frame index list directly
        aspect_ratio: float = 1.0,
        return_distractors: bool = False,
        distractor_num: int = 12,
        distractor_load_gt: bool = False,   # distractors usually don't need depth; default False
    ) -> dict:
        if (not self.training) and hasattr(self, "eval_samples") and self.eval_samples:
            if seq_index is None:
                seq_index = 0
            seq_name, ids = self.eval_samples[int(seq_index) % len(self.eval_samples)]
        # Select a sequence directory
        if seq_name is not None:
            rel_dir = seq_name
        else:
            if self.inside_random or seq_index is None:
                seq_index = random.randint(0, max(0, len(self.sequence_list) - 1))
            rel_dir = self.sequence_list[seq_index]

        img_paths_all = self._collect_images(rel_dir)
        n_total = len(img_paths_all)

        if n_total == 0:
            raise RuntimeError(f"No images found in sequence: {osp.join(self.ROOT_DIR, rel_dir)}")
        
        # Select frame indices
        if ids is not None and len(ids) > 0:
            chosen_idx = [i for i in ids if 0 <= i < n_total]
        else:
            if  self.fix_img_num == -1:
                base = list(range(0, n_total, self.frame_stride))

                # Sample indices from base (respects frame_stride) satisfying span constraints
                chosen_idx = _sample_indices(
                    base_indices=base,
                    img_per_seq =int(img_per_seq),
                    rng=random,
                )
            else:
                k = self.fix_img_num if self.fix_img_num != -1 else (img_per_seq or 2)
                k = max(1, int(k))
                if self.sampling == "uniform":
                    chosen_idx = self._uniform_indices(n_total, k)
                else:
                    if k >= n_total:
                        chosen_idx = list(range(n_total))
                    else:
                        chosen_idx = sorted(random.sample(range(n_total), k))

        # Target image size
        target_image_shape = self.get_target_shape(aspect_ratio)

        images, original_sizes, frame_ids, used_img_paths = [], [], [], []
        gt_depths, gt_masks = [], []
        for idx in chosen_idx:
            img_path = img_paths_all[idx]
            rgb = self._read_frame_rgb(img_path)
            if rgb is None:
                continue
            original_size = np.array(rgb.shape[:2])
            original_sizes.append(original_size)  # H,W
            # resized = self._resize_letterbox(rgb, (target_h, target_w))
            
            intri_opencv = make_dummy_intrinsic(rgb.shape, fov_deg=60.0)
            
            img = self.process_one_image_wild(
                rgb,  
                intri_opencv,              
                original_size,
                target_image_shape,
            )
            
            images.append(img)
            frame_ids.append(idx)
            used_img_paths.append(img_path)
            
            # load gt depth if have
            if self.game:
                depth_path = img_path.replace("/color/", "/depth/")
                gt_d, gt_m = load_depth(depth_path) # [H,W]
                gt_depths.append(gt_d)
                gt_masks.append(gt_m)

        frame_num = len(images)
        if frame_num == 0:
            raise RuntimeError(f"All chosen frames failed to load in {rel_dir}")

        # Build batch dict with same fields as video dataset version
        batch = {
            "seq_name": rel_dir,                                   # relative subdirectory name under ROOT_DIR
            "ids": np.asarray(frame_ids, dtype=np.int64),          # frame indices relative to the sequence
            "frame_num": frame_num,
            "images": images,                                       # list of (H,W,3) uint8
            "original_sizes": original_sizes,                       # list of [H0,W0]
            "image_paths": used_img_paths,                          # absolute paths to images
        }

        H, W = images[0].shape[:2]
        N = len(images)
        if self.game:
            batch["depths"]       = gt_depths
            batch["point_masks"]  = gt_masks
        else: 
            batch["depths"]       = [np.zeros((H, W), dtype=np.float32) for _ in range(N)]
            batch["point_masks"]  = [np.ones((H, W), dtype=np.bool_)  for _ in range(N)]
        
        batch["cam_points"]   = [np.zeros((H, W, 3), dtype=np.float32) for _ in range(N)]
        batch["world_points"] = [np.zeros((H, W, 3), dtype=np.float32) for _ in range(N)]
        batch["extrinsics"]   = [np.eye(4, dtype=np.float32) for _ in range(N)]
        batch["intrinsics"]   = [np.eye(3, dtype=np.float32) for _ in range(N)]

        # ---------- distractors ----------
        if return_distractors and distractor_num > 0:
            d_rel = self._pick_other_seq(rel_dir, rng=random)
            d_img_paths_all = self._collect_images(d_rel)
            if len(d_img_paths_all) > 0:
                # Sample distractor_num frames (using the same _sample_indices for consistency)
                base_d = list(range(0, len(d_img_paths_all), self.frame_stride))
                d_idx = _sample_indices(base_indices=base_d, img_per_seq=int(distractor_num), rng=random)

                d_images, d_ids = [], []
                for idx in d_idx:
                    img_path = d_img_paths_all[idx]
                    rgb = self._read_frame_rgb(img_path)
                    if rgb is None:
                        continue
                    original_size = np.array(rgb.shape[:2])
                    intri_opencv = make_dummy_intrinsic(rgb.shape, fov_deg=60.0)
                    img = self.process_one_image_wild(rgb, intri_opencv, original_size, target_image_shape)
                    d_images.append(img)
                    d_ids.append(idx)

                batch["distractor_seq_name"] = d_rel
                batch["distractor_ids"] = np.asarray(d_ids, dtype=np.int64)
                batch["distractor_images"] = d_images
            else:
                batch["distractor_seq_name"] = d_rel
                batch["distractor_ids"] = np.zeros((0,), dtype=np.int64)
                batch["distractor_images"] = []
                
        return batch
