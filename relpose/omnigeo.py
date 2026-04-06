import os
import os.path as osp
import numpy as np
import torch

from typing import Optional, Union, Iterable, List, Dict
from torch.utils.data import Dataset

import rootutils
root = rootutils.setup_root(__file__, indicator="vggt", pythonpath=True)


def _try_int_stem(fname: str) -> int:
    stem = osp.splitext(fname)[0]
    try:
        return int(stem)
    except Exception:
        return 10**18  # push non-numeric to the end


def _sorted_frame_files(image_dir: str) -> List[str]:
    files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    files.sort(key=lambda x: (_try_int_stem(x), x))
    return files


class OmniGeoDataset(Dataset):
    """
    OmniGeo benchmark dataset.
    Directory structure:
      <UID>_<split_index>/
        image/000000.png ...
        camera_poses.npy   # (N,4,4) C2W
        intrinsics.npy     # (N,3,3)
        ...
    """

    def __init__(
        self,
        OmniGeo_DIR: str,
        min_num_images: int = 2,
        sort_by_filename: bool = True,
        cache_file: Optional[str] = None,
    ):
        self.OmniGeo_DIR = OmniGeo_DIR
        self.min_num_images = min_num_images
        self.sort_by_filename = sort_by_filename

        print(f"[OmniGeo] OmniGeo_DIR is {OmniGeo_DIR}")

        # Build lightweight metadata (paths + image file list + num_frames).
        # Optionally cache this dict.
        if cache_file is not None and osp.isfile(cache_file):
            print(f"[OmniGeo] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
        else:
            self.metadata: Dict[str, dict] = {}
            seqs = [
                d for d in os.listdir(OmniGeo_DIR)
                if osp.isdir(osp.join(OmniGeo_DIR, d))
            ]
            seqs.sort()

            for seq in seqs:
                seq_dir = osp.join(OmniGeo_DIR, seq)
                image_dir = osp.join(seq_dir, "image")
                cam_path = osp.join(seq_dir, "camera_poses.npy")
                intr_path = osp.join(seq_dir, "intrinsics.npy")

                if (not osp.isdir(image_dir)) or (not osp.isfile(cam_path)) or (not osp.isfile(intr_path)):
                    continue

                image_files = _sorted_frame_files(image_dir) if sort_by_filename else sorted(os.listdir(image_dir))
                if len(image_files) < self.min_num_images:
                    continue

                # Use mmap to cheaply read shape without loading full arrays into memory
                try:
                    cam_n = np.load(cam_path, mmap_mode="r").shape[0]
                    intr_n = np.load(intr_path, mmap_mode="r").shape[0]
                except Exception:
                    continue

                n = min(len(image_files), cam_n, intr_n)
                if n < self.min_num_images:
                    continue

                if (len(image_files) != cam_n) or (cam_n != intr_n):
                    print(f"[OmniGeo][WARN] Frame count mismatch in {seq}: "
                          f"images={len(image_files)}, cam={cam_n}, intr={intr_n}. Using n={n}.")

                self.metadata[seq] = {
                    "seq_dir": seq_dir,
                    "image_dir": image_dir,
                    "image_files": image_files,
                    "cam_path": cam_path,
                    "intr_path": intr_path,
                    "n": n,
                }

            if cache_file is not None:
                os.makedirs(osp.dirname(cache_file), exist_ok=True)
                np.save(cache_file, self.metadata)

        self.sequence_list = sorted(list(self.metadata.keys()))
        print(f"[OmniGeo] Data size: {len(self.sequence_list)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None) -> int:
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return int(self.metadata[sequence_name]["n"])

    def __getitem__(self, idx_N):
        # Keep the same calling convention as Re10K (index, n_per_seq)
        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        n = self.get_seq_framenum(sequence_name=sequence_name)
        n_per_seq = min(int(n_per_seq), n)
        ids = np.random.choice(n, n_per_seq, replace=False)
        return self.get_data(sequence_name=sequence_name, ids=ids)

    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
        ids: Union[Iterable, None] = None,
    ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]

        info = self.metadata[sequence_name]
        n = int(info["n"])

        if ids is None:
            ids = np.arange(n)
        else:
            ids = np.array(list(ids), dtype=np.int64)
            if ids.min() < 0 or ids.max() >= n:
                raise IndexError(f"[OmniGeo] ids out of range for {sequence_name}: n={n}, ids=[{ids.min()}, {ids.max()}]")

        # image paths
        image_files = info["image_files"]
        image_paths = [osp.join(info["image_dir"], image_files[i]) for i in ids]

        # load poses/intrinsics (stored as C2W) -> convert to W2C
        c2w = np.load(info["cam_path"])[:n]          # (n,4,4)
        intr = np.load(info["intr_path"])[:n]        # (n,3,3)

        c2w_sel = torch.from_numpy(c2w[ids]).float()     # (L,4,4)
        w2c_sel = torch.linalg.inv(c2w_sel)              # (L,4,4)

        intr_sel = torch.from_numpy(intr[ids]).float()   # (L,3,3)

        batch = {"seq_id": sequence_name, "n": n, "ind": torch.tensor(ids)}
        batch["image_paths"] = image_paths
        batch["extrs"] = w2c_sel
        batch["intrs"] = intr_sel
        return batch
