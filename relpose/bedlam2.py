"""
BEDLAM2 Dataset Class for Relative Pose Evaluation

Follows the pattern of omnigeo.py for compatibility with the evaluation framework.
Handles Unreal Engine → OpenCV coordinate transformation.

Directory structure (BEDLAM2):
  <ROOT_DIR>/<session>/
    png/<session>/png/<seq_id>/<seq_id>_<frame>.png
    depth/<session>/exr_depth/<seq_id>/<seq_id>_<frame>.exr
    groundtruth/<session>/ground_truth/meta_exr_depth/<seq_id>/<seq_id>_<frame>_meta.json
"""

import os
import os.path as osp
import json
import re
import numpy as np
import torch

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterable
from torch.utils.data import Dataset

import sys
_eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pi3_root = os.path.join(os.path.dirname(_eval_dir), "uncharted4d-Pi3")
if os.path.isdir(_pi3_root):
    sys.path.insert(0, _pi3_root)
from pi3.utils.geometry import se3_inverse


# Pattern for natural sorting
_num_pat = re.compile(r"(\d+)")


def _natural_key(path: str):
    """Natural sort key: frame_2 < frame_10"""
    s = osp.basename(path)
    parts = _num_pat.split(s)
    return [int(p) if p.isdigit() else p for p in parts]


# Transformation matrix from Unreal to OpenCV world coordinates
# Unreal: X=Forward, Y=Right, Z=Up (left-handed, centimeters)
# OpenCV: X=Right, Y=Down, Z=Forward (right-handed, meters)
T_UNREAL_TO_OPENCV = np.array([
    [0,  1,  0],   # OpenCV X = Unreal Y (right)
    [0,  0, -1],   # OpenCV Y = -Unreal Z (down)
    [1,  0,  0]    # OpenCV Z = Unreal X (forward)
], dtype=np.float32)


def euler_to_rotation_matrix_unreal(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Convert Unreal Euler angles to a world-to-camera rotation matrix.

    Rotation order: Yaw -> Pitch -> Roll (ZYX intrinsic).
    Per BEDLAM convention: yaw is LEFT-handed around Z, pitch/roll are right-handed.
    The result is a w2c rotation (maps world vectors to camera-local vectors).
    """
    c_roll, s_roll = np.cos(roll), np.sin(roll)
    R_roll = np.array([
        [1, 0, 0],
        [0, c_roll, -s_roll],
        [0, s_roll, c_roll]
    ])

    c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
    R_pitch = np.array([
        [c_pitch, 0, s_pitch],
        [0, 1, 0],
        [-s_pitch, 0, c_pitch]
    ])

    c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([
        [c_yaw, s_yaw, 0],
        [-s_yaw, c_yaw, 0],
        [0, 0, 1]
    ])  # left-handed

    R = R_roll @ R_pitch @ R_yaw
    return R.astype(np.float32)


def parse_camera_params(meta_path: str, img_width: int = 1280, img_height: int = 720) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse camera parameters from BEDLAM2 metadata JSON.

    Args:
        meta_path: Path to *_meta.json file
        img_width: Image width (default 1280)
        img_height: Image height (default 720)

    Returns:
        K: (3, 3) intrinsic matrix
        c2w: (4, 4) camera-to-world transformation matrix in OpenCV coordinates
    """
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    focal_length = float(meta["unreal/camera/FinalImage/focalLength"])
    sensor_width = float(meta["unreal/camera/FinalImage/sensorWidth"])
    sensor_height = float(meta["unreal/camera/FinalImage/sensorHeight"])

    fx = focal_length * img_width / sensor_width
    fy = focal_length * img_height / sensor_height
    cx, cy = img_width / 2.0, img_height / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Camera position in Unreal coordinates (convert cm to meters)
    pos_unreal = np.array([
        float(meta["unreal/camera/curPos/x"]),
        float(meta["unreal/camera/curPos/y"]),
        float(meta["unreal/camera/curPos/z"])
    ], dtype=np.float32) / 100.0

    # Unreal Euler angles (degrees to radians)
    pitch = np.radians(float(meta["unreal/camera/curRot/pitch"]))
    yaw = np.radians(float(meta["unreal/camera/curRot/yaw"]))
    roll = np.radians(float(meta["unreal/camera/curRot/roll"]))

    # Get w2c rotation matrix in Unreal coordinates
    R_w2c_unreal = euler_to_rotation_matrix_unreal(pitch, yaw, roll)

    # Transform to OpenCV coordinates
    T = T_UNREAL_TO_OPENCV

    # Convert position: pos_opencv = T @ pos_unreal
    pos_opencv = T @ pos_unreal

    # Convert rotation: transpose w2c to get c2w, then change coordinate system
    R_c2w_opencv = T @ R_w2c_unreal.T @ T.T

    # Build 4x4 c2w (camera-to-world) matrix
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w_opencv
    c2w[:3, 3] = pos_opencv

    return K, c2w


class BEDLAM2Dataset(Dataset):
    """
    BEDLAM2 benchmark dataset for relative pose evaluation.

    Directory structure:
      <ROOT_DIR>/<session>/
        png/<session>/png/<seq_id>/<seq_id>_<frame>.png
        groundtruth/<session>/ground_truth/meta_exr_depth/<seq_id>/<seq_id>_<frame>_meta.json
    """

    def __init__(
        self,
        BEDLAM2_DIR: str,
        test_selection_file: Optional[str] = None,
        cache_file: Optional[str] = None,
        min_num_images: int = 2,
        img_width: int = 1280,
        img_height: int = 720,
    ):
        """
        Initialize BEDLAM2Dataset.

        Args:
            BEDLAM2_DIR: Root directory of BEDLAM2 dataset
            test_selection_file: JSON file with selected test sequences (session -> [seq_ids])
            cache_file: Optional cache file for metadata
            min_num_images: Minimum number of frames per sequence
            img_width: Image width for intrinsics calculation
            img_height: Image height for intrinsics calculation
        """
        self.BEDLAM2_DIR = BEDLAM2_DIR
        self.test_selection_file = test_selection_file
        self.min_num_images = min_num_images
        self.img_width = img_width
        self.img_height = img_height

        print(f"[BEDLAM2] BEDLAM2_DIR is {BEDLAM2_DIR}")

        # Load test selection if provided
        self.test_selection: Optional[Dict[str, List[str]]] = None
        if test_selection_file is not None and osp.isfile(test_selection_file):
            print(f"[BEDLAM2] Loading test selection from: {test_selection_file}")
            with open(test_selection_file, 'r') as f:
                self.test_selection = json.load(f)

        # Build lightweight metadata
        if cache_file is not None and osp.isfile(cache_file):
            print(f"[BEDLAM2] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
        else:
            self.metadata: Dict[str, dict] = {}
            self._collect_sequences()

            if cache_file is not None:
                os.makedirs(osp.dirname(cache_file), exist_ok=True)
                np.save(cache_file, self.metadata)
                print(f"[BEDLAM2] Saved cache to: {cache_file}")

        self.sequence_list = sorted(list(self.metadata.keys()))
        print(f"[BEDLAM2] Data size: {len(self.sequence_list)}")

    def _get_paths(self, session: str, seq_id: str) -> Tuple[Path, Path]:
        """Get paths to PNG and metadata directories for a sequence."""
        root = Path(self.BEDLAM2_DIR)
        png_dir = root / session / 'png' / session / 'png' / seq_id
        meta_dir = root / session / 'groundtruth' / session / 'ground_truth' / 'meta_exr_depth' / seq_id
        return png_dir, meta_dir

    def _collect_frames(self, png_dir: Path, seq_id: str) -> List[Tuple[str, int]]:
        """Collect all frame files and their frame IDs."""
        frames = []
        for f in sorted(png_dir.glob('*.png'), key=lambda p: _natural_key(str(p))):
            name = f.stem  # seq_000000_0042
            parts = name.split('_')
            if len(parts) >= 3:
                try:
                    frame_id = int(parts[-1])
                    frames.append((f.name, frame_id))
                except ValueError:
                    continue
        return frames

    def _collect_sequences(self):
        """Scan BEDLAM2_DIR for all valid sequences."""
        root = Path(self.BEDLAM2_DIR)

        # Determine which sessions to scan
        if self.test_selection is not None:
            sessions_to_scan = list(self.test_selection.keys())
        else:
            # Find all session directories
            skip_dirs = {'datasample', 'logs', '.git'}
            sessions_to_scan = []
            for item in sorted(root.iterdir()):
                if not item.is_dir():
                    continue
                if item.name.startswith('.'):
                    continue
                if item.name in skip_dirs:
                    continue
                if (item / 'png').exists():
                    sessions_to_scan.append(item.name)

        for session in sessions_to_scan:
            # Determine which sequences to include
            if self.test_selection is not None:
                seqs_to_include = set(self.test_selection.get(session, []))
            else:
                seqs_to_include = None  # Include all

            png_base = root / session / 'png' / session / 'png'
            meta_base = root / session / 'groundtruth' / session / 'ground_truth' / 'meta_exr_depth'

            if not png_base.exists():
                print(f"[BEDLAM2][WARN] PNG base not found: {png_base}")
                continue

            if not meta_base.exists():
                print(f"[BEDLAM2][WARN] Meta base not found: {meta_base}")
                continue

            for seq_dir in sorted(png_base.iterdir()):
                if not seq_dir.is_dir():
                    continue
                if not seq_dir.name.startswith('seq_'):
                    continue

                seq_id = seq_dir.name

                # Filter by test selection
                if seqs_to_include is not None and seq_id not in seqs_to_include:
                    continue

                # Collect frames
                frames = self._collect_frames(seq_dir, seq_id)
                if len(frames) < self.min_num_images:
                    continue

                # Check metadata exists
                meta_dir = meta_base / seq_id
                if not meta_dir.exists():
                    continue

                # Verify metadata count matches
                meta_files = list(meta_dir.glob('*_meta.json'))
                if len(meta_files) < self.min_num_images:
                    continue

                # Build frame list with validated metadata
                valid_frames = []
                for fname, fid in frames:
                    meta_path = meta_dir / f"{seq_id}_{fid:04d}_meta.json"
                    if meta_path.exists():
                        valid_frames.append((fname, fid))

                if len(valid_frames) < self.min_num_images:
                    continue

                n = len(valid_frames)

                # Use composite key: session/seq_id
                seq_key = f"{session}/{seq_id}"
                self.metadata[seq_key] = {
                    "session": session,
                    "seq_id": seq_id,
                    "png_dir": str(seq_dir),
                    "meta_dir": str(meta_dir),
                    "frames": valid_frames,  # List of (filename, frame_id)
                    "n": n,
                }

        print(f"[BEDLAM2] Collected {len(self.metadata)} valid sequences")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None) -> int:
        """Get the number of frames in a sequence."""
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return int(self.metadata[sequence_name]["n"])

    def __getitem__(self, idx_N):
        """Get data for a sequence with random frame sampling."""
        # Keep the same calling convention as OmniGeo (index, n_per_seq)
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
        """
        Load data for a sequence.

        Args:
            index: Sequence index (alternative to sequence_name)
            sequence_name: Sequence key in format "session/seq_id"
            ids: Frame indices to load (default: all frames)

        Returns:
            dict with keys:
                - seq_id: Sequence identifier
                - n: Total frame count
                - ind: Selected frame indices (tensor)
                - image_paths: List of image paths
                - extrs: W2C extrinsic matrices (L, 4, 4) - world-to-camera
                - intrs: Intrinsic matrices (L, 3, 3)
        """
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]

        info = self.metadata[sequence_name]
        n = int(info["n"])
        frames = info["frames"]  # List of (filename, frame_id)
        png_dir = Path(info["png_dir"])
        meta_dir = Path(info["meta_dir"])
        seq_id = info["seq_id"]

        if ids is None:
            ids = np.arange(n)
        else:
            ids = np.array(list(ids), dtype=np.int64)
            if ids.min() < 0 or ids.max() >= n:
                raise IndexError(f"[BEDLAM2] ids out of range for {sequence_name}: n={n}, ids=[{ids.min()}, {ids.max()}]")

        # Load data for selected frames (skip corrupted metadata gracefully)
        image_paths = []
        c2w_list = []
        intr_list = []
        valid_ids = []

        for i in ids:
            fname, fid = frames[i]
            img_path = png_dir / fname
            meta_path = meta_dir / f"{seq_id}_{fid:04d}_meta.json"

            # Parse camera parameters, skip corrupted/empty files
            try:
                K, c2w = parse_camera_params(str(meta_path), self.img_width, self.img_height)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[BEDLAM2][WARN] Skipping frame {fid} in {sequence_name}: {e}")
                continue

            image_paths.append(str(img_path))
            c2w_list.append(c2w)
            intr_list.append(K)
            valid_ids.append(i)

        if len(c2w_list) < 2:
            return None

        ids = np.array(valid_ids, dtype=np.int64)

        # Stack and convert to tensors
        c2w = torch.from_numpy(np.stack(c2w_list, axis=0)).float()  # (L, 4, 4)
        w2c = se3_inverse(c2w)  # (L, 4, 4) - world-to-camera
        intr = torch.from_numpy(np.stack(intr_list, axis=0)).float()  # (L, 3, 3)

        batch = {
            "seq_id": sequence_name,
            "n": n,
            "ind": torch.tensor(ids),
        }
        batch["image_paths"] = image_paths
        batch["extrs"] = w2c  # W2C for angular evaluation
        batch["intrs"] = intr

        return batch
