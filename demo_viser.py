# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
import json
import hashlib


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import re

def sort_by_index(paths):
    def extract_idx(p):
        filename = os.path.basename(p)
        stem, _ = os.path.splitext(filename)  # 去掉后缀
        # 找“结尾处的数字”
        m = re.search(r'(\d+)$', stem)
        if not m:
            return float('inf')
        return int(m.group(1))
    return sorted(paths, key=extract_idx)


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)
    
    EPS = 1e-5
    valid = np.isfinite(conf_flat) & (conf_flat > EPS)
    sorted_cache = {}

    def select_top_by_rank(drop_percent: float, frame_value: str) -> np.ndarray:
        """
        drop_percent: 0~100，表示丢掉最差的 drop_percent%
        frame_value: "All" 或者某一帧的字符串编号
        返回：要保留的点的 index（一维 index，对应 points_centered/colors_flat 的第一维）
        """
        if frame_value == "All":
            cand = np.nonzero(valid)[0]
        else:
            f = int(frame_value)
            cand = np.nonzero(valid & (frame_indices == f))[0]

        n = cand.size
        if n == 0:
            return cand

        keep = int(np.ceil((100.0 - float(drop_percent)) / 100.0 * n))
        keep = max(0, min(keep, n))

        if keep == 0:
            return cand[:0]
        if keep == n:
            return cand

        key = frame_value
        if key not in sorted_cache:
            c = conf_flat[cand] + 1e-12 * (cand.astype(np.float64) % 1000003)  # 可选 tie-break
            order = np.argsort(c, kind="stable")   # low -> high
            sorted_cache[key] = cand[order]

        sorted_cand = sorted_cache[key]
        return sorted_cand[-keep:]

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_keep_idx = select_top_by_rank(init_conf_threshold, gui_frame_selector.value)

    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_keep_idx],
        colors=colors_flat[init_keep_idx],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        drop_p = gui_points_conf.value
        sel = gui_frame_selector.value

        keep_idx = select_top_by_rank(drop_p, sel)
        point_cloud.points = points_centered[keep_idx]
        point_cloud.colors = colors_flat[keep_idx]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sort_by_index(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8081, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--sparse", action="store_true", help="keep ~10% frames (but at least --min_keep)")
parser.add_argument("--min_keep", type=int, default=2, help="when --sparse, keep at least this many frames")
parser.add_argument("--cache", action="store_true", help="Save/load predictions cache (.npz)")
parser.add_argument("--cache_path", type=str, default=None, help="Explicit cache .npz path (optional)")
parser.add_argument("--model", type=str, default="facebook/VGGT-1B", help="HuggingFace model repo ID to load (e.g. facebook/VGGT-1B or Changearthmore/SelfEvoVGGT)")

def save_full_predictions_npz(cache_path: str, pred_np: dict, image_names: list, ckpt_path: str):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    payload = {}
    for k, v in pred_np.items():
        if isinstance(v, np.ndarray):
            payload[k] = v
        elif isinstance(v, (int, float, bool, str)):
            payload[k] = np.array(v)
        else:
            # 尽量不动你现有逻辑；真遇到非数组/非标量就跳过并提示
            print(f"[cache] skip key={k}, type={type(v)} (not ndarray/scalar)")

    payload["__image_names__"] = np.array(image_names)
    payload["__ckpt_path__"] = np.array(ckpt_path)

    np.savez_compressed(cache_path, **payload)
    print(f"[cache] saved: {cache_path}")


def load_full_predictions_npz(cache_path: str, image_names: list):
    if (cache_path is None) or (not os.path.exists(cache_path)):
        return None

    try:
        data = np.load(cache_path, allow_pickle=False)
    except Exception as e:
        print(f"[cache] failed to load {cache_path}: {e}")
        return None

    if "__image_names__" in data.files:
        saved = data["__image_names__"].tolist()
        if list(saved) != list(image_names):
            print("[cache] image_names mismatch -> recompute")
            return None

    pred = {}
    for k in data.files:
        if k in ("__image_names__", "__ckpt_path__"):
            continue
        pred[k] = data[k]

    print(f"[cache] loaded: {cache_path}")
    return pred


def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Initializing and loading VGGT model from {args.model}...")
    model = VGGT.from_pretrained(args.model)

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = sort_by_index(glob.glob(os.path.join(args.image_folder, "*")))
    if args.sparse:
        num = len(image_names)
        target = max(int(args.min_keep), max(1, num // 10))
        target = min(target, num)
        idx = np.linspace(0, num - 1, target, dtype=int)
        image_names = [image_names[i] for i in idx]
        print(f"[sparse] keep {len(image_names)}/{num} (target={target}, min_keep={args.min_keep})")

    print(f"Found {len(image_names)} images")
    ckpt_path = "/home/nan.huang/code/vggt/training/logs/bot_995_loss_frcam/ckpts/checkpoint_14.pt"
    cache_path = args.cache_path or os.path.join(args.image_folder, "_pred_cache.npz")

    predictions_np = None
    if args.cache:
        predictions_np = load_full_predictions_npz(cache_path, image_names)

    if predictions_np is None:
        print(f"Loading images from {args.image_folder}...")
        images = load_and_preprocess_images(image_names).to(device)
        print(f"Preprocessed images shape: {images.shape}")

        print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # ✅ 最小但关键：保证 pred_dict 里一定有 images（你后面可视化要用）
        predictions["images"] = images

        print("Processing model outputs...")
        predictions_np = {}
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                print("Processing model outputs...")
                predictions_np = {}
                for key, v in predictions.items():
                    if isinstance(v, torch.Tensor):
                        arr = v.detach().cpu().numpy()
                        # 只在第一维确实是 1 的时候才去掉 batch 维
                        if arr.ndim > 0 and arr.shape[0] == 1:
                            arr = arr.squeeze(0)
                        predictions_np[key] = arr
                    else:
                        predictions_np[key] = v

            else:
                # 一般不会走到这，但保留
                predictions_np[key] = predictions[key]

        if args.cache:
            save_full_predictions_npz(cache_path, predictions_np, image_names, ckpt_path)

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")

    viser_server = viser_wrapper(
        predictions_np,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()
