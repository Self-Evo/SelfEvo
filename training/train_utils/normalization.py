# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from typing import Optional, Tuple
from vggt.utils.geometry import closed_form_inverse_se3
from train_utils.general import check_and_fix_inf_nan


def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")

@torch.no_grad()
def lift_depth_to_cam_world_points_torch(
    depths: torch.Tensor,       # (B,L,H,W)  or (B,L,H,W,1)
    extrinsics: torch.Tensor,   # (B,L,3,4)  camera-from-world (OpenCV)
    intrinsics: torch.Tensor,   # (B,L,3,3)
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch/GPU batched version of:
      cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)
      cam_to_world = inverse(extrinsic)
      world = cam @ Rcw^T + tcw

    Returns:
      cam_points   : (B,L,H,W,3)
      world_points : (B,L,H,W,3)
      mask         : (B,L,H,W)  depth>eps and finite
    """
    device = depths.device

    # --- sanitize shapes ---
    if depths.dim() == 5 and depths.size(-1) == 1:
        depths = depths[..., 0]
    assert depths.dim() == 4, f"depths should be (B,L,H,W), got {depths.shape}"
    assert extrinsics.shape[-2:] == (3,4), f"extrinsics should be (...,3,4), got {extrinsics.shape}"
    assert intrinsics.shape[-2:] == (3,3), f"intrinsics should be (...,3,3), got {intrinsics.shape}"

    # do math in fp32 for stability
    depths_f = depths.float()
    extr_f   = extrinsics.float()
    intr_f   = intrinsics.float()

    B, L, H, W = depths_f.shape

    # --- point_mask: same as numpy (depth > eps) + finite ---
    mask = (depths_f > eps) & torch.isfinite(depths_f)

    # =========================
    # 1) depth_to_cam_coords_points (torch version of your numpy)
    # =========================
    fu = intr_f[..., 0, 0]   # (B,L)
    fv = intr_f[..., 1, 1]
    cu = intr_f[..., 0, 2]
    cv = intr_f[..., 1, 2]

    # meshgrid: u in [0..W-1], v in [0..H-1]
    # shapes: (H,W)
    u = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W).expand(B, L, H, W)
    v = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1).expand(B, L, H, W)

    # broadcast intrinsics to (B,L,H,W)
    fu_ = fu.view(B, L, 1, 1)
    fv_ = fv.view(B, L, 1, 1)
    cu_ = cu.view(B, L, 1, 1)
    cv_ = cv.view(B, L, 1, 1)

    x_cam = (u - cu_) * depths_f / (fu_ + 1e-12)
    y_cam = (v - cv_) * depths_f / (fv_ + 1e-12)
    z_cam = depths_f
    cam_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (B,L,H,W,3)

    # =========================
    # 2) cam_to_world_extrinsic = inverse(extrinsic)  (torch version)
    # =========================
    # extrinsic is (3,4), convert to (4,4)
    bottom = torch.zeros((B, L, 1, 4), device=device, dtype=extr_f.dtype)
    bottom[..., 0, 3] = 1.0
    extr44 = torch.cat([extr_f, bottom], dim=-2)  # (B,L,4,4)

    # closed_form_inverse_se3 expects (N,4,4) in many implementations; flatten safely
    extr44_flat = extr44.view(B * L, 4, 4)
    cam_to_world_flat = closed_form_inverse_se3(extr44_flat)  # (B*L,4,4)
    cam_to_world = cam_to_world_flat.view(B, L, 4, 4)

    Rcw = cam_to_world[..., :3, :3]  # (B,L,3,3)
    tcw = cam_to_world[..., :3,  3]  # (B,L,3)

    # =========================
    # 3) world = cam @ Rcw^T + tcw   (use bmm to avoid broadcast shape bugs)
    # =========================
    cam_flat = cam_points.view(B * L, H * W, 3)          # (BL,HW,3)
    R_flat   = Rcw.view(B * L, 3, 3)                     # (BL,3,3)
    t_flat   = tcw.view(B * L, 1, 3)                     # (BL,1,3)

    world_flat = torch.bmm(cam_flat, R_flat.transpose(1, 2)) + t_flat  # (BL,HW,3)
    world_points = world_flat.view(B, L, H, W, 3)

    return cam_points, world_points, mask

def normalize_camera_extrinsics_and_points_batch_gpu(
    extrinsics: torch.Tensor,
    cam_points: torch.Tensor,
    world_points: torch.Tensor,
    depths: torch.Tensor,
    scale_by_points: bool = True,
    point_masks: torch.Tensor = None,
):
    """
    这是 VGGT 的 normalize_camera_extrinsics_and_points_batch 的 GPU 版本：
    - 除了移除 “必须是 CPU” 的 assert，其余逻辑保持一致。
    """
    # ---- 原 VGGT：check_valid_tensor(...) 这些你项目里应该已经有 ----
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")

    B, S, _, _ = extrinsics.shape
    device = extrinsics.device

    # ===== 改动点：删掉这行 =====
    # assert device == torch.device("cpu")

    # Convert extrinsics to homogeneous form: (B,S,4,4)
    extrinsics_homog = torch.cat(
        [extrinsics, torch.zeros((B, S, 1, 4), device=device)],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv = inverse of first cam extrinsic
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])

    # new_extrinsics = extrinsics_homog @ first_cam_extrinsic_inv
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,S,4,4)

    if world_points is not None:
        # transform world points to first camera coordinate system
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + \
                           t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None

    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1, 2, 3])
        valid_count = point_masks.sum(dim=[1, 2, 3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)

        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = new_extrinsics[:, :, :3]  # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    return new_extrinsics, new_cam_points, new_world_points, new_depths

def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)


        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)


    return new_extrinsics, new_cam_points, new_world_points, new_depths





