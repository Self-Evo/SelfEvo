# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CropAugConfig:
    enabled: bool = True
    scale_min: float = 0.6
    scale_max: float = 0.9
    keep_aspect: bool = True
    resize_back: bool = True
    img_mode: str = "bilinear"
    depth_mode: str = "bilinear"
    conf_mode: str = "bilinear"
    mask_mode: str = "nearest"
    align_corners: bool = False  # for bilinear/bicubic
    min_crop_size: int = 16

def _sample_crop_boxes_per_sample(
    B: int, H: int, W: int, cfg: CropAugConfig, device: torch.device
) -> Dict[str, torch.Tensor]:
    # do_crop mask
    if not cfg.enabled:
        do_crop = torch.zeros(B, device=device, dtype=torch.bool)
    else:
        do_crop = torch.ones(B, device=device, dtype=torch.bool)

    # sample scale
    s = torch.empty(B, device=device).uniform_(cfg.scale_min, cfg.scale_max)

    if cfg.keep_aspect:
        crop_h = torch.round(torch.tensor(H, device=device) * s).long()
        crop_w = torch.round(torch.tensor(W, device=device) * s).long()
    else:
        sy = torch.empty(B, device=device).uniform_(cfg.scale_min, cfg.scale_max)
        sx = torch.empty(B, device=device).uniform_(cfg.scale_min, cfg.scale_max)
        crop_h = torch.round(torch.tensor(H, device=device) * sy).long()
        crop_w = torch.round(torch.tensor(W, device=device) * sx).long()

    # clamp
    crop_h = crop_h.clamp(min=cfg.min_crop_size, max=H)
    crop_w = crop_w.clamp(min=cfg.min_crop_size, max=W)

    # if no crop, box = full image
    crop_h = torch.where(do_crop, crop_h, torch.full_like(crop_h, H))
    crop_w = torch.where(do_crop, crop_w, torch.full_like(crop_w, W))

    # sample top/left (vectorized: rand * (max+1))
    max_top = (H - crop_h).clamp(min=0)
    max_left = (W - crop_w).clamp(min=0)

    top = torch.floor(torch.rand(B, device=device) * (max_top.to(torch.float32) + 1.0)).long()
    left = torch.floor(torch.rand(B, device=device) * (max_left.to(torch.float32) + 1.0)).long()

    # resize scales (output is original H,W if resize_back)
    out_h, out_w = (H, W)
    sy = out_h / crop_h.to(torch.float32)
    sx = out_w / crop_w.to(torch.float32)

    return dict(left=left, top=top, crop_w=crop_w, crop_h=crop_h, sx=sx, sy=sy, do_crop=do_crop)


def _crop_resize_blchw(
    x: torch.Tensor,  # (B,L,C,H,W)
    *,
    left: torch.Tensor, top: torch.Tensor, crop_w: torch.Tensor, crop_h: torch.Tensor,
    out_hw: Tuple[int, int],
    mode: str,
    align_corners: bool,
) -> torch.Tensor:
    assert x.ndim == 5, f"expected (B,L,C,H,W), got {tuple(x.shape)}"
    B, L, C, H, W = x.shape
    out_h, out_w = out_hw

    outs = []
    for b in range(B):
        l = int(left[b].item()); t = int(top[b].item())
        cw = int(crop_w[b].item()); ch = int(crop_h[b].item())
        xb = x[b]  # (L,C,H,W)
        crop = xb[:, :, t:t+ch, l:l+cw]  # (L,C,ch,cw)

        if (ch != out_h) or (cw != out_w):
            if mode in ["bilinear", "bicubic"]:
                crop = F.interpolate(crop, size=(out_h, out_w), mode=mode, align_corners=align_corners)
            else:
                crop = F.interpolate(crop, size=(out_h, out_w), mode=mode)
        outs.append(crop)
    return torch.stack(outs, dim=0)  # (B,L,C,out_h,out_w)


def _crop_resize_blhw(
    x: torch.Tensor,  # (B,L,H,W)
    *,
    left: torch.Tensor, top: torch.Tensor, crop_w: torch.Tensor, crop_h: torch.Tensor,
    out_hw: Tuple[int, int],
    mode: str,
    align_corners: bool,
    is_mask: bool = False,
) -> torch.Tensor:
    assert x.ndim == 4, f"expected (B,L,H,W), got {tuple(x.shape)}"
    B, L, H, W = x.shape
    out_h, out_w = out_hw

    x_in = x
    if is_mask:
        x_in = x_in.to(torch.float32)

    # add channel dim => (B,L,1,H,W)
    x_in = x_in.unsqueeze(2)

    outs = []
    for b in range(B):
        l = int(left[b].item()); t = int(top[b].item())
        cw = int(crop_w[b].item()); ch = int(crop_h[b].item())
        xb = x_in[b]  # (L,1,H,W)
        crop = xb[:, :, t:t+ch, l:l+cw]  # (L,1,ch,cw)

        if (ch != out_h) or (cw != out_w):
            if mode in ["bilinear", "bicubic"]:
                crop = F.interpolate(crop, size=(out_h, out_w), mode=mode, align_corners=align_corners)
            else:
                crop = F.interpolate(crop, size=(out_h, out_w), mode=mode)
        outs.append(crop)

    y = torch.stack(outs, dim=0)  # (B,L,1,out_h,out_w)
    y = y.squeeze(2)              # (B,L,out_h,out_w)

    if is_mask:
        y = (y > 0.5)
    return y


def _update_intrinsics_for_crop_resize(
    K: torch.Tensor,  # (B,L,3,3)
    *,
    left: torch.Tensor, top: torch.Tensor,
    crop_w: torch.Tensor, crop_h: torch.Tensor,
    out_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Computes K' after cropping at (left, top, cw, ch) and resizing to out_hw.
    Standard pinhole adjustment:
      fx' = fx * sx
      fy' = fy * sy
      cx' = (cx - left) * sx
      cy' = (cy - top ) * sy
    """
    assert K.ndim == 4 and K.shape[-2:] == (3,3), f"K expected (B,L,3,3), got {tuple(K.shape)}"
    B, L = K.shape[:2]
    out_h, out_w = out_hw

    # (B,1) broadcast to (B,L)
    left_f = left.to(K.device).to(torch.float32).view(B, 1)
    top_f  = top.to(K.device).to(torch.float32).view(B, 1)
    cw_f   = crop_w.to(K.device).to(torch.float32).view(B, 1)
    ch_f   = crop_h.to(K.device).to(torch.float32).view(B, 1)

    sx = (float(out_w) / cw_f)  # (B,1)
    sy = (float(out_h) / ch_f)  # (B,1)

    K2 = K.clone().to(torch.float32)

    fx = K2[:, :, 0, 0]
    fy = K2[:, :, 1, 1]
    cx = K2[:, :, 0, 2]
    cy = K2[:, :, 1, 2]

    fx = fx * sx
    fy = fy * sy
    cx = (cx - left_f) * sx
    cy = (cy - top_f)  * sy

    K2[:, :, 0, 0] = fx
    K2[:, :, 1, 1] = fy
    K2[:, :, 0, 2] = cx
    K2[:, :, 1, 2] = cy

    # scale skew (element [0,1]) by sx; typically 0, so this has no effect:
    K2[:, :, 0, 1] = K2[:, :, 0, 1] * sx

    return K2.to(dtype=K.dtype)
