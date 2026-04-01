# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils.general import _unwrap_module


def _rep_feature_loss(
    f_s: torch.Tensor,  # (B,L,C)
    f_t: torch.Tensor,  # (B,L,C)
    kind: str = "cosine",  # "cosine" | "smoothl1" | "mse" | "cosine_smoothl1"
    smoothl1_beta: float = 0.05,
    mix_alpha: float = 0.5,  # only for cosine_smoothl1
) -> torch.Tensor:
    f_s = f_s.float()
    f_t = f_t.float()

    if kind == "cosine":
        f_s = F.normalize(f_s, dim=-1)
        f_t = F.normalize(f_t, dim=-1)
        return (2.0 - 2.0 * (f_s * f_t).sum(dim=-1)).mean()

    if kind == "smoothl1":
        return F.smooth_l1_loss(f_s, f_t, beta=smoothl1_beta)

    if kind == "mse":
        return F.mse_loss(f_s, f_t)

    if kind == "cosine_smoothl1":
        loss_cos = _rep_feature_loss(f_s, f_t, kind="cosine")
        loss_l1  = _rep_feature_loss(f_s, f_t, kind="smoothl1", smoothl1_beta=smoothl1_beta)
        return mix_alpha * loss_cos + (1.0 - mix_alpha) * loss_l1

    raise ValueError(f"Unknown rep loss kind: {kind}")


def _frame_rep_from_agg_tokens(
    model: nn.Module,
    agg_tokens,
    images: torch.Tensor,   # (B,S,C,H,W)
    layer: int,
    global_feat_start: int = 1024,
) -> torch.Tensor:
    """
    Returns per-frame features: (B,S,C).
    Compatible with agg_tokens[layer] shaped (B,S,T,C) or (B,T,C);
    the latter is reshaped using tokens_per_image.
    """
    feat = agg_tokens[layer]  # expected token features
    core = _unwrap_module(model)
    aggregator = core.aggregator
    patch_size = aggregator.patch_size
    patch_start_idx = aggregator.patch_start_idx

    B, S, _, H, W = images.shape
    h_patches = H // patch_size
    w_patches = W // patch_size
    num_patch_tokens = h_patches * w_patches
    tokens_per_image = patch_start_idx + num_patch_tokens

    if feat.ndim == 4:
        # (B,S,T,C)
        x = feat
    elif feat.ndim == 3:
        # (B, S*tpi, C) -> reshape
        T = feat.shape[1]
        num_vis = T // tokens_per_image
        if num_vis < S:
            raise ValueError(f"[Rep] num_vis < S: S={S}, num_vis={num_vis}. tokens_per_image mismatch?")
        x = feat[:, :S * tokens_per_image, :].view(B, S, tokens_per_image, -1)
    else:
        raise ValueError(f"[Rep] Unexpected feat.ndim={feat.ndim}")

    x = x[:, :, patch_start_idx:, :]  # (B,S,Np,C)

    if global_feat_start:
        x = x[:, :, :, global_feat_start:]

    f = x.float().mean(dim=2)  # (B,S,C)
    return f
