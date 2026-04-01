# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils.general import _unwrap_module, _minmax_norm


@dataclass
class AttnGuidedDropConfig:
    layer: int = 23
    attn_a: float = 0.5      # default: attention only (most stable); increase cos_a when token dims are verified
    cos_a: float = 0.5
    mode: str = "keep_bottom"   # keep_top / keep_bottom / random_like: which frames to keep
    global_feat_start: Optional[int] = 1024


class VGGTAttnGuidedScorer:
    """
    Extracts q/k from one teacher forward pass and computes per-frame scores relative to anchor frame 0.
    """
    def __init__(self, model: torch.nn.Module, cfg: AttnGuidedDropConfig):
        self.model = model
        self.cfg = cfg
        self.q_out: Dict[int, torch.Tensor] = {}
        self.k_out: Dict[int, torch.Tensor] = {}

    def _get_attn_blk(self, core: torch.nn.Module, layer: int):
        # paper release code: self.model.aggregator.global_blocks[i].attn
        return core.aggregator.global_blocks[layer].attn

    def _install_hooks(self) -> List[Any]:
        core = _unwrap_module(self.model)
        layer = self.cfg.layer
        blk = self._get_attn_blk(core, layer)

        def _make_hook(store: Dict[int, torch.Tensor], idx: int):
            def _hook(_module, _inp, out):
                store[idx] = out.detach()
            return _hook

        handles = []
        handles.append(blk.q_norm.register_forward_hook(_make_hook(self.q_out, layer)))
        handles.append(blk.k_norm.register_forward_hook(_make_hook(self.k_out, layer)))
        return handles

    @torch.no_grad()
    def forward_and_score_NN(
        self,
        images: torch.Tensor,          # (B,S,C,H,W)
        *,
        amp_dtype: Optional[torch.dtype] = None,
        return_aggregated_tokens: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        NxN token-level attention (frame->frame), then reduce to per-frame "incoming" score:
        attn_mat[i,j] = how much frame-i (as Q) attends to frame-j (as KV)

        This version REMOVES frame0 from both Q and KV when constructing attn_mat:
        attn_mat is (S-1)x(S-1) over frames 1..S-1, so frame0 won't dominate.
        Returned scores is (B,S) where:
        scores[:,0] = NaN
        scores[:,1:] = minmax-normalized incoming scores over frames 1..S-1

        return:
        preds: teacher output
        scores: (B,S) float32
        """
        self.q_out.clear()
        self.k_out.clear()

        B, S, C, H, W = images.shape
        if S < 2:
            raise ValueError(f"S={S} too short (need >=2)")

        image_hw = (H, W)
        handles = self._install_hooks()

        if amp_dtype is None:
            amp_dtype = torch.bfloat16

        # -------- forward (always run, cuda or cpu) --------
        try:
            if images.is_cuda:
                with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                    out = self.model(images=images, is_attn=return_aggregated_tokens)
            else:
                out = self.model(images=images, is_attn=return_aggregated_tokens)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        # -------- unpack output --------
        if isinstance(out, tuple) and len(out) == 2:
            preds, agg_tokens = out
        else:
            preds, agg_tokens = out, None

        # hook not captured -> fallback: NaN at frame0, zeros elsewhere
        if self.cfg.layer not in self.q_out or self.cfg.layer not in self.k_out:
            scores_full = torch.zeros((B, S), device=images.device, dtype=torch.float32)
            scores_full[:, 0] = float("nan")
            return preds, scores_full

        Q = self.q_out[self.cfg.layer]  # (B, Hh, T, D)
        K = self.k_out[self.cfg.layer]  # (B, Hh, T, D)

        core = _unwrap_module(self.model)
        aggregator = core.aggregator
        patch_size = aggregator.patch_size
        patch_start_idx = aggregator.patch_start_idx

        h_patches = image_hw[0] // patch_size
        w_patches = image_hw[1] // patch_size
        num_patch_tokens = h_patches * w_patches
        tokens_per_image = patch_start_idx + num_patch_tokens

        # how many images are actually in token sequence
        T = int(K.shape[-2])
        num_images_in_seq = T // tokens_per_image
        num_vis = min(S, num_images_in_seq)

        # keep your strict behavior: refuse padding
        if num_vis < S:
            raise ValueError(
                f"[Score] num_vis < S, refuse to pad. "
                f"S={S}, num_vis={num_vis}. "
                f"Likely tokens_per_image mismatch or model truncated frames."
            )

        Hh = int(Q.shape[1])
        D  = int(Q.shape[-1])

        Tk = int(num_vis * tokens_per_image)
        # (B,Hh,Tk,D)
        K_flat = K[:, :, :Tk, :]
        Q_flat = Q[:, :, :Tk, :]

        # reshape into per-frame blocks
        # (B,Hh,num_vis,tpi,D)
        K_blk = K_flat.view(B, Hh, num_vis, tokens_per_image, D)
        Q_blk = Q_flat.view(B, Hh, num_vis, tokens_per_image, D)

        # drop frame0 from BOTH Q and KV
        # keep frames [1..num_vis-1]
        if num_vis <= 1:
            scores_full = torch.zeros((B, S), device=images.device, dtype=torch.float32)
            scores_full[:, 0] = float("nan")
            return preds, scores_full

        K_blk_wo0 = K_blk[:, :, 1:, :, :]                       # (B,Hh,S2,tpi,D)
        Q_blk_wo0 = Q_blk[:, :, 1:, :, :]                       # (B,Hh,S2,tpi,D)
        S2 = num_vis - 1

        K_flat_wo0 = K_blk_wo0.reshape(B, Hh, S2 * tokens_per_image, D)  # (B,Hh,S2*tpi,D)

        scale = 1.0 / math.sqrt(float(D))

        # (B,S2,S2): row=query frame (i+1), col=key frame (j+1)
        attn_mat = torch.empty((B, S2, S2), device=images.device, dtype=torch.float32)

        for i in range(S2):
            # q_i: (B,Hh,Np,D) for frame (i+1) patch queries
            q_i = Q_blk_wo0[:, :, i, patch_start_idx:patch_start_idx + num_patch_tokens, :]

            logits = torch.einsum("bhqd,bhtd->bhqt", q_i, K_flat_wo0) * scale  # (B,Hh,Np,S2*tpi)
            probs  = torch.softmax(logits, dim=-1)

            # average over heads & query patches -> (B, S2*tpi)
            attn_all = probs.mean(dim=1).mean(dim=1).to(torch.float32)

            # (B,S2,tpi) -> take key-frame patch tokens -> mean over key patches -> (B,S2)
            attn_blocks = attn_all.view(B, S2, tokens_per_image)
            patch_block = attn_blocks[:, :, patch_start_idx:patch_start_idx + num_patch_tokens]  # (B,S2,Np)
            attn_mat[:, i, :] = patch_block.mean(dim=-1)  # (B,S2)

        # ---- reduce NxN -> per-frame incoming score (B,S2) ----
        eye = torch.eye(S2, device=attn_mat.device, dtype=torch.bool).unsqueeze(0)  # (1,S2,S2)
        attn_no_self = attn_mat.masked_fill(eye, 0.0)

        attn_scores_wo0 = attn_no_self.sum(dim=1) / max(1, (S2 - 1))  # (B,S2)
        attn_n_wo0 = _minmax_norm(attn_scores_wo0, dim=1)             # (B,S2)

        # ---- cosine branch (also drop frame0 for consistency) ----
        cos_n_wo0 = torch.zeros_like(attn_n_wo0)

        if (self.cfg.cos_a > 0) and (agg_tokens is not None):
            try:
                feat = agg_tokens[self.cfg.layer]

                if feat.ndim == 4:
                    # expected (B,S,T,C) in your heuristic; we only keep frames 1..S-1
                    feat_bstc = feat[:, 1:, :, :]  # (B,S2,T,C)

                    if (self.cfg.global_feat_start is not None) and (feat_bstc.shape[-1] > self.cfg.global_feat_start):
                        feat_bstc = feat_bstc[:, :, :, self.cfg.global_feat_start:]

                    if feat_bstc.shape[2] > patch_start_idx:
                        feat_bstc = feat_bstc[:, :, patch_start_idx:, :]

                    # (B,S2,C): per-frame vector
                    feat_mean = feat_bstc.float().mean(dim=2)
                    feat_mean = F.normalize(feat_mean, p=2, dim=-1)

                    # (B,S2,S2): cosine similarity
                    cos_mat = torch.matmul(feat_mean, feat_mean.transpose(1, 2))

                    eye2 = torch.eye(S2, device=cos_mat.device, dtype=torch.bool).unsqueeze(0)
                    cos_no_self = cos_mat.masked_fill(eye2, 0.0)
                    cos_scores_wo0 = cos_no_self.sum(dim=1) / max(1, (S2 - 1))  # (B,S2)
                    cos_n_wo0 = _minmax_norm(cos_scores_wo0, dim=1)
            except Exception:
                pass

        # ---- mix branches and re-pack to (B,S) with frame0=NaN ----
        scores_wo0 = (self.cfg.attn_a * attn_n_wo0 + self.cfg.cos_a * cos_n_wo0).to(torch.float32)  # (B,S2)

        scores_full = torch.full((B, S), float("nan"), device=images.device, dtype=torch.float32)
        scores_full[:, 1:] = scores_wo0  # frame0 stays NaN

        return preds, scores_full, agg_tokens
