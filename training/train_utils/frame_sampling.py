# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from train_utils.general import _gather_BS
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def _fixed_equal_with_endpoints(S: int, L: int, *, rng) -> List[int]:
    """Most uniform discrete equal spacing: gap difference <= 1, spanning [0, S-1]."""
    L = max(2, min(L, S))
    if L == 2:
        return [0, S - 1]
    n_gaps = L - 1
    total = S - 1
    base = total // n_gaps
    rem  = total % n_gaps
    long_pos = set(rng.sample(range(n_gaps), rem)) if rem > 0 else set()
    seq = [0]
    acc = 0
    for i in range(n_gaps):
        gap = base + (1 if i in long_pos else 0)
        acc += gap
        seq.append(acc)
    return seq


def _resolve_collisions(seq: List[int], S: int, step: int, rng) -> List[int]:
    """
    For duplicate values, nudge them to the nearest empty slot (left or right).
    Prefer closer positions within bounds; maximum nudge distance is at most step.
    """
    used = set()
    out: List[int] = []

    # helper: find the nearest available slot for target t
    def place_near(t: int) -> Optional[int]:
        if t not in used and 0 <= t < S:
            return t
        max_nudge = max(1, step - 1)  # jitter stays < step; keep nudge within same range
        # try positions from nearest to farthest, alternating left/right
        for d in range(1, max_nudge + 1):
            left  = t - d
            right = t + d
            pick_order = []
            # randomize left/right priority to avoid systematic bias
            if rng.random() < 0.5:
                pick_order = [left, right]
            else:
                pick_order = [right, left]
            for cand in pick_order:
                if 0 <= cand < S and cand not in used:
                    return cand
        return None

    for x in sorted(seq):
        spot = place_near(x)
        if spot is not None:
            used.add(spot)
            out.append(spot)
        # if no slot found, skip for now; fill gaps later

    return out


def sample_jittered_fixed(
    S: int,
    L: int,
    *,
    rng,
    max_jitter: Optional[int] = 1,  # jitter magnitude per point; if None uses min(1, step-1)
) -> List[int]:
    """
    Take L indices at equal spacing (step), then randomly jitter each point by < step,
    then sort, deduplicate, and nudge duplicates to guarantee exactly L unique indices in [0, S-1].
    """
    if S < 2:
        raise ValueError(f"S={S} is too short")
    L = max(2, min(L, S))

    # ----- choose step from [min(s_low,3), ..., allowed_max_step] -----
    # allowed_max_step from length constraint: 1 + (L-1)*step <= S  => step <= (S-1)//(L-1)
    allowed_max_step = max(1, (S - 1) // (L - 1))
    s_low = 3
    if s_low > allowed_max_step:
        step = allowed_max_step
    else:
        candidates = [s for s in range(s_low, allowed_max_step + 1)]
        if not candidates:
            # fallback: uniformly spaced indices differing by at most 1
            return _fixed_equal_with_endpoints(S, L, rng=rng)

        step = rng.choice(candidates)

    # start position ensuring all L frames fit: need = 1 + (L-1)*step
    need = 1 + (L - 1) * step
    start_hi = S - need
    start = rng.randint(0, max(0, start_hi))

    # base equally-spaced sequence
    base = [start + i * step for i in range(L)]

    # jitter: magnitude must be < step
    if max_jitter is None:
        jitter_max = min(1, step - 1)
    else:
        jitter_max = min(max(0, max_jitter), max(0, step - 1))

    if jitter_max > 0:
        jittered = [min(S - 1, max(0, x + rng.randint(-jitter_max, jitter_max))) for x in base]
    else:
        jittered = base[:]

    jittered[0] = 0
    # sort + deduplicate via collision resolution, guaranteeing exactly L unique indices
    seq = _resolve_collisions(jittered, S, step, rng)
    # if still short (rare edge case), fill up from remaining positions
    if len(seq) < L:
        pool = [i for i in range(S) if i not in seq]
        rng.shuffle(pool)
        seq += pool[: (L - len(seq))]

    seq = sorted(seq)[:L]
    return seq


def _choose_strategy(rng, p_fixed: float = 0.3) -> str:
    """Randomly returns 'fixed' or 'random'; p_fixed is the probability of fixed-interval."""
    return "fixed" if rng.random() < p_fixed else "random"


def sample_one_set(
    S: int,
    k_min: int = 2,
    k_max: int = 6,
    *,
    strategy: str = "random",     # "random" uses first+last+random-middle strategy
    must_include_first_last: bool = True,
    rng = random,
) -> List[int]:
    if S < 2:
        raise ValueError(f"S={S} is too short")

    L = rng.randint(max(2, k_min), max(2, k_max))
    L = max(2, min(L, S))  # cap at S

    if strategy == "random":
        if not must_include_first_last:
            pool_start, pool_end = 1, max(1, S)  # half-open interval [1, S-1)
            pool = list(range(pool_start, pool_end))
            m = min(L - 1, len(pool))
            mids = rng.sample(pool, k=m)
            return [0] + sorted(mids)

        if L == 2:
            return [0, S - 1]
        pool_start, pool_end = 1, max(1, S - 1)  # half-open interval [1, S-1)
        pool = list(range(pool_start, pool_end))
        m = min(L - 2, len(pool))
        mids = rng.sample(pool, k=m)
        return [0] + sorted(mids) + [S - 1]

    if strategy == "fixed":
        return sample_jittered_fixed(S, L, rng=rng, max_jitter=1)


def sample_one_set_by_scores_batch_prob(
    scores: torch.Tensor,            # (B,S)
    *,
    k_min: int = 2,
    k_max: int = 12,
    mode: str = "keep_bottom",       # keep_top / keep_bottom / random_like
    must_include_first_last: bool = False,
    tau: float = 0.4,                # smaller -> closer to deterministic topk/bottomk
    eps: float = 0.0,                # 0~0.2: mix in a small amount of uniform exploration
    rng = random,                    # used only to sample L
    generator: torch.Generator | None = None,  # controls torch.multinomial randomness
) -> torch.Tensor:
    """
    Sample the same L for the whole batch, then for each sample b draw L frames
    probabilistically (without replacement) according to that sample's scores.
    Returns: idx (B,L) long, sorted ascending.
    """
    B, S = scores.shape
    if S < 2:
        raise ValueError(f"S={S} is too short")

    L = rng.randint(max(2, k_min), max(2, k_max))
    L = max(2, min(L, S))

    fixed = [0]
    if must_include_first_last and S > 1:
        fixed.append(S - 1)
    fixed = sorted(set(fixed))
    n_fixed = len(fixed)

    if L <= n_fixed:
        out = torch.tensor(fixed[:L], device=scores.device, dtype=torch.long)[None, :].expand(B, L)
        return out

    need = L - n_fixed

    # ---- prepare logits ----
    x = scores.to(torch.float32)
    # replace non-finite values
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "random_like":
        logits = torch.zeros_like(x)  # softmax -> uniform
    elif mode == "keep_top":
        logits = x
    elif mode == "keep_bottom":
        logits = -x
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # mask out fixed indices
    mask = torch.ones((B, S), device=scores.device, dtype=torch.bool)
    for fi in fixed:
        mask[:, fi] = False

    # mask out ineligible positions with -inf
    logits = logits.masked_fill(~mask, float("-inf"))

    # softmax(prob)
    tau = max(float(tau), 1e-6)
    probs = torch.softmax(logits / tau, dim=1)  # (B,S)

    # optional: mix in uniform distribution for exploration
    if eps > 0:
        eps = float(eps)
        uni = mask.to(torch.float32)
        uni = uni / uni.sum(dim=1, keepdim=True).clamp_min(1.0)
        probs = (1 - eps) * probs + eps * uni

    # numerical safety: if any row sums to 0 (edge case), fall back to uniform
    row_sum = probs.sum(dim=1, keepdim=True)
    bad = row_sum.squeeze(1) <= 0
    if bad.any():
        uni = mask.to(torch.float32)
        uni = uni / uni.sum(dim=1, keepdim=True).clamp_min(1.0)
        probs[bad] = uni[bad]

    # ---- sample without replacement ----
    picked = torch.multinomial(probs, num_samples=need, replacement=False, generator=generator)  # (B,need)

    fixed_t = torch.tensor(fixed, device=scores.device, dtype=torch.long)[None, :].expand(B, n_fixed)
    idx = torch.cat([fixed_t, picked], dim=1)  # (B,L)
    idx, _ = torch.sort(idx, dim=1)
    return idx


def sample_one_set_by_scores_batch(
    scores: torch.Tensor,            # (B,S)
    *,
    k_min: int = 2,
    k_max: int = 12,
    mode: str = "keep_top",
    must_include_first_last: bool = False,  # first frame always included; this controls whether last is also included
    rng = random,
) -> torch.Tensor:
    """
    Sample the same L for the whole batch, then for each sample b select L frames
    by ranking scores[b].
    Returns: idx (B,L) long, sorted ascending (preserves temporal order).
    """
    # assert scores.ndim == 2, f"scores must be (B,S), got {tuple(scores.shape)}"
    B, S = scores.shape
    if S < 2:
        raise ValueError(f"S={S} is too short")

    L = rng.randint(max(2, k_min), max(2, k_max))
    L = max(2, min(L, S))

    fixed = [0]
    if must_include_first_last and S > 1:
        fixed.append(S - 1)
    fixed = sorted(set(fixed))
    n_fixed = len(fixed)

    # if L is too small, return just the first L fixed indices (same for every sample)
    if L <= n_fixed:
        out = torch.tensor(fixed[:L], device=scores.device, dtype=torch.long).view(1, L).expand(B, L)
        return out

    need = L - n_fixed

    # prepare ranking scores (exclude fixed indices; handle nan/inf)
    scores_rank = scores.to(torch.float32)

    # replace non-finite values with extremes that prevent selection under current mode
    if mode == "keep_top":
        scores_rank = torch.nan_to_num(scores_rank, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
        largest = True
    elif mode == "keep_bottom":
        scores_rank = torch.nan_to_num(scores_rank, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        largest = False
    elif mode == "random_like":
        # use random scores to select
        scores_rank = torch.rand((B, S), device=scores.device, dtype=torch.float32)
        largest = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # exclude fixed indices
    scores_rank = scores_rank.clone()
    for fi in fixed:
        if mode == "keep_bottom":
            scores_rank[:, fi] = float("inf")
        else:
            scores_rank[:, fi] = float("-inf")

    # topk / bottomk
    picked = torch.topk(scores_rank, k=need, dim=1, largest=largest).indices  # (B,need)

    fixed_t = torch.tensor(fixed, device=scores.device, dtype=torch.long).view(1, n_fixed).expand(B, n_fixed)
    idx = torch.cat([fixed_t, picked], dim=1)  # (B,L)

    # sort ascending to preserve temporal order
    idx, _ = torch.sort(idx, dim=1)
    return idx


def build_seq_from_indices_batch(
    teacher_out: Dict[str, Any],
    batch_in: Dict[str, Any],
    indices: torch.Tensor,            # (B,L)
    prune_ratio: float = 0.05,
) -> Dict[str, Any]:
    """
    indices: (B,L) per-sample frame selection
    """
    device = batch_in["images"].device
    idx = indices.to(device=device, dtype=torch.long)

    out: Dict[str, Any] = {}

    # 1) gather from batch_in
    for k in ["images", "ids", "images_aug", "images_aug2"]:
        if k in batch_in and torch.is_tensor(batch_in[k]) and batch_in[k].ndim >= 2:
            out[k] = _gather_BS(batch_in[k], idx)
        elif k in batch_in:
            out[k] = batch_in[k]

    # 2) gather pose/depth/conf from teacher_out
    t_pose  = _gather_BS(teacher_out["pose_enc"],  idx)          # (B,L,9)
    t_depth = _gather_BS(teacher_out["depth"],     idx)          # (B,L,H,W,1)
    t_conf  = _gather_BS(teacher_out["depth_conf"],idx)          # (B,L,H,W)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(t_pose, out["images"].shape[-2:])
    out["extrinsics"] = extrinsic
    out["intrinsics"] = intrinsic
    out["depth_conf"] = t_conf

    depths = t_depth.squeeze(-1)                                 # (B,L,H,W)
    out["depths"] = depths
    mask_depth = torch.isfinite(depths) & (depths > 0)

    if prune_ratio > 0.0:
        B, L, H, W = t_conf.shape
        conf_flat = t_conf.reshape(B, -1)

        try:
            q = torch.nanquantile(conf_flat, prune_ratio, dim=-1, keepdim=True)
        except AttributeError:
            q = torch.quantile(torch.nan_to_num(conf_flat, nan=float("-inf")),
                               prune_ratio, dim=-1, keepdim=True)

        mask_conf_flat = conf_flat >= q
        mask_conf = mask_conf_flat.view(B, L, H, W)
        point_masks = (mask_depth & mask_conf).to(torch.bool)
    else:
        point_masks = mask_depth.to(torch.bool)

    out["point_masks"] = point_masks
    return out
