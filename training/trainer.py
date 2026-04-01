# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# Environment Variable Setup for Performance and Debugging
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import contextlib
import csv
import gc
import hashlib
import json
import logging
import math
import os.path as osp
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dataclasses import dataclass
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import *
from train_utils.optimizer import construct_optimizers
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from train_utils.rep_loss import _rep_feature_loss, _frame_rep_from_agg_tokens
from train_utils.crop_aug import (
    CropAugConfig,
    _sample_crop_boxes_per_sample,
    _crop_resize_blchw,
    _crop_resize_blhw,
    _update_intrinsics_for_crop_resize,
)
from train_utils.attn_scorer import AttnGuidedDropConfig, VGGTAttnGuidedScorer
from train_utils.frame_sampling import (
    sample_one_set_by_scores_batch_prob,
    sample_one_set_by_scores_batch,
    sample_jittered_fixed,
    sample_one_set,
    build_seq_from_indices_batch,
    _choose_strategy,
)

class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        exp_name: str = "exp",
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        distraction: bool = True,
        attn_drop_enabled: bool = True,
        keep_mode: str = "keep_top",
        rep_enabled: bool = False,
        aug:bool = True,
        drop:bool = True,
        drop_max: int = 12,
        prob_tau: float = 0.5,
        p_fixed_interval: float = 0.3,
        rep_loss_kind = "cosine",
        crop_enabled = False,
        tea_aug = False,
        prob_sample = False,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        self.exp_name = exp_name
        self.distraction = distraction
        self.rep_enabled = rep_enabled
        self.aug = aug
        self.drop = drop
        self.drop_max = drop_max
        self.rep_loss_kind = rep_loss_kind
        self.tea_aug = tea_aug
        
        self.attn_drop_enabled = attn_drop_enabled
        self.keep_mode = keep_mode
        self.prob_sample = prob_sample
        self.prob_tau = prob_tau
        self.p_fixed_interval = p_fixed_interval
        self.attn_drop_cfg = AttnGuidedDropConfig(
            layer=23,
            attn_a=0.5,   # use attention only initially
            cos_a=0.5,
            mode=keep_mode,  # also run keep_bottom / random_like for ablation
        )
        
        crop_scale_min: float = 0.6
        crop_scale_max: float = 0.9
        self.crop_cfg = CropAugConfig(
            enabled=crop_enabled,
            scale_min=crop_scale_min,
            scale_max=crop_scale_max,
            keep_aspect=True,
            resize_back=True,
            min_crop_size=32,
        )
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)
        # ema
        self.ema_model: Optional[AveragedModel] = None
        self._last_loaded_ckpt_path: Optional[str] = None

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        log_dir = os.path.join(self.logging_conf.log_dir, self.exp_name)
        setup_logging(
            __name__,
            output_dir=log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._last_loaded_ckpt_path = self.checkpoint_conf.resume_checkpoint_path
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._last_loaded_ckpt_path = ckpt_path
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        # Create EMA after DDP wrapping
        if getattr(self.optim_conf, "ema", None) and getattr(self.optim_conf.ema, "enabled", False):
            decay = float(self.optim_conf.ema.decay)
            base = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            self.ema_model = AveragedModel(base, multi_avg_fn=get_ema_multi_avg_fn(decay))
            # If resuming from a checkpoint, also try to restore EMA weights
            if self._last_loaded_ckpt_path is not None:
                try:
                    with g_pathmgr.open(self._last_loaded_ckpt_path, "rb") as f:
                        _ckpt = torch.load(f, map_location="cpu")
                    if "ema_model" in _ckpt and _ckpt["ema_model"] is not None:
                        self.ema_model.load_state_dict(_ckpt["ema_model"])
                        print("EMA state found, loading EMA weight.")
                except Exception as _:
                    logging.warning("EMA state not found or failed to load; will re-build EMA during training.")
            
            # Mean-Teacher: use the EMA model as the teacher
            self.teacher = self.ema_model
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False        
        self.attn_scorer = VGGTAttnGuidedScorer(self.teacher, self.attn_drop_cfg)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _apply_crop_aug(self, batch: Dict[str, Any], phase: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Apply crop+resize augmentation to student images/images_aug,
        and synchronously transform depths/depth_conf/point_masks/intrinsics.
        Returns crop_meta for debugging/logging.
        """
        if (phase != "train") or (not getattr(self, "crop_cfg", None)) or (not self.crop_cfg.enabled):
            return batch, None

        # determine H,W from a reference image (images_aug when available)
        ref_key = "images_aug" if ("images_aug" in batch and torch.is_tensor(batch["images_aug"])) else "images"
        if ref_key not in batch or (not torch.is_tensor(batch[ref_key])) or batch[ref_key].ndim != 5:
            return batch, None

        imgs_ref = batch[ref_key]
        B, L, C, H, W = imgs_ref.shape
        device = imgs_ref.device
        out_hw = (H, W)  # resize back to original

        box = _sample_crop_boxes_per_sample(B, H, W, self.crop_cfg, device=device)

        # --- crop images (both images & images_aug if exist) ---
        for k in ["images", "images_aug"]:
            if k in batch and torch.is_tensor(batch[k]) and batch[k].ndim == 5:
                batch[k] = _crop_resize_blchw(
                    batch[k],
                    left=box["left"], top=box["top"], crop_w=box["crop_w"], crop_h=box["crop_h"],
                    out_hw=out_hw,
                    mode=self.crop_cfg.img_mode,
                    align_corners=self.crop_cfg.align_corners,
                )

        # --- crop depth targets ---
        if "depths" in batch and torch.is_tensor(batch["depths"]) and batch["depths"].ndim == 4:
            batch["depths"] = _crop_resize_blhw(
                batch["depths"],
                left=box["left"], top=box["top"], crop_w=box["crop_w"], crop_h=box["crop_h"],
                out_hw=out_hw,
                mode=self.crop_cfg.depth_mode,
                align_corners=self.crop_cfg.align_corners,
                is_mask=False,
            )

        # --- crop depth_conf ---
        if "depth_conf" in batch and torch.is_tensor(batch["depth_conf"]) and batch["depth_conf"].ndim == 4:
            batch["depth_conf"] = _crop_resize_blhw(
                batch["depth_conf"],
                left=box["left"], top=box["top"], crop_w=box["crop_w"], crop_h=box["crop_h"],
                out_hw=out_hw,
                mode=self.crop_cfg.conf_mode,
                align_corners=self.crop_cfg.align_corners,
                is_mask=False,
            )

        # --- crop point_masks (nearest + bool) ---
        if "point_masks" in batch and torch.is_tensor(batch["point_masks"]) and batch["point_masks"].ndim == 4:
            batch["point_masks"] = _crop_resize_blhw(
                batch["point_masks"],
                left=box["left"], top=box["top"], crop_w=box["crop_w"], crop_h=box["crop_h"],
                out_hw=out_hw,
                mode=self.crop_cfg.mask_mode,
                align_corners=False,
                is_mask=True,
            )

        # --- update intrinsics K -> K' ---
        if "intrinsics" in batch and torch.is_tensor(batch["intrinsics"]) and batch["intrinsics"].ndim == 4:
            batch["intrinsics"] = _update_intrinsics_for_crop_resize(
                batch["intrinsics"],
                left=box["left"], top=box["top"],
                crop_w=box["crop_w"], crop_h=box["crop_h"],
                out_hw=out_hw,
            )

        crop_meta = {
            "left": box["left"],
            "top": box["top"],
            "crop_w": box["crop_w"],
            "crop_h": box["crop_h"],
            "do_crop": box["do_crop"],
        }
        return batch, crop_meta

    def _log_frame_selection_txt(
        self,
        *,
        phase: str,
        chunked_batch: Dict[str, Any],
        scores: torch.Tensor,     # (B,S) float
        indices: torch.Tensor,    # (B,L) long
        mode: str,
    ):
        """
        Write jsonl to frame_selection_debug.txt.
        Called only on rank0 when step % log_freq == 0.
        """
        if self.rank != 0:
            return

        step = int(self.steps.get(phase, 0))
        log_freq = int(getattr(self.logging_conf, "log_freq", 50))
        if log_freq > 0 and (step % log_freq != 0):
            return

        log_dir = os.path.join(self.logging_conf.log_dir, self.exp_name)
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "frame_selection_debug.txt")  # jsonl

        seq_names = chunked_batch.get("seq_name", None)
        ids = chunked_batch.get("ids", None)  # (B,S) ideally

        B, S = scores.shape
        k = indices.shape[1]

        # student ids (if available)
        stu_ids = None
        if torch.is_tensor(ids) and ids.ndim == 2 and ids.shape[0] == B and ids.shape[1] == S:
            stu_ids = _gather_BS(ids, indices)  # (B,L)

        # write one line per sample
        with open(path, "a", encoding="utf-8") as f:
            for b in range(B):
                name_b = seq_names[b] if isinstance(seq_names, (list, tuple)) else str(seq_names)

                # top/bottom frames by score
                srow = scores[b].detach().float()

                top_idx = torch.topk(srow, k=k, largest=True).indices.tolist()
                bot_idx = torch.topk(srow, k=k, largest=False).indices.tolist()

                top_pairs = [(int(i), float(srow[int(i)].item())) for i in top_idx]
                bot_pairs = [(int(i), float(srow[int(i)].item())) for i in bot_idx]

                # map to ids if possible
                teacher_ids_b = None
                student_ids_b = None
                top_ids = None
                bot_ids = None
                if torch.is_tensor(ids) and ids.ndim == 2 and ids.shape[0] == B and ids.shape[1] == S:
                    teacher_ids_b = [int(x) for x in ids[b].detach().cpu().tolist()]
                    if stu_ids is not None:
                        student_ids_b = [int(x) for x in stu_ids[b].detach().cpu().tolist()]
                    top_ids = [int(ids[b, int(i)].item()) for i in top_idx]
                    bot_ids = [int(ids[b, int(i)].item()) for i in bot_idx]

                row = {
                    "step": step,
                    "mode": mode,
                    "seq_name": name_b,
                    "teacher_ids": teacher_ids_b,
                    "student_ids": student_ids_b,
                    "teacher_frame_idx": list(range(S)),
                    "student_frame_idx": [int(x) for x in indices[b].detach().cpu().tolist()],
                    "topk_idx_score": top_pairs,
                    "bottomk_idx_score": bot_pairs,
                    "topk_ids": top_ids,
                    "bottomk_ids": bot_ids,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _tb_log_global_grad_norm(
        self,
        *,
        phase: str,
        model: nn.Module,
        tag: str = "Grad/global_norm",
        step: Optional[int] = None,
    ):
        """
        Compute the DDP global grad L2 norm (sqrt(sum_all_ranks(sum(g^2)))) and log to TB.
        IMPORTANT: collective calls must be made on every rank; only rank0 writes to TB.
        Call after scaler.unscale_() so gradients reflect their true values.
        """
        if step is None:
            step = int(self.steps.get(phase, 0))

        # To reduce log frequency (e.g. every 10 steps), this condition must be consistent across ranks:
        # do_log = (step % 10 == 0)
        # if not do_log: return
        # (do NOT evaluate do_log inside a rank0-only branch)

        device = self.device
        sq = torch.zeros((), device=device, dtype=torch.float32)
        nonfinite = torch.zeros((), device=device, dtype=torch.float32)
        total = torch.zeros((), device=device, dtype=torch.float32)

        for p in model.parameters():
            g = p.grad
            if g is None:
                continue
            g = g.detach()

            total += float(g.numel())

            finite = torch.isfinite(g)
            nonfinite += (~finite).sum().to(torch.float32)

            # only include finite grads in the sum-of-squares to avoid nan/inf contamination
            g32 = g.float()
            g32 = torch.where(finite, g32, torch.zeros_like(g32))
            sq += (g32 * g32).sum()

        # DDP all-reduce: all ranks must participate
        if dist.is_available() and dist.is_initialized():
            dist.reduce(sq, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(nonfinite, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)

        # only rank0 writes to TB
        if self.rank == 0:
            gn = torch.sqrt(sq).item()
            nf = (nonfinite / total.clamp(min=1)).item()
            self.tb_writer.log(tag, float(gn), int(step))
            self.tb_writer.log("Grad/nonfinite_frac", float(nf), int(step))

    def _tb_log_teacher_conf_pixel_stats(
        self,
        *,
        phase: str,
        teacher_conf: Optional[torch.Tensor],   # (B,L,H,W) or (B,S,H,W) or (...,1)
        tag_prefix: str = "Conf/teacher",
        step: Optional[int] = None,
        log_quantiles: bool = False,            # quantiles are exact but slightly slower; disable if needed
    ):
        """
        Compute statistics over all pixels directly (deterministic and exact, no random sampling).
        """
        if self.rank != 0:
            return
        if teacher_conf is None or (not torch.is_tensor(teacher_conf)):
            return

        if step is None:
            step = int(self.steps.get(phase, 0))

        x = teacher_conf.detach()
        if x.ndim >= 1 and x.shape[-1] == 1:
            x = x[..., 0]
        x = x.float().reshape(-1)  # all pixels

        # non-finite
        finite = torch.isfinite(x)
        nonfinite_frac = 1.0 - finite.float().mean().item()

        x = x[finite]
        if x.numel() == 0:
            self.tb_writer.log(f"{tag_prefix}/nonfinite_frac", float(nonfinite_frac), int(step))
            return

        mean = x.mean().item()
        std  = x.std(unbiased=False).item()
        mn   = x.min().item()
        mx   = x.max().item()

        # cheap but exact distribution-shape indicators
        frac_lt_0p1 = (x < 0.1).float().mean().item()
        frac_gt_0p9 = (x > 0.9).float().mean().item()
        frac_lt_0p01 = (x < 0.01).float().mean().item()
        frac_gt_0p99 = (x > 0.99).float().mean().item()

        # TB log
        self.tb_writer.log(f"{tag_prefix}/mean", float(mean), int(step))
        self.tb_writer.log(f"{tag_prefix}/std", float(std), int(step))
        self.tb_writer.log(f"{tag_prefix}/min", float(mn), int(step))
        self.tb_writer.log(f"{tag_prefix}/max", float(mx), int(step))
        self.tb_writer.log(f"{tag_prefix}/nonfinite_frac", float(nonfinite_frac), int(step))

        self.tb_writer.log(f"{tag_prefix}/frac_lt_0p1", float(frac_lt_0p1), int(step))
        self.tb_writer.log(f"{tag_prefix}/frac_gt_0p9", float(frac_gt_0p9), int(step))
        self.tb_writer.log(f"{tag_prefix}/frac_lt_0p01", float(frac_lt_0p01), int(step))
        self.tb_writer.log(f"{tag_prefix}/frac_gt_0p99", float(frac_gt_0p99), int(step))

        # optional: exact quantiles (deterministic but slightly slower)
        if log_quantiles:
            qs = torch.quantile(x, torch.tensor([0.05, 0.50, 0.95], device=x.device))
            q05, q50, q95 = [t.item() for t in qs]
            self.tb_writer.log(f"{tag_prefix}/q05", float(q05), int(step))
            self.tb_writer.log(f"{tag_prefix}/q50", float(q50), int(step))
            self.tb_writer.log(f"{tag_prefix}/q95", float(q95), int(step))

    def _tb_log_depth_conf_stats(
        self,
        *,
        phase: str,
        teacher_depth_conf: Optional[torch.Tensor],  # (B,S,H,W) or (B,S,...) float
        indices: torch.Tensor,                       # (B,L)
        scores: torch.Tensor,                        # (B,S)  used to define top/bottom
        mode: str,
        k: int = 5,
    ):
        """
        TB logging (ranked by score):
        1) all frames mean depth_conf
        2) student selected mean depth_conf
        3) top-k(score) frames mean depth_conf
        4) bottom-k(score) frames mean depth_conf
        """
        if self.rank != 0:
            return
        if teacher_depth_conf is None or (not torch.is_tensor(teacher_depth_conf)):
            return

        step = int(self.steps.get(phase, 0))
        log_freq = int(getattr(self.logging_conf, "log_freq", 50))
        if log_freq > 0 and (step % log_freq != 0):
            return

        conf = teacher_depth_conf.detach().float()
        if conf.ndim < 2:
            return

        B, S = conf.shape[:2]
        if S < 1:
            return

        # ---- reduce conf to per-frame scalar: (B,S) ----
        conf_frame = conf.reshape(B, S, -1)               # (B,S,HW...)
        conf_frame_mean = torch.nanmean(conf_frame, dim=-1)  # (B,S)

        # ---- all frames mean ----
        conf_all_mean = torch.nanmean(conf_frame_mean)  # scalar

        # ---- student selected mean ----
        try:
            conf_sel = _gather_BS(conf_frame_mean, indices)      # (B,L)
            conf_sel_mean = torch.nanmean(conf_sel)              # scalar
        except Exception:
            return

        # ---- top/bottom-k(score) mean ----
        sc = scores.detach().float()
        if sc.shape[0] != B or sc.shape[1] != S:
            # shape mismatch: skip top/bottom logging
            top_mean = torch.tensor(float("nan"), device=conf.device)
            bot_mean = torch.tensor(float("nan"), device=conf.device)
            k_eff = min(k, S)
        else:
            k_eff = int(min(max(1, k), S))
            top_idx = torch.topk(sc, k=k_eff, dim=1, largest=True).indices    # (B,k)
            bot_idx = torch.topk(sc, k=k_eff, dim=1, largest=False).indices   # (B,k)

            top_conf = _gather_BS(conf_frame_mean, top_idx)   # (B,k)
            bot_conf = _gather_BS(conf_frame_mean, bot_idx)   # (B,k)
            top_mean = torch.nanmean(top_conf)
            bot_mean = torch.nanmean(bot_conf)

        def _to_float(x: torch.Tensor) -> float:
            return float(x.item()) if torch.is_tensor(x) and torch.isfinite(x) else float("nan")

        all_f = _to_float(conf_all_mean)
        sel_f = _to_float(conf_sel_mean)
        top_f = _to_float(top_mean)
        bot_f = _to_float(bot_mean)

        # ---- TB logging ----
        self.tb_writer.log("Conf/teacher_depth_conf_allframes_mean", all_f, step)
        self.tb_writer.log(f"Conf/teacher_depth_conf_selected_mean", sel_f, step)
        self.tb_writer.log(f"Conf/teacher_depth_conf_top{k_eff}_by_score_mean", top_f, step)
        self.tb_writer.log(f"Conf/teacher_depth_conf_bottom{k_eff}_by_score_mean", bot_f, step)

        # ratios (more intuitive for monitoring)
        denom = all_f if (all_f == all_f and abs(all_f) > 1e-12) else 1e-12
        self.tb_writer.log(f"Conf/ratio_selected_over_all", sel_f / denom, step)
        self.tb_writer.log(f"Conf/ratio_top{k_eff}_over_all_by_score", top_f / denom, step)
        self.tb_writer.log(f"Conf/ratio_bottom{k_eff}_over_all_by_score", bot_f / denom, step)

        # gap (optional but useful for monitoring selection quality)
        if top_f == top_f and bot_f == bot_f:
            self.tb_writer.log(f"Conf/gap_top{k_eff}_minus_bottom{k_eff}_by_score", top_f - bot_f, step)
        if sel_f == sel_f and bot_f == bot_f:
            self.tb_writer.log(f"Conf/gap_selected_minus_bottom{k_eff}_by_score", sel_f - bot_f, step)

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            self.optims[0].optimizer.load_state_dict(checkpoint["optimizer"])

        # Load training progress
        if "prev_epoch" in checkpoint:
            self.epoch = checkpoint["prev_epoch"] + 1
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        # self.teacher = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # ===== Rep-matching (BYOL-style) minimal add-on =====
        self.rep_layers = [4, 11, 17, 23]         # ✅ DPT multi-scale layers
        self.rep_loss_w = 0.01                    # suggested sweep range: 0.01~0.1
        # self.rep_loss_kind = "cosine_smoothl1"  # stronger constraint; use if needed
        self.rep_smoothl1_beta = 0.05
        self.rep_mix_alpha = 0.5                  # mixing coefficient for cosine_smoothl1
        self.rep_global_feat_start = 0


        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            sum_dir = os.path.join(self.logging_conf.log_dir, self.exp_name)
            model_summary_path = os.path.join(sum_dir, "model.txt")
            os.makedirs(sum_dir, exist_ok=True)
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def _safe_key(self, s: str) -> str:
        return "".join([c if c.isalnum() or c in "_-." else "_" for c in str(s)])[:180]

    def _hash_ids(self, ids_1d: torch.Tensor) -> str:
        x = ids_1d.detach().to("cpu").to(torch.int64).numpy().tobytes()
        return hashlib.md5(x).hexdigest()[:10]

    def _squeeze_depth(self, d: torch.Tensor) -> torch.Tensor:
        if d is None:
            return None
        if d.ndim == 5 and d.shape[-1] == 1:
            d = d[..., 0]
        return d

    def _make_point_mask_from_conf(self, depth: torch.Tensor, depth_conf: torch.Tensor | None, prune_ratio: float):
        """
        depth: (B,L,H,W) float
        depth_conf: (B,L,H,W) or None
        return point_masks: (B,L,H,W) bool
        """
        mask_depth = torch.isfinite(depth) & (depth > 0)
        if depth_conf is None or (not torch.is_tensor(depth_conf)) or prune_ratio <= 0:
            return mask_depth

        B, L, H, W = depth_conf.shape
        conf_flat = depth_conf.reshape(B, -1)

        try:
            q = torch.nanquantile(conf_flat, prune_ratio, dim=-1, keepdim=True)
        except Exception:
            q = torch.quantile(torch.nan_to_num(conf_flat, nan=float("-inf")),
                            prune_ratio, dim=-1, keepdim=True)

        mask_conf = (conf_flat >= q).view(B, L, H, W)
        return (mask_depth & mask_conf).bool()

    def _geom_pack(self, out_dict: dict, images: torch.Tensor, prune_ratio: float = 0.05):
        """
        Build extrinsics/intrinsics/depths/point_masks from teacher outputs (pose_enc/depth/depth_conf)
        and normalize them consistently with training.
        Returns: (E, K, D, point_masks) or (None, None, None, None)
        """
        if out_dict is None:
            return None, None, None, None
        pose = out_dict.get("pose_enc", None)
        depth = out_dict.get("depth", None)
        conf  = out_dict.get("depth_conf", None)

        if (pose is None) or (depth is None):
            return None, None, None, None

        D = self._squeeze_depth(depth)  # (B,L,H,W)
        B, L, H, W = D.shape

        # pose_enc -> extr/intri (consistent with build_seq)
        E, K = pose_encoding_to_extri_intri(pose, images.shape[-2:])  # (B,L,3,4), (B,L,3,3)
        # E = self._w2c_to_4x4(E_w2c)
        
        point_masks = self._make_point_mask_from_conf(D, conf, prune_ratio=prune_ratio)

        # normalize (consistent with training)
        with torch.no_grad():
            cam_pts, world_pts, _ = lift_depth_to_cam_world_points_torch(
                depths=D, extrinsics=E, intrinsics=K
            )
            new_E, _, _, new_D = normalize_camera_extrinsics_and_points_batch_gpu(
                extrinsics=E,
                cam_points=cam_pts,
                world_points=world_pts,
                depths=D,
                point_masks=point_masks,
                scale_by_points=True,
            )
        return new_E, K, new_D, point_masks

    def _w2c_to_4x4(self, E: torch.Tensor) -> torch.Tensor:
        """
        E: (...,3,4) or (...,4,4) world->cam
        return: (...,4,4) world->cam
        """
        if E.shape[-2:] == (4, 4):
            return E
        if E.shape[-2:] == (3, 4):
            out = torch.zeros((*E.shape[:-2], 4, 4), device=E.device, dtype=E.dtype)
            out[..., :3, :4] = E
            out[..., 3, 3] = 1.0
            return out
        raise RuntimeError(f"Unsupported E shape: {tuple(E.shape)}")

    def _invert_rigid_w2c_4x4(self, Tw2c: torch.Tensor) -> torch.Tensor:
        """
        Tw2c: (...,4,4) world->cam
        return: (...,4,4) cam->world (rigid inverse)
        """
        R = Tw2c[..., :3, :3]
        t = Tw2c[..., :3, 3:4]
        Rt = R.transpose(-1, -2)
        t_inv = -Rt @ t
        out = torch.zeros_like(Tw2c)
        out[..., :3, :3] = Rt
        out[..., :3, 3:4] = t_inv
        out[..., 3, 3] = 1.0
        return out    

    @torch.no_grad()
    def _build_refmask_anchor0_from_ref_geom(
        self,
        *,
        D: torch.Tensor,            # (B,L,H,W) float, already normalized
        K: torch.Tensor,            # (B,L,3,3)
        E: torch.Tensor,            # (B,L,4,4) w2c (world->cam)  ✅
        point_masks: torch.Tensor,  # (B,L,H,W) bool  (from conf+depth_valid)
        anchor: int = 0,
        occ_rel: float = 0.10,
        outlier_q: float = 0.95,
        dist_cap: float | None = None,
        chunk: int = 65536,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Build anchor-space reference mask.
        mask_ref[:, j-1, v, u] indicates anchor pixel (u,v) is usable when projected to frame j.
        Returns: (B, L-1, H, W) bool
        """
        assert D.ndim == 4
        B, L, H, W = D.shape
        assert L >= 2
        assert E.shape[-2:] == (4, 4), f"E must be 4x4 w2c, got {E.shape}"

        device = D.device

        # E is w2c, get c2w by rigid inverse (fast & stable)
        Tw2c = E
        Tc2w = self._invert_rigid_w2c_4x4(Tw2c)  # (B,L,4,4) cam->world

        # anchor base mask: fixed evaluation point set (from reference ckpt only)
        base0 = point_masks[:, anchor].bool() & torch.isfinite(D[:, anchor]) & (D[:, anchor] > 0)

        # full-resolution pixel grid
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        u_all = gx.reshape(-1).float()  # (HW,)
        v_all = gy.reshape(-1).float()

        out = torch.zeros((B, L - 1, H, W), dtype=torch.bool, device=device)

        for j in range(L):
            if j == anchor:
                continue

            out_j = torch.zeros((B, H * W), dtype=torch.bool, device=device)
            base_flat = base0.reshape(B, -1)  # (B,HW)

            for b in range(B):
                idx = torch.nonzero(base_flat[b], as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue

                K0 = K[b, anchor]
                Kj = K[b, j]

                # anchor cam -> world
                Tc2w0 = Tc2w[b, anchor]     # cam0->world
                # world -> cam_j
                Tw2cj = Tw2c[b, j]          # world->cam_j

                fx0 = K0[0, 0]; fy0 = K0[1, 1]; cx0 = K0[0, 2]; cy0 = K0[1, 2]
                fxj = Kj[0, 0]; fyj = Kj[1, 1]; cxj = Kj[0, 2]; cyj = Kj[1, 2]

                D0 = D[b, anchor].reshape(-1)          # (HW,)
                Dj = D[b, j].unsqueeze(0).unsqueeze(0) # (1,1,H,W)

                keep_mask = torch.zeros((idx.numel(),), dtype=torch.bool, device=device)
                dist_all  = torch.full((idx.numel(),), float("inf"), device=device)

                for s in range(0, idx.numel(), chunk):
                    sl = idx[s:s + chunk]
                    u = u_all[sl]
                    v = v_all[sl]
                    z0 = D0[sl].clamp_min(eps)

                    # cam0 point
                    x0 = (u - cx0) / fx0 * z0
                    y0 = (v - cy0) / fy0 * z0
                    cam0 = torch.stack([x0, y0, z0, torch.ones_like(z0)], dim=-1)  # (P,4)

                    # cam0 -> world -> camj
                    world = (Tc2w0 @ cam0.t()).t()      # (P,4)
                    camj  = (Tw2cj @ world.t()).t()     # (P,4)
                    xj_c, yj_c, zj_raw = camj[:, 0], camj[:, 1], camj[:, 2]
                    valid_z = torch.isfinite(zj_raw) & (zj_raw > eps)
                    zj = zj_raw.clamp_min(eps)

                    # project to frame j pixel
                    uj = fxj * (xj_c / zj) + cxj
                    vj = fyj * (yj_c / zj) + cyj
                    nx = (uj / (W - 1)) * 2 - 1
                    ny = (vj / (H - 1)) * 2 - 1
                    inside = (nx > -1) & (nx < 1) & (ny > -1) & (ny < 1)

                    grid = torch.stack([nx, ny], dim=-1).view(1, 1, -1, 2)
                    z_obs = F.grid_sample(Dj, grid, mode="bilinear", align_corners=True).view(-1)

                    valid = inside & valid_z & torch.isfinite(z_obs) & (z_obs > eps)

                    # occlusion gate: relative depth agreement in cam_j
                    rel = (z_obs - zj).abs() / z_obs.abs().clamp_min(eps)
                    valid = valid & (rel < occ_rel)

                    if valid.any():
                        # backproject observed depth to cam_j, compare 3D distance in cam_j
                        x_obs = (uj - cxj) / fxj * z_obs
                        y_obs = (vj - cyj) / fyj * z_obs
                        dist = torch.sqrt((x_obs - xj_c) ** 2 + (y_obs - yj_c) ** 2 + (z_obs - zj) ** 2)

                        dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
                        dist_all[s:s + sl.numel()] = dist
                        keep_mask[s:s + sl.numel()] = valid

                # outlier removal (ONLY for reference ckpt mask construction)
                inlier_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
                if inlier_idx.numel() > 0:
                    d_valid = dist_all[inlier_idx]
                    thr = torch.quantile(d_valid, outlier_q)
                    if dist_cap is not None:
                        thr = torch.minimum(thr, torch.tensor(dist_cap, device=device, dtype=d_valid.dtype))
                    good = keep_mask & (dist_all <= thr)
                else:
                    good = keep_mask

                out_j_b = torch.zeros((H * W,), dtype=torch.bool, device=device)
                out_j_b[idx] = good
                out_j[b] = out_j_b

            out[:, j - 1] = out_j.view(B, H, W)

        return out

    def _frame_mean_absrel(
        self,
        a: torch.Tensor,                 # (B,L,H,W)
        b: torch.Tensor,                 # (B,L,H,W)
        mask: torch.Tensor | None = None,# (B,L,H,W) bool or None
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute per-frame AbsRel: mean(|a-b| / (|b|+eps)) over valid pixels.
        Returns: (B,L)

        Valid pixels:
        - finite(a) & finite(b)
        - |b| > eps
        - (optional) mask == True
        If a frame has 0 valid pixels, output NaN for that (B,L).
        """
        assert a.ndim == 4 and b.ndim == 4, f"expect (B,L,H,W), got {a.shape}, {b.shape}"
        a = a.float()
        b = b.float()

        valid = torch.isfinite(a) & torch.isfinite(b) & (b.abs() > eps)
        if mask is not None:
            valid = valid & mask.bool()

        rel = (a - b).abs() / b.abs().clamp_min(eps)

        # sum over pixels
        rel_sum = torch.where(valid, rel, torch.zeros_like(rel)).flatten(2).sum(-1)   # (B,L)
        denom   = valid.flatten(2).sum(-1)                                            # (B,L)

        out = rel_sum / denom.clamp_min(1.0)
        # frames with denom==0 => NaN (honest: avoids treating as 0)
        out = torch.where(denom > 0, out, torch.full_like(out, float("nan")))
        return out
    @torch.no_grad()
    def _reproj_3d_distance_anchor_eval(
        self,
        *,
        D: torch.Tensor,           # (1,L,H,W)
        K: torch.Tensor,           # (1,L,3,3)
        E: torch.Tensor,           # (1,L,4,4) w2c (world->cam) ✅
        ref_mask: torch.Tensor,    # (L-1,H,W) bool (anchor pixel space)
        anchor: int = 0,
        chunk: int = 65536,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate 3D distance error on a fixed reference mask.
        Returns:
        dist_mean:  (1, L-1)
        valid_frac: (1, L-1)  fraction of ref points that remain valid under current ckpt
        """
        assert D.shape[0] == 1, "eval helper assumes B=1"
        _, L, H, W = D.shape
        device = D.device
        if E.shape[-2:] != (4, 4):
            E = self._w2c_to_4x4(E)

        Tw2c = E
        Tc2w = self._invert_rigid_w2c_4x4(Tw2c)

        # full-resolution pixel grid
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        u_all = gx.reshape(-1).float()
        v_all = gy.reshape(-1).float()

        # anchor intrinsics / transforms
        K0 = K[0, anchor]
        Tc2w0 = Tc2w[0, anchor]          # cam0->world
        fx0 = K0[0, 0]; fy0 = K0[1, 1]; cx0 = K0[0, 2]; cy0 = K0[1, 2]
        D0 = D[0, anchor].reshape(-1)

        out_dist = []
        out_keep = []

        for j in range(L):
            if j == anchor:
                continue

            # ref points in anchor pixel space
            m = ref_mask[j - 1].reshape(-1).bool()
            idx = torch.nonzero(m, as_tuple=False).squeeze(1)
            denom = float(idx.numel())
            if idx.numel() == 0:
                out_dist.append(torch.tensor(float("nan"), device=device))
                out_keep.append(torch.tensor(float("nan"), device=device))
                continue

            Kj = K[0, j]
            Tw2cj = Tw2c[0, j]            # world->cam_j
            fxj = Kj[0, 0]; fyj = Kj[1, 1]; cxj = Kj[0, 2]; cyj = Kj[1, 2]
            Dj = D[0, j].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            sum_dist = torch.zeros((), device=device)
            cnt = torch.zeros((), device=device)

            for s in range(0, idx.numel(), chunk):
                sl = idx[s:s + chunk]
                u = u_all[sl]; v = v_all[sl]
                z0 = D0[sl].clamp_min(eps)

                # cam0 point
                x0 = (u - cx0) / fx0 * z0
                y0 = (v - cy0) / fy0 * z0
                cam0 = torch.stack([x0, y0, z0, torch.ones_like(z0)], dim=-1)  # (P,4)

                # cam0 -> world -> camj
                world = (Tc2w0 @ cam0.t()).t()      # (P,4)
                camj  = (Tw2cj @ world.t()).t()     # (P,4)
                xj_c, yj_c, zj_raw = camj[:, 0], camj[:, 1], camj[:, 2]
                valid_z = torch.isfinite(zj_raw) & (zj_raw > eps)
                zj = zj_raw.clamp_min(eps)

                # project to frame j pixel
                uj = fxj * (xj_c / zj) + cxj
                vj = fyj * (yj_c / zj) + cyj
                nx = (uj / (W - 1)) * 2 - 1
                ny = (vj / (H - 1)) * 2 - 1
                inside = (nx > -1) & (nx < 1) & (ny > -1) & (ny < 1)

                grid = torch.stack([nx, ny], dim=-1).view(1, 1, -1, 2)
                z_obs = F.grid_sample(Dj, grid, mode="bilinear", align_corners=True).view(-1)

                valid = inside & valid_z & torch.isfinite(z_obs) & (z_obs > eps)
                if valid.any():
                    x_obs = (uj - cxj) / fxj * z_obs
                    y_obs = (vj - cyj) / fyj * z_obs
                    dist = torch.sqrt((x_obs - xj_c) ** 2 + (y_obs - yj_c) ** 2 + (z_obs - zj) ** 2)

                    dist = dist[valid]
                    sum_dist += dist.sum()
                    cnt += dist.numel()

            mean = sum_dist / cnt.clamp_min(1.0)
            keep = cnt / max(1.0, denom)

            out_dist.append(mean)
            out_keep.append(keep)

        dist_mean = torch.stack(out_dist, dim=0).view(1, L - 1)
        valid_frac = torch.stack(out_keep, dim=0).view(1, L - 1)
        return dist_mean, valid_frac

    def _get_val_refmask_dir(self) -> str:
        """
        Use data.val.save_mask_dir if set; otherwise default to log_dir/exp_name.
        """
        val_conf = None
        if isinstance(self.data_conf, dict):
            val_conf = self.data_conf.get("val", None)
        else:
            val_conf = getattr(self.data_conf, "val", None)

        d = None
        if val_conf is not None:
            d = getattr(val_conf, "save_mask_dir", None)

        if d is None or str(d).strip() == "":
            d = osp.join(self.logging_conf.log_dir, self.exp_name, "val_refmask_dir")
        return str(d)

    def _refmask_path(self, ref_dir: str, seq_name: str, ids_1d: torch.Tensor) -> str:
        """
        One mask file per sample (= one sequence + one frame set).
        """
        key_ids = self._hash_ids(ids_1d)
        key = f"{self._safe_key(seq_name)}_{key_ids}"
        return osp.join(ref_dir, f"{key}.npz")

    def _save_refmask_npz(self, path: str, mask_ref: torch.Tensor, ids_1d: torch.Tensor, anchor: int = 0):
        """
        mask_ref: (L-1, H, W) bool on CPU or GPU.
        Stored with packbits compression for compact files.
        """
        m = mask_ref.detach().to("cpu")
        assert m.ndim == 3, f"mask_ref must be (L-1,H,W), got {m.shape}"
        Lm1, H, W = m.shape
        m_np = m.to(torch.uint8).numpy().reshape(Lm1, -1)  # (Lm1, HW)
        bits = np.packbits(m_np, axis=1)                   # (Lm1, HW/8)

        meta = {
            "H": int(H),
            "W": int(W),
            "Lm1": int(Lm1),
            "anchor": int(anchor),
            "ids": [int(x) for x in ids_1d.detach().to("cpu").tolist()],
        }
        os.makedirs(osp.dirname(path), exist_ok=True)
        np.savez_compressed(path, mask_bits=bits, meta=json.dumps(meta))

    def _load_refmask_npz(self, path: str, device: torch.device) -> tuple[torch.Tensor, dict]:
        """
        return mask_ref: (L-1,H,W) bool on device, meta dict
        """
        z = np.load(path, allow_pickle=False)
        bits = z["mask_bits"]
        meta = json.loads(str(z["meta"].item() if hasattr(z["meta"], "item") else z["meta"]))

        H = int(meta["H"]); W = int(meta["W"]); Lm1 = int(meta["Lm1"])
        # unpack -> (Lm1, HW)
        flat = np.unpackbits(bits, axis=1)[:, :H * W].reshape(Lm1, H, W).astype(np.bool_)
        m = torch.from_numpy(flat).to(device=device)
        return m, meta

    def _get_or_build_refmask(
        self,
        *,
        ref_dir: str,
        is_ref_run: bool,
        seq_name: str,
        ids_1d: torch.Tensor,                 # (L,)
        out_ref_b: dict,                      # output dict for a single sample (B=1)
        images_b: torch.Tensor,               # (1,L,C,H,W)
        anchor: int = 0,
        prune_ratio: float = 0.05,
        occ_rel: float = 0.10,
        outlier_q: float = 0.95,
        dist_cap: float | None = None,
    ) -> torch.Tensor:
        """
        - If ref_dir does not exist => is_ref_run=True => force-build and save.
        - Otherwise: load if file exists; build and save if it does not.
        """
        path = self._refmask_path(ref_dir, seq_name, ids_1d)

        if (not is_ref_run) and osp.isfile(path):
            m, meta = self._load_refmask_npz(path, device=self.device)
            return m

        # ---- build from current model outputs (reference) ----
        # use _geom_pack to ensure E/K/D are consistent with training (including normalization)
        E, K, D, point_masks = self._geom_pack(out_ref_b, images_b, prune_ratio=prune_ratio)
        E = self._w2c_to_4x4(E)

        # build anchor-space reference mask: (L-1,H,W) bool
        m = self._build_refmask_anchor0_from_ref_geom(
            D=D, K=K, E=E, point_masks=point_masks,
            anchor=anchor, occ_rel=occ_rel, outlier_q=outlier_q, dist_cap=dist_cap
        )  # (1,L-1,H,W)
        if m.ndim == 4:
            m = m[0]
        self._save_refmask_npz(path, m, ids_1d, anchor=anchor)
        return m
    
    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        checkpoint_content["ema_model"] = (self.ema_model.state_dict() if self.ema_model is not None else None)
        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            # self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            if self.distributed_rank == 0:

                self.save_checkpoint(self.epoch)

            dist.barrier()
            
            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            # if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
            #     self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if self.rank != 0:
            return  # ✅ only rank0 runs validation

        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        """
        Offline-friendly validation:
        - Geo 3D distance (teacher/student) on fixed reference masks
        - Teacher-student depth AbsRel consistency
        - Logs BOTH:
            (1) per-batch CSV (one row per batch)
            (2) per-checkpoint summary CSV (one row per ckpt)

        Notes:
        - `self._geom_pack` must return E as (B,L,3,4) w2c, K as (B,L,3,3), D as (B,L,H,W)
        - `self._get_or_build_refmask` should be strict: only ref run builds masks; others only load
        - `self._reproj_3d_distance_anchor_eval` should accept E w2c (3x4 or 4x4) and use ref_mask
        - `_frame_mean_absrel` must exist
        - This function does NOT rely on TensorBoard step for offline sweeps; it writes CSVs keyed by ckpt name.
        """
        assert self.rank == 0, "val_epoch should be called on rank0 only."
        # -------------------------
        # helpers (local)
        # -------------------------
        def _append_csv_row(path: str, row: dict):
            os.makedirs(osp.dirname(path), exist_ok=True)
            exists = osp.isfile(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    w.writeheader()
                w.writerow(row)

        def _cat(xs):
            if len(xs) == 0:
                return None
            return torch.cat(xs, dim=0)

        def _stats(x: torch.Tensor | None):
            if x is None or (not torch.is_tensor(x)) or x.numel() == 0:
                return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
            x = x.detach().float().reshape(-1)
            x = x[torch.isfinite(x)]
            if x.numel() == 0:
                return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
            return {
                "mean": float(x.mean().item()),
                "p50": float(torch.quantile(x, 0.50).item()),
                "p90": float(torch.quantile(x, 0.90).item()),
            }

        # -------------------------
        # setup
        # -------------------------
        self.model.eval()
        if getattr(self, "teacher", None) is not None:
            self.teacher.eval()

        out_dir = osp.join(self.logging_conf.log_dir, self.exp_name)
        os.makedirs(out_dir, exist_ok=True)

        # checkpoint tag for offline sweeps
        ckpt_tag = "unknown"
        p = getattr(self.checkpoint_conf, "resume_checkpoint_path", None)
        if p:
            ckpt_tag = osp.basename(p).replace(".pt", "")

        # CSV outputs
        summary_csv = osp.join(out_dir, "val_ckpt_summary.csv")
        batch_csv   = osp.join(out_dir, "val_batches.csv")

        # reference mask dir
        ref_dir = self._get_val_refmask_dir()
        is_ref_run = (not osp.isdir(ref_dir))  # directory absent => reference ckpt
        if is_ref_run:
            os.makedirs(ref_dir, exist_ok=True)

        # mask-building hyperparams (used ONLY if is_ref_run and mask missing)
        prune_ratio = 0.10
        occ_rel     = 0.10
        outlier_q   = 0.85
        dist_cap    = None

        # collect distributions across the whole val epoch
        geo_teacher = []
        geo_teacher_keep = []
        geo_student = []
        geo_student_keep = []
        ts_frames = []  # IMPORTANT: keep across batches

        limit_val_batches = len(val_loader) if self.limit_val_batches is None else self.limit_val_batches

        num_batches_ran = 0

        # -------------------------
        # main loop
        # -------------------------
        for it, batch in enumerate(val_loader):
            if it > limit_val_batches:
                break
            num_batches_ran += 1

            # record list lengths to compute batch-local means for geo
            geo_teacher_len0 = len(geo_teacher)
            geo_teacher_keep_len0 = len(geo_teacher_keep)
            geo_student_len0 = len(geo_student)
            geo_student_keep_len0 = len(geo_student_keep)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            images = batch["images"]                 # (B,L,C,H,W)
            ids    = batch.get("ids", None)          # (B,L)
            seqs   = batch.get("seq_name", None)     # list[str] or str

            B = images.shape[0]
            L = images.shape[1]

            if isinstance(seqs, str):
                seqs = [seqs] * B

            # forward
            teacher_out = self.teacher(images=images) if getattr(self, "teacher", None) is not None else None
            student_out = self.model(images=images)

            # -------------------------
            # (A) teacher-student depth consistency (AbsRel)
            # -------------------------
            ts = None
            tea_D = self._squeeze_depth(teacher_out.get("depth", None)) if teacher_out is not None else None
            stu_D = self._squeeze_depth(student_out.get("depth", None))

            mask = batch.get("point_masks", None)
            if torch.is_tensor(mask):
                mask = mask.bool()

            if (tea_D is not None) and (stu_D is not None):
                if tea_D.shape[-2:] != stu_D.shape[-2:]:
                    tea_D_rs = F.interpolate(tea_D, size=stu_D.shape[-2:], mode="bilinear", align_corners=False)
                    m_rs = F.interpolate(mask.float(), size=stu_D.shape[-2:], mode="nearest").bool() if mask is not None else None
                else:
                    tea_D_rs = tea_D
                    m_rs = mask
                ts = self._frame_mean_absrel(stu_D, tea_D_rs, m_rs)  # (B,L)
                ts_frames.append(ts.detach().cpu().reshape(-1))

            # -------------------------
            # (B) geo 3D distance on fixed ref masks (per-sample)
            # -------------------------
            def _pack(out_full, b_idx: int):
                if out_full is None:
                    return None, None, None, None
                out_b = {}
                for k, v in out_full.items():
                    if torch.is_tensor(v) and v.shape[0] == B:
                        out_b[k] = v[b_idx:b_idx+1]
                    else:
                        out_b[k] = v
                return self._geom_pack(out_b, images[b_idx:b_idx+1], prune_ratio=0.0)  # eval: no conf prune

            for b in range(B):
                if (ids is None) or (not torch.is_tensor(ids)):
                    continue

                seq_name = seqs[b]
                ids_b = ids[b].detach().to("cpu")   # (L,)
                images_b = images[b:b+1]

                # reference output for building mask (only used if is_ref_run)
                out_ref_full = teacher_out if teacher_out is not None else student_out
                out_ref_b = {}
                for k, v in out_ref_full.items():
                    if torch.is_tensor(v) and v.shape[0] == B:
                        out_ref_b[k] = v[b:b+1]
                    else:
                        out_ref_b[k] = v

                ref_mask = self._get_or_build_refmask(
                    ref_dir=ref_dir,
                    is_ref_run=is_ref_run,
                    seq_name=seq_name,
                    ids_1d=ids_b,
                    out_ref_b=out_ref_b,
                    images_b=images_b,
                    anchor=0,
                    prune_ratio=prune_ratio,
                    occ_rel=occ_rel,
                    outlier_q=outlier_q,
                    dist_cap=dist_cap,
                )  # (L-1,H,W) on device

                # teacher geo
                if teacher_out is not None:
                    E, K, D, _ = _pack(teacher_out, b)
                    if (E is not None) and (K is not None) and (D is not None):
                        dist, keep = self._reproj_3d_distance_anchor_eval(
                            D=D, K=K, E=E, ref_mask=ref_mask, anchor=0
                        )
                        geo_teacher.append(dist.detach().cpu().reshape(-1))
                        geo_teacher_keep.append(keep.detach().cpu().reshape(-1))

                # student geo
                E, K, D, _ = _pack(student_out, b)
                if (E is not None) and (K is not None) and (D is not None):
                    dist, keep = self._reproj_3d_distance_anchor_eval(
                        D=D, K=K, E=E, ref_mask=ref_mask, anchor=0
                    )
                    geo_student.append(dist.detach().cpu().reshape(-1))
                    geo_student_keep.append(keep.detach().cpu().reshape(-1))

            # -------------------------
            # per-batch CSV row
            # -------------------------
            def _mean_new(xs, start):
                if len(xs) <= start:
                    return float("nan")
                t = torch.cat(xs[start:], dim=0).float().reshape(-1)
                t = t[torch.isfinite(t)]
                return float(t.mean().item()) if t.numel() else float("nan")

            ts_batch_mean = float("nan")
            if ts is not None and torch.is_tensor(ts):
                t = ts.detach().float().reshape(-1)
                t = t[torch.isfinite(t)]
                ts_batch_mean = float(t.mean().item()) if t.numel() else float("nan")

            geoT_batch_mean  = _mean_new(geo_teacher, geo_teacher_len0)
            geoTk_batch_mean = _mean_new(geo_teacher_keep, geo_teacher_keep_len0)
            geoS_batch_mean  = _mean_new(geo_student, geo_student_len0)
            geoSk_batch_mean = _mean_new(geo_student_keep, geo_student_keep_len0)

            _append_csv_row(batch_csv, {
                "ckpt": ckpt_tag,
                "batch_idx": int(it),
                "B": int(B),
                "L": int(L),
                "ts_absrel_mean": ts_batch_mean,
                "geo_teacher_3ddist_mean": geoT_batch_mean,
                "geo_teacher_keep_mean": geoTk_batch_mean,
                "geo_student_3ddist_mean": geoS_batch_mean,
                "geo_student_keep_mean": geoSk_batch_mean,
            })

        # -------------------------
        # end-of-epoch summary CSV row
        # -------------------------
        geoT  = _cat(geo_teacher)
        geoTk = _cat(geo_teacher_keep)
        geoS  = _cat(geo_student)
        geoSk = _cat(geo_student_keep)
        ts_all = _cat(ts_frames)

        row = {
            "ckpt": ckpt_tag,
            "num_batches": int(num_batches_ran),
            **{f"ts_{k}": v for k, v in _stats(ts_all).items()},
            **{f"geoT_{k}": v for k, v in _stats(geoT).items()},
            **{f"geoTk_{k}": v for k, v in _stats(geoTk).items()},
            **{f"geoS_{k}": v for k, v in _stats(geoS).items()},
            **{f"geoSk_{k}": v for k, v in _stats(geoSk).items()},
        }
        _append_csv_row(summary_csv, row)

        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            objective_mean = self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )
            
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                do_log = (self.steps[phase] % self.logging_conf.log_freq == 0)

                # 1) global norm before clipping
                if do_log:
                    self._tb_log_global_grad_norm(
                        phase=phase,
                        model=self.model,
                        tag="Grad/global_norm_preclip",
                        step=int(self.steps[phase]),
                    )

                grad_norm_dict = self.gradient_clipper(model=self.model)

                # 2) global norm after clipping
                if do_log:
                    self._tb_log_global_grad_norm(
                        phase=phase,
                        model=self.model,
                        tag="Grad/global_norm_postclip",
                        step=int(self.steps[phase]),
                    )

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()
            # Update EMA immediately after each optimizer step
            if self.ema_model is not None:
                base = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
                self.ema_model.update_parameters(base)

            # === log objective once per optimizer update (rank0 only) ===
            if (objective_mean is not None) and (self.rank == 0):
                # self.steps[train] increments per chunk; divide to get optimizer step
                opt_step = (self.steps[phase] // accum_steps) - 1  # first update is step 0
                if opt_step % self.logging_conf.log_freq == 0:
                    self.tb_writer.log("Values/train/objective_optstep", float(objective_mean), int(opt_step))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)
        
        obj_sum = 0.0
        obj_cnt = 0

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            keys_to_drop = {"extrinsics", "intrinsics", "cam_points", "world_points"}
            for k in keys_to_drop:
                chunked_batch.pop(k, None)
            
            gt_depth = chunked_batch["depths"]
            gt_mask = chunked_batch["point_masks"]
            
            S = chunked_batch["images"].size(1)
            t_agg = None

            with torch.inference_mode():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    if self.attn_drop_enabled:
                        t_out, scores, t_agg = self.attn_scorer.forward_and_score_NN(
                            images=chunked_batch["images_aug2"] if self.tea_aug else chunked_batch["images"],
                            amp_dtype=amp_type,
                            return_aggregated_tokens=True,
                        )
                        B, S = scores.shape
                        if not self.drop:
                            # ablation-only
                            # 1. same number with teacher
                            indices = torch.arange(S, device=scores.device, dtype=torch.long)[None, :].expand(B, S)
                        else:
                            if self.prob_sample:
                                indices = sample_one_set_by_scores_batch_prob(
                                    scores=scores,
                                    k_min=2,
                                    k_max=self.drop_max,
                                    mode=self.attn_drop_cfg.mode,   # keep_top / keep_bottom / random_like
                                    must_include_first_last=False,
                                    tau=self.prob_tau,        # suggested initial range: 0.1~0.5
                                    eps=0,  # optional: 0.0~0.1
                                    rng=random,
                                )
                            else:
                                indices = sample_one_set_by_scores_batch(
                                    scores=scores,
                                    k_min=2,
                                    k_max=self.drop_max,
                                    mode=self.attn_drop_cfg.mode,  # keep_top / keep_bottom / random_like
                                    must_include_first_last=False,
                                    rng=random,
                                )
                        # 1) txt log: which frames have low/high scores, teacher vs student frames
                        self._log_frame_selection_txt(
                            phase=phase,
                            chunked_batch=chunked_batch,
                            scores=scores,
                            indices=indices,
                            mode=self.attn_drop_cfg.mode,
                        )

                        # 2) TB stats: mean depth_conf of selected frames vs all frames
                        self._tb_log_depth_conf_stats(
                            phase=phase,
                            teacher_depth_conf=t_out.get("depth_conf", None),
                            indices=indices,
                            scores=scores,
                            mode=self.attn_drop_cfg.mode,
                            k=5,
                        )

                    else:
                        t_out = self.teacher(images=chunked_batch["images_aug2"] if self.tea_aug else chunked_batch["images"],)
                        device = chunked_batch["images"].device
                        B, S = chunked_batch["images"].shape[:2]  # (B,S,C,H,W)

                        if not self.drop:
                            # ablation: keep all frames
                            indices = torch.arange(S, device=device, dtype=torch.long)[None, :].expand(B, S)
                        else:

                            strategy = _choose_strategy(random, p_fixed=getattr(self, "p_fixed_interval", 0.3))
                            indices = sample_one_set(
                                S=S, k_min=2, k_max=self.drop_max, strategy=strategy,
                                must_include_first_last=False, rng=random,
                            )
                            # ---- convert to tensor here ----
                            idx = torch.as_tensor(indices, dtype=torch.long, device=device)   # (L,)
                            idx = idx.unsqueeze(0).expand(B, -1).contiguous()                      # (B,L)

                            indices = idx
            
            chunk_2f = build_seq_from_indices_batch(t_out, chunked_batch, indices)
            
            # --- Crop augmentation (harder student input) ---
            # Synchronously crops images/images_aug + depths/conf/masks + K->K'
            if getattr(self, “crop_cfg”, None) and self.crop_cfg.enabled and (phase == “train”):
                # When rep_enabled: teacher_fs comes from un-cropped teacher tokens -> domain mismatch.
                # Avoid enabling rep_enabled + crop simultaneously, or align by also cropping teacher.
                if not self.rep_enabled:
                    chunk_2f, crop_meta = self._apply_crop_aug(chunk_2f, phase=phase)
                else:
                    crop_meta = None
            # chunk_2f already has: depths / extrinsics / intrinsics / point_masks
            with torch.no_grad():
                cam_pts, world_pts, point_mask = lift_depth_to_cam_world_points_torch(
                    depths=chunk_2f["depths"],
                    extrinsics=chunk_2f["extrinsics"],
                    intrinsics=chunk_2f["intrinsics"],
                )

                new_E, _, _, new_D = normalize_camera_extrinsics_and_points_batch_gpu(
                    extrinsics=chunk_2f["extrinsics"],
                    cam_points=cam_pts,
                    world_points=world_pts,
                    depths=chunk_2f["depths"],
                    point_masks=chunk_2f["point_masks"],
                    scale_by_points=True,
                )

                chunk_2f["extrinsics"] = new_E
                chunk_2f["depths"] = new_D

            # ===== build teacher targets (multi-layer features) =====
            if self.rep_enabled and (t_agg is not None):
                teacher_fs = {}
                for l in self.rep_layers:
                    f_t = _frame_rep_from_agg_tokens(
                        model=self.teacher,
                        agg_tokens=t_agg,
                        images=chunked_batch["images"],  # full S frames
                        layer=int(l),
                        global_feat_start=self.rep_global_feat_start,
                    )  # (B,S,C)

                    f_t_sel = _gather_BS(f_t, indices).detach()  # (B,L,C)
                    teacher_fs[int(l)] = f_t_sel

                chunk_2f["teacher_fs"] = teacher_fs
            
            if self.rank == 0 and (int(self.steps[phase]) % self.logging_conf.log_freq == 0):
                step1 = int(self.steps[phase])  

                tea_conf_sel = chunk_2f.get("depth_conf", None)  # (B,L,H,W)
                self._tb_log_teacher_conf_pixel_stats(
                    phase=phase,
                    teacher_conf=tea_conf_sel,
                    tag_prefix=f"Conf/teacher_selected_pixels",
                    step=step1,
                    log_quantiles=True,
                )

            del t_out
            
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    if self.distraction:
                        loss_dict, stu_depth = self._step(
                            chunk_2f, self.model, phase, loss_meters, chunked_batch["distractor_images_aug"]
                        )
                    else:
                        loss_dict, stu_depth = self._step(
                            chunk_2f, self.model, phase, loss_meters, None
                        )

                # === accumulate objective for opt-step logging (simple mean over chunks) ===
                if "objective" in loss_dict:
                    obj_sum += float(loss_dict["objective"].detach().item())
                    obj_cnt += 1
                                    
                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_objective"
                batch_size = chunk_2f["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)
                
        return (obj_sum / obj_cnt) if obj_cnt > 0 else None


    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict, dis_imgs=None):
        """
        dis_imgs: (B, L_dis, C, H, W)  -- distractor images from the dataset/collate
        """
        if self.aug:
            sup_imgs = batch["images_aug"]  # (B, L_sup, C, H, W)
        else:
            sup_imgs = batch["images"]
        
        L_sup = sup_imgs.size(1)

        if dis_imgs is not None:
            dis_imgs = dis_imgs.to(device=sup_imgs.device, dtype=sup_imgs.dtype)
            dis_sel = _sample_dis_imgs(dis_imgs, L_sup)                           # (B, L_sup, C, H, W)
            images_in = torch.cat([sup_imgs, dis_sel], dim=1)  # (B, L_sup+L_dis, C, H, W)
        else:
            images_in = sup_imgs

        L_total = images_in.size(1)

        need_rep = (self.rep_enabled and phase == "train" and ("teacher_fs" in batch))

        out = model(images=images_in, is_attn=need_rep)
        
        if isinstance(out, tuple) and len(out) == 2:
            y_hat_full, s_agg = out
        else:
            y_hat_full, s_agg = out, None

        y_hat = _slice_pred_dict(y_hat_full, L_sup=L_sup, L_total=L_total)

        loss_dict = self.loss(y_hat, batch)

        if need_rep and (s_agg is not None):
            teacher_fs = batch["teacher_fs"]   # dict: layer -> (B,L_sup,C)

            rep_losses = []
            for l, f_t_sup in teacher_fs.items():
                f_s = _frame_rep_from_agg_tokens(
                    model=model,
                    agg_tokens=s_agg,
                    images=images_in,                # contains distractors
                    layer=int(l),
                    global_feat_start=self.rep_global_feat_start,
                )  # (B,L_total,C)

                f_s_sup = f_s[:, :L_sup, :]         # (B,L_sup,C)

                loss_l = _rep_feature_loss(
                    f_s_sup, f_t_sup,
                    kind=self.rep_loss_kind,
                    smoothl1_beta=self.rep_smoothl1_beta,
                    mix_alpha=self.rep_mix_alpha,
                )
                rep_losses.append(loss_l)

            rep_loss = sum(rep_losses) / max(1, len(rep_losses))
            loss_dict["rep_loss"] = rep_loss
            loss_dict["objective"] = loss_dict["objective"] + self.rep_loss_w * rep_loss

            # logging
            step = int(self.steps[phase])
            if (self.rank == 0) and (step % self.logging_conf.log_freq == 0):
                self.tb_writer.log("Loss/train_rep_loss", float(rep_loss.item()), int(step))
                self.tb_writer.log("Loss/train_rep_w", float(self.rep_loss_w), int(step))
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict, y_hat["depth"]

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )





