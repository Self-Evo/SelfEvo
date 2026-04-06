
import os
import os.path as osp
import numpy as np
import torch
import hydra
import logging
import json

from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

import rootutils
root = rootutils.setup_root(__file__, indicator="vggt", pythonpath=True)
from vggt.models.vggt import VGGT

import sys
_eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pi3_root = os.path.join(os.path.dirname(_eval_dir), "uncharted4d-Pi3")
if os.path.isdir(_pi3_root):
    sys.path.insert(0, _pi3_root)
from messages import set_default_arg, write_csv
from metric import se3_to_relative_pose_error, calculate_auc_np, build_pair_index
from geometry import closed_form_inverse_se3
from pi3.utils.geometry import se3_inverse
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from typing import List, Union, Dict, Any
from PIL import Image
import pandas as pd
import torchvision.transforms.functional as TF

def maybe_sparse_sample_ids(
    ids: np.ndarray,
    *,
    sparse: bool,
    divisor: int = 10,          # 1/10
    min_frames: int = 2,        # 至少 2 帧，否则相对位姿没意义
) -> np.ndarray:
    """
    输入：全量 ids（升序）
    输出：sparse 后的 ids（升序，整数）
    """
    if not sparse:
        return ids

    ids = np.asarray(ids, dtype=int)
    num = len(ids)
    if num < min_frames:
        return ids  # 交给上层决定是否 skip

    target = max(min_frames, num // max(1, int(divisor)))
    target = min(target, num)

    # 均匀采样（包含首尾）
    picked = np.linspace(0, num - 1, target, dtype=int)
    out = ids[picked]

    # 去重&排序（极端情况下 linspace 会重复）
    out = np.unique(out)
    if len(out) < min_frames and num >= min_frames:
        # 兜底：至少保留首尾
        out = np.unique(np.array([ids[0], ids[-1]], dtype=int))
    return out

def generate_strided_subsequences(
    num_frames: int,
    stride: int,
    num_subseqs: int,
    min_len: int = 3,
) -> List[np.ndarray]:
    """
    Generate num_subseqs maximally-long subsequences with given stride.
    Offsets are evenly spread across [0, stride-1]. No wrapping.
    Returns list of np.ndarray, each containing frame indices.
    Subsequences shorter than min_len are discarded.
    """
    max_offsets = min(num_subseqs, stride)
    offsets = np.round(np.linspace(0, stride - 1, max_offsets)).astype(int)
    offsets = np.unique(offsets)  # deduplicate after rounding

    subseqs = []
    for o in offsets:
        ids = np.arange(o, num_frames, stride)
        if len(ids) >= min_len:
            subseqs.append(ids)
    return subseqs

def load_and_preprocess_images(
    images_or_paths: List[Union[str, np.ndarray]],
    mode: str = "crop",
    arrays_are_bgr: bool = True,
) -> torch.Tensor:
    """
    支持两种输入：
      1) List[str]：图像文件路径（原行为，完全兼容）
      2) List[np.ndarray]：内存中的图像（形状 HxWx3），通常来自 cv2.VideoCapture.read()
         - 默认为 BGR（OpenCV），会自动转为 RGB。可用 arrays_are_bgr=False 跳过转换。

    返回：
      torch.Tensor: 形状 (N, 3, H, W)

    预处理策略与原版一致：
      - mode="crop"：固定宽 518，等比缩放；若高超过 518，做中心裁剪到 518
      - mode="pad" ：最大边缩放到 518，等比；再把短边 padding 到 518x518（白色=1.0）
      - 最后若尺寸不一致，会做统一 padding；尺寸都对齐到可被 14 整除
    """
    if len(images_or_paths) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.to_tensor  # note: torchvision >=0.15 推荐用小写别名
    target_size = 518  # 518 = 14 * 37

    def _as_pil_rgb(x: Union[str, np.ndarray]) -> Image.Image:
        if isinstance(x, str):
            img = Image.open(x)
            # 处理 RGBA 贴白底
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            return img.convert("RGB")
        elif isinstance(x, np.ndarray):
            arr = x
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"NumPy image must be HxWx3, got shape={arr.shape}")
            # OpenCV => BGR，转为 RGB
            if arrays_are_bgr:
                arr = arr[..., ::-1]
            return Image.fromarray(arr.astype(np.uint8), mode="RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

    for item in images_or_paths:
        img = _as_pil_rgb(item)
        width, height = img.size

        if mode == "pad":
            # 最大边缩放到 518，等比，且四舍五入到 14 的倍数
            if width >= height:
                new_width = target_size
                new_height = max(14, round(height * (new_width / width) / 14) * 14)
            else:
                new_height = target_size
                new_width  = max(14, round(width  * (new_height / height) / 14) * 14)
        else:  # "crop"
            new_width  = target_size
            new_height = max(14, round(height * (new_width / width) / 14) * 14)

        # 调整大小（注意 PIL resize 传 (W,H)）
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        ten = to_tensor(img)  # [0,1], 形状 (3,H,W)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            ten = ten[:, start_y : start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - ten.shape[1]
            w_padding = target_size - ten.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top    = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left   = w_padding // 2
                pad_right  = w_padding - pad_left
                ten = torch.nn.functional.pad(
                    ten, (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant", value=1.0
                )

        shapes.add((ten.shape[1], ten.shape[2]))
        images.append(ten)

    # 若尺寸不一致，做统一 pad（保留内容、四周补白）
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(h for h, _ in shapes)
        max_width  = max(w for _, w in shapes)
        padded = []
        for ten in images:
            h_padding = max_height - ten.shape[1]
            w_padding = max_width  - ten.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top    = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left   = w_padding // 2
                pad_right  = w_padding - pad_left
                ten = torch.nn.functional.pad(
                    ten, (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant", value=1.0
                )
            padded.append(ten)
        images = padded

    images = torch.stack(images)  # (N,3,H,W)
    if len(images_or_paths) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)
    return images

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict):
        return sd
    return {
        (k.replace("module.", "", 1) if isinstance(k, str) and k.startswith("module.") else k): v
        for k, v in sd.items()
    }

def load_ema_for_eval(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    ema_key: str = "ema_model",
    map_location: str = "cpu",
    device: str = "cuda",
    strict: bool = False,
    use_ema: bool = True,  
) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = None
    
    if use_ema:
        if isinstance(ckpt.get(ema_key, None), dict) and len(ckpt[ema_key]) > 0:
            state_dict = _strip_module_prefix(ckpt[ema_key])
        elif isinstance(ckpt.get("ema_teacher", None), dict) and len(ckpt["ema_teacher"]) > 0:
            print(f"[load_ema_for_eval] Using ema_teacher weights from {ckpt_path}")
            state_dict = _strip_module_prefix(ckpt["ema_teacher"])
        else:
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            print(f"[load_ema_for_eval] Using student weights from {ckpt_path}")
            state_dict = _strip_module_prefix(state_dict)
    else:
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        print(f"[load_ema_for_eval] Using student weights from {ckpt_path}")
        state_dict = _strip_module_prefix(state_dict)
    if state_dict is None:
        print(f"No usable model weights found in {ckpt_path} (ema/model missing).")
        return model.to(device).eval()
        # raise ValueError(f"No usable model weights found in {ckpt_path} (ema/model missing).")

    base = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = base.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"[load_ema_for_eval] missing={len(missing)}, unexpected={len(unexpected)}")

    base.to(device).eval()
    return model

@torch.no_grad()
def infer_cameras_w2c(
    img_paths,
    model,
    model_type: str = "vggt",
    device: str = "cuda",
    crop: bool = True,
) -> torch.Tensor:
    """
    Return W2C extrinsics as (S, 4, 4) tensors.
    Supports both VGGT (outputs W2C directly) and Pi3 (outputs C2W, converted here).
    """
    model = model.to(device).eval()
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    mode = "crop" if crop else "pad"

    imgs = load_and_preprocess_images(img_paths, mode=mode).to(device)  # (S,3,H,W)

    if model_type == "pi3":
        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            preds = model(imgs.unsqueeze(0))  # Pi3 expects (B, S, 3, H, W)

        c2w = preds['camera_poses'].cpu()[0]  # (S, 4, 4)
        w2c = se3_inverse(c2w)               # (S, 4, 4)
        return w2c
    else:
        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            preds = model(imgs)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                preds["pose_enc"], preds["images"].shape[-2:]
            )
            extrinsic = extrinsic.cpu()[0]

        S = extrinsic.shape[0]
        last_row = torch.tensor([0, 0, 0, 1], dtype=extrinsic.dtype, device=extrinsic.device).view(1,1,4)
        last_row = last_row.expand(S, 1, 4)
        extrinsic44 = torch.cat([extrinsic, last_row], dim=-2)  # (S,4,4)
        return extrinsic44  # (S,4,4)

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/relpose-angular.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    pretrained_model_name_or_path: str = hydra_cfg.vggt.pretrained_model_name_or_path  # see configs/evaluation/relpose-angular.yaml
    use_ema: bool = getattr(hydra_cfg.vggt, 'use_ema', True)
    model_type: str = getattr(hydra_cfg, 'model', 'vggt')

    # 0. create model
    if model_type == "pi3":
        from pi3.models.pi3 import Pi3
        print(f"[Pi3] Loading checkpoint: {pretrained_model_name_or_path}")
        model = Pi3()
    else:
        print(f"[VGGT] Loading checkpoint: {pretrained_model_name_or_path}")
        model = VGGT()
    model = load_ema_for_eval(model, ckpt_path=pretrained_model_name_or_path, device="cuda", strict=False, use_ema=use_ema).eval()

    logger = logging.getLogger("relpose-angle")
    logger.info(f"Loaded {model_type.upper()} from {pretrained_model_name_or_path}")
    
    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data, decide the dataset name
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 2. ready to read: decide whether we use seq_id_map sampling or full frames
        model.eval()

        has_sampling = ("sampling" in dataset_info) and (dataset_info.sampling is not None)
        has_seq_id_map = ("seq_id_map" in dataset_info) and (dataset_info.seq_id_map is not None)

        seq_id_map = None
        if has_sampling and has_seq_id_map:
            sample_config: DictConfig = dataset_info.sampling
            logger.info(f"Sampling strategy: {sample_config.strategy}")
            with open(dataset_info.seq_id_map, "r") as f:
                seq_id_map = json.load(f)
        else:
            logger.info("No sampling/seq_id_map provided -> evaluating ALL frames in each sequence.")

        # 3. prepare for metrics
        rError = []
        tError = []
        metric_dict: dict = {}
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating {dataset_name} with {model_type.upper()}...")

        per_seq_rows = []
        tbar = tqdm(dataset.sequence_list, desc=f"[{dataset_name} eval]")
        # New config flags for strided subsequence mode
        fixed_stride = int(getattr(dataset_info, "fixed_stride", 0))
        num_test_per_seq = int(getattr(dataset_info, "num_test_per_seq", 1))

        for seq_name in tbar:
            if fixed_stride > 0:
                # ---- Strided subsequence mode ----
                num_frames = dataset.get_seq_framenum(sequence_name=seq_name)
                subseq_list = generate_strided_subsequences(
                    num_frames, fixed_stride, num_test_per_seq, min_len=3,
                )
                if not subseq_list:
                    logger.warning(f"[{dataset_name}] Skip {seq_name}: no valid subseq (stride={fixed_stride}, N={num_frames})")
                    continue

                seq_rError_all, seq_tError_all = [], []
                for ids in subseq_list:
                    batch = dataset.get_data(sequence_name=seq_name, ids=ids)
                    gt_extrs = batch["extrs"]

                    with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
                        pred_extrs = infer_cameras_w2c(batch["image_paths"], model, model_type=model_type)
                        rel_r, rel_t = se3_to_relative_pose_error(
                            pred_se3=pred_extrs,
                            gt_se3=gt_extrs,
                            num_frames=len(ids),
                        )

                        # optionally filter out pairs with small GT translation
                        filter_flag = bool(getattr(dataset_info, "filter_small_translation", False))
                        filter_thr = float(getattr(dataset_info, "filter_small_translation_thr", 0.7))
                        if filter_flag:
                            pair_i1, pair_i2 = build_pair_index(len(ids))
                            relative_pose_gt = gt_extrs[pair_i2].bmm(
                                closed_form_inverse_se3(gt_extrs[pair_i1])
                            )
                            gt_t_norms = torch.norm(relative_pose_gt[:, :3, 3], dim=1)
                            keep_mask = gt_t_norms >= filter_thr
                            rel_r = rel_r[keep_mask]
                            rel_t = rel_t[keep_mask]

                    if len(rel_r) > 0:
                        seq_rError_all.append(rel_r)
                        seq_tError_all.append(rel_t)

                if not seq_rError_all:
                    logger.warning(f"[{dataset_name}] Skip {seq_name}: all subseq pairs filtered.")
                    continue

                rel_rangle_deg = torch.cat(seq_rError_all)
                rel_tangle_deg = torch.cat(seq_tError_all)

            else:
                # ---- Existing logic (seq_id_map / sparse) ----
                # 4. decide frame ids
                if seq_id_map is not None:
                    ids = np.asarray(seq_id_map[seq_name], dtype=int)
                else:
                    num_frames = dataset.get_seq_framenum(sequence_name=seq_name)
                    ids = np.arange(num_frames, dtype=int)

                # 4b. optional sparse eval (default: 1/10)
                sparse_flag = bool(getattr(dataset_info, "sparse", False))
                sparse_divisor = int(getattr(dataset_info, "sparse_divisor", 10))
                sparse_min_frames = int(getattr(dataset_info, "sparse_min_frames", 2))

                if sparse_flag and (seq_id_map is None):
                    ids = maybe_sparse_sample_ids(
                        ids,
                        sparse=True,
                        divisor=sparse_divisor,
                        min_frames=sparse_min_frames,
                    )

                if len(ids) < 2:
                    logger.warning(f"[{dataset_name}] Skip {seq_name}: only {len(ids)} frame(s) after sampling.")
                    continue

                # 5. load data sample (only extrinsics are used)
                batch = dataset.get_data(sequence_name=seq_name, ids=ids)
                gt_extrs = batch["extrs"]

                with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
                    # 6. infer cameras
                    pred_extrs = infer_cameras_w2c(batch["image_paths"], model, model_type=model_type)

                    # 7. compute metrics
                    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                        pred_se3=pred_extrs,
                        gt_se3=gt_extrs,
                        num_frames=len(ids),
                    )

                    # 7b. optionally filter out pairs with small GT translation
                    filter_flag = bool(getattr(dataset_info, "filter_small_translation", False))
                    filter_thr = float(getattr(dataset_info, "filter_small_translation_thr", 0.7))

                    if filter_flag:
                        pair_i1, pair_i2 = build_pair_index(len(ids))
                        relative_pose_gt = gt_extrs[pair_i2].bmm(
                            closed_form_inverse_se3(gt_extrs[pair_i1])
                        )
                        gt_t_norms = torch.norm(relative_pose_gt[:, :3, 3], dim=1)
                        keep_mask = gt_t_norms >= filter_thr
                        rel_rangle_deg = rel_rangle_deg[keep_mask]
                        rel_tangle_deg = rel_tangle_deg[keep_mask]

                        if keep_mask.sum() == 0:
                            logger.warning(f"[{dataset_name}] Skip {seq_name}: all pairs filtered (small translation).")
                            continue

            tbar.set_postfix_str(
                f"Sequence {seq_name} RotErr(Deg): {rel_rangle_deg.mean():5.2f} | TransErr(Deg): {rel_tangle_deg.mean():5.2f}"
            )

            rError.extend(rel_rangle_deg.cpu().numpy())
            tError.extend(rel_tangle_deg.cpu().numpy())

            # per-sequence metrics
            seq_rError = rel_rangle_deg.cpu().numpy()
            seq_tError = rel_tangle_deg.cpu().numpy()
            row = {"seq_name": seq_name}
            for threshold in dataset_info.metric_thresholds:
                row[f"Racc_{threshold}"] = np.mean(seq_rError < threshold).item() * 100
                row[f"Tacc_{threshold}"] = np.mean(seq_tError < threshold).item() * 100
                Auc, _ = calculate_auc_np(seq_rError, seq_tError, max_threshold=threshold)
                row[f"Auc_{threshold}"] = Auc.item() * 100
            per_seq_rows.append(row)
        
        rError = np.array(rError)
        tError = np.array(tError)
        
        # 9. arrange all intermediate results to metrics
        for threshold in dataset_info.metric_thresholds:
            metric_dict[f"Racc_{threshold}"] = np.mean(rError < threshold).item() * 100
            metric_dict[f"Tacc_{threshold}"] = np.mean(tError < threshold).item() * 100
            Auc, _ = calculate_auc_np(rError, tError, max_threshold=threshold)
            metric_dict[f"Auc_{threshold}"]  = Auc.item() * 100

        # save per-sequence metrics to csv
        perseq_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-camera_perseq")
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            perseq_file += f"-{hydra_cfg.save_suffix}"
        perseq_file += ".csv"
        pd.DataFrame(per_seq_rows).to_csv(perseq_file, index=False)
        logger.info(f"Per-sequence results saved to {perseq_file}")

        logger.info(f"{dataset_name} - Average pose estimation metrics: {metric_dict}")

        # 9. save evaluation metrics to csv
        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, metric_dict)

    del model
    torch.cuda.empty_cache()
    logger.info(f"Finished evaluating model {model_type.upper()} on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-angular")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
