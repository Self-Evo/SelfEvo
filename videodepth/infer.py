import hydra
import os
import os.path as osp
import torch
import logging
import json
from omegaconf import DictConfig, ListConfig

from files import get_all_sequences, list_imgs_a_sequence
from messages import set_default_arg
from depth_utils import save_depth_maps
from typing import List, Union, Dict, Any
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import time
import sys

_eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pi3_root = os.path.join(os.path.dirname(os.path.dirname(_eval_dir)), "uncharted4d-Pi3")
if os.path.isdir(_pi3_root):
    sys.path.insert(0, _pi3_root)

_da3_root = os.path.join("/home/pyu/code/uncharted4d-DA3/src")
if os.path.isdir(_da3_root):
    sys.path.insert(0, _da3_root)
DA3_CONFIGS = {
    "da3-large": "/home/pyu/code/uncharted4d-DA3/src/depth_anything_3/configs/da3-large_rayongly.yaml",
    "da3-giant": "/home/pyu/code/uncharted4d-DA3/src/depth_anything_3/configs/da3-giant_rayonly.yaml",
}


CKPT_pretrained = {
    "da3-large": "/home/pyu/code/uncharted4d-DA3/training/ckpts/model_large.pt",
    "da3-giant": "/home/pyu/code/uncharted4d-DA3/training/ckpts/model_giant.pt",
}


def _normalize_state_dict(sd):
    """Normalize state dict keys: apply legacy renames and strip ``model.`` prefix.

    Based on ``convert_general_state_dict`` from ``model_loading.py`` but
    preserves ``camera_token`` and strips the ``model.`` prefix so that keys
    match ``DepthAnything3Net`` parameters directly.
    """
    # Legacy key renames (from convert_general_state_dict)
    sd = {k.replace("module.", "model."): v for k, v in sd.items()}
    sd = {k.replace(".net.", ".backbone."): v for k, v in sd.items()}
    sd = {k.replace(".camera_token_extra", ".camera_token"): v for k, v in sd.items()}
    sd = {
        k.replace("model.all_heads.camera_cond_head", "model.cam_enc"): v
        for k, v in sd.items()
    }
    sd = {
        k.replace("model.all_heads.camera_head", "model.cam_dec"): v
        for k, v in sd.items()
    }
    sd = {k.replace(".more_mlps.", ".backbone."): v for k, v in sd.items()}
    sd = {k.replace(".fc_rot.", ".fc_qvec."): v for k, v in sd.items()}
    sd = {
        k.replace("model.all_heads.head", "model.head"): v for k, v in sd.items()
    }
    sd = {
        k.replace("output_conv2_additional.sky_mask", "sky_output_conv2"): v
        for k, v in sd.items()
    }
    sd = {k.replace("_ray.", "_aux."): v for k, v in sd.items()}
    sd = {k.replace("gaussian_param_head.", "gs_head."): v for k, v in sd.items()}
    # Strip model. prefix so keys match DepthAnything3Net parameters
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    return sd


def create_da3_model(model_type, ckpt_path, use_ema=True, device="cuda"):
    from depth_anything_3.cfg import load_config, create_object

    cfg = load_config(DA3_CONFIGS[model_type])
    model = create_object(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Extract raw state_dict from checkpoint
    if use_ema and isinstance(ckpt.get("ema_model", None), dict) and len(ckpt["ema_model"]) > 0:
        state_dict = ckpt["ema_model"]
        print(f"[create_da3_model] Using ema_model weights from {ckpt_path}")
    elif isinstance(ckpt.get("model", None), dict):
        state_dict = ckpt["model"]
        print(f"[create_da3_model] Using model weights from {ckpt_path}")
    elif isinstance(ckpt, dict) and any(k.startswith("model.") for k in ckpt):
        state_dict = ckpt
        print(f"[create_da3_model] Using flat state_dict from {ckpt_path}")
    else:
        state_dict = ckpt
        print(f"[create_da3_model] Using raw checkpoint from {ckpt_path}")

    state_dict = _normalize_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[create_da3_model] loaded: {len(state_dict) - len(unexpected)} keys, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)

    return model.eval()

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
def infer_videodepth(
    img_paths,
    model,
    device: str = "cuda",
    crop: bool = True,
) -> torch.Tensor:
    model = model.to(device).eval()
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    mode = "crop" if crop else "pad"

    imgs = load_and_preprocess_images(img_paths, mode=mode).to(device)  # (S,3,H,W)

    start = time.time()
    with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
        preds = model(imgs)
    end = time.time()        
    
    depth_map = preds['depth'][0, ..., 0]
    depth_conf = preds['depth_conf'][0, ...]
    return end - start, depth_map, depth_conf

@torch.no_grad()
def infer_videodepth_pi3(
    img_paths,
    model,
    device: str = "cuda",
    crop: bool = True,
):
    model = model.to(device).eval()
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    mode = "crop" if crop else "pad"

    imgs = load_and_preprocess_images(img_paths, mode=mode).to(device)  # (S,3,H,W)

    start = time.time()
    with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
        preds = model(imgs.unsqueeze(0))  # Pi3 expects (B, S, 3, H, W)
    end = time.time()

    depth_map = preds['local_points'][0, ..., 2]        # (S, H, W) — already linear depth
    depth_conf = torch.sigmoid(preds['conf'][0, ..., 0])  # (S, H, W) — logits → probability

    return end - start, depth_map, depth_conf

@torch.no_grad()
def infer_videodepth_da3(
    img_paths,
    model,
    device: str = "cuda",
    crop: bool = True,
):
    model = model.to(device).eval()
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    mode = "crop" if crop else "pad"

    # NOTE: import from 0_test_vis/da3_infer_droid.py or define locally
    imgs = load_and_preprocess_images(img_paths, mode=mode).to(device)  # (S,3,H,W)

    start = time.time()
    with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
        preds = model(imgs.unsqueeze(0), use_ray_pose=True, ref_view_strategy="saddle_balanced")
    end = time.time()

    depth_map = preds['depth'][0]       # (S, H, W)
    depth_conf = preds['depth_conf'][0] # (S, H, W)
    return end - start, depth_map, depth_conf

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/videodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/videodepth.yaml
    model_type: str = getattr(hydra_cfg, 'model', 'vggt')

    # 0. create model
    if model_type == "pi3":
        from pi3.models.pi3 import Pi3
        print(f"[Pi3] Loading checkpoint: {pretrained_model_name_or_path}")
        model = Pi3()
    elif model_type == "da3":
        model = create_da3_model(model_type = "da3-large", ckpt_path=pretrained_model_name_or_path)
    else:
        from vggt.models.vggt import VGGT
        print(f"[VGGT] Loading checkpoint: {pretrained_model_name_or_path}")
        model = VGGT()
    if model_type != "da3":
        model = load_ema_for_eval(model, use_ema=True, ckpt_path=pretrained_model_name_or_path, device="cuda", strict=False).eval()

    logger = logging.getLogger("relpose-angle")
    logger.info(f"Loaded {model_type.upper()} from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        if dataset_info.type == "video":
            # most of the datasets have many sequences of video
            seq_list = get_all_sequences(dataset_info)
        elif dataset_info.type == "mono":
            raise ValueError("dataset type `mono` is not supported for videodepth evaluation")
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        model = model.eval()
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering videodepth on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

        # 3. infer for each sequence (video)
        for seq_idx, seq in enumerate(seq_list, start=1):
            filelist = list_imgs_a_sequence(dataset_info, seq)
            if dataset_info.sparse:
                num = len(filelist)
                target = max(1, num // dataset_info.eval_gap)
                indices = np.linspace(0, num - 1, target, dtype=int)
                filelist = [filelist[i] for i in indices]
                # =============================

            save_dir = osp.join(output_root, seq)

            if not hydra_cfg.overwrite and (osp.isdir(save_dir) and len(os.listdir(save_dir)) == 2 * len(filelist) + 1):
                logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} already processed, skipping.")
                continue
            
            # time_used: float, or List[float] (len = 2)
            # depth_maps: (N, H, W), torch.Tensor
            # conf_self: (N, H, W) torch.Tensor, or just None is ok
            if hydra_cfg.no_crop:
                mode = False
            else:
                mode = True
            if model_type == "pi3":
                time_used, depth_maps, conf_self = infer_videodepth_pi3(filelist, model, hydra_cfg.device, mode)
            elif model_type == "da3":
                time_used, depth_maps, conf_self = infer_videodepth_da3(filelist, model, hydra_cfg.device, mode)
            else:
                time_used, depth_maps, conf_self = infer_videodepth(filelist, model, hydra_cfg.device, mode)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} processed, time: {time_used}, saving depth maps...")

            os.makedirs(save_dir, exist_ok=True)
            save_depth_maps(depth_maps, save_dir, conf_self=conf_self)
            # save time
            with open(osp.join(save_dir, "_time.json"), "w") as f:
                json.dump({
                    "time": time_used,
                    "frames": len(filelist),
                }, f, indent=4)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    set_default_arg("evaluation", "videodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()