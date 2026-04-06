import os
import os.path as osp
import hydra
import numpy as np
import cv2
import json
import logging
import glob
from omegaconf import DictConfig, ListConfig

from depth import EVAL_DEPTH_METADATA, depth_evaluation
from files import get_all_sequences, list_depths_a_sequence
from messages import set_default_arg, write_csv


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/videodepth.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data

    logger = logging.getLogger("videodepth-eval")
    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data, decide the dataset name
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get gt and pred depth pathes
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        if dataset_info.type == "video":
            # most of the datasets have many sequences of video
            seq_list = get_all_sequences(dataset_info)
            gt_paths = {
                seq: list_depths_a_sequence(dataset_info, seq)
                for seq in seq_list
            }
            pred_paths = {
                seq: sorted(glob.glob(f"{output_root}/{seq}/*.npy"))
                for seq in seq_list
            }
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        # 3. get depth read function and evaluation kwargs
        videodepth_metadata = EVAL_DEPTH_METADATA.get(dataset_name, None)
        if videodepth_metadata is None:
            raise ValueError(f"Dataset {dataset_name} doesn't have video depth metadata")
        depth_read_func = videodepth_metadata["depth_read_func"]
        depth_evaluation_kwargs = videodepth_metadata["depth_evaluation_kwargs"]
        if hydra_cfg.align == "scale&shift":
            depth_evaluation_kwargs["align_with_lad2"] = True
        elif hydra_cfg.align == "scale":
            depth_evaluation_kwargs["align_with_scale"] = True
        elif hydra_cfg.align == "metric":
            depth_evaluation_kwargs["metric_scale"] = True
        else:
            raise ValueError(f"Unknown alignment method: {hydra_cfg.align}")

        gathered_depth_metrics = []
        all_fps = []
        total_time_per_seq = []
        infer_time_per_seq = []

        # >>> MOD: 新增，用于保存每个 seq 的一行结果
        per_seq_rows = []  # list[dict]

        logger.info(
            f"[{idx_dataset}/{len(all_eval_datasets)}] Start evaluating dataset: "
            f"{dataset_name}, {len(pred_paths)} sequences in total"
        )

        for idx_seq, seq in enumerate(pred_paths, start=1):
            # 3.1 get pred & gt depths for each seq
            seq_pd_paths = pred_paths[seq]
            seq_gt_paths = gt_paths[seq]

            # 如果长度不一样，用和推理时一样的等距采样方式从 GT 里取出同样数量
            if len(seq_pd_paths) != len(seq_gt_paths):
                num_gt   = len(seq_gt_paths)
                num_pred = len(seq_pd_paths)
                assert num_pred <= num_gt, f"pred more than gt for seq {seq}"

                indices = np.linspace(0, num_gt - 1, num_pred, dtype=int)
                seq_gt_paths = [seq_gt_paths[i] for i in indices]
            else:
                seq_gt_paths = seq_gt_paths

            logger.info(
                f"[{idx_seq}/{len(pred_paths)}] Evaluating {seq} in {dataset_name}, "
                f"{len(seq_pd_paths)} images in total"
            )

            # 3.2 pack the pred & gt depths
            gt_depth = np.stack([depth_read_func(gt_path) for gt_path in seq_gt_paths], axis=0)
            pr_depth = np.stack(
                [
                    cv2.resize(
                        np.load(pd_path),
                        (gt_depth.shape[2], gt_depth.shape[1]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    for pd_path in seq_pd_paths
                ],
                axis=0,
            )

            # 3.3 evaluate depth
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth, gt_depth, **depth_evaluation_kwargs
            )
            gathered_depth_metrics.append(depth_results)

            # 3.4 calculate fps
            with open(osp.join(output_root, seq, f"_time.json"), "r") as f:
                timing_data = json.load(f)
            timing = timing_data["time"]
            if isinstance(timing, list):
                total_time = sum(timing)
                total_time_per_seq.append(total_time)
                infer_time_per_seq.append(timing[0])
            elif isinstance(timing, float):
                total_time = timing
                total_time_per_seq.append(timing)
                infer_time_per_seq.append(timing)
            else:
                raise ValueError(f"Unknown timing type: {type(timing)}")
            all_fps.append(timing_data["frames"] / total_time)

            # 3.5 calculate metrics for this sequence
            # >>> MOD: 这里原来只是 logger.info，没有保存。现在把每个 seq 的结果做成一行 dict 并收集起来
            this_seq_metrics = dict(depth_results)  # copy，避免后面 dict 被意外复用/修改
            this_seq_metrics["seq"] = seq
            this_seq_metrics["dataset"] = dataset_name
            this_seq_metrics["num_frames"] = len(seq_pd_paths)

            this_seq_metrics["fps"] = all_fps[-1]
            this_seq_metrics["total_time"] = total_time_per_seq[-1]
            this_seq_metrics["infer_time"] = infer_time_per_seq[-1]

            per_seq_rows.append(this_seq_metrics)

            logger.info(f"{dataset_name} - seq {seq} - videodepth metrics: {this_seq_metrics}")

        # 4. save evaluation metrics to csv (overall)
        total_average_metrics = {
            key: np.average(
                [metrics[key] for metrics in gathered_depth_metrics],
                weights=[metrics["valid_pixels"] for metrics in gathered_depth_metrics],
            )
            for key in gathered_depth_metrics[0].keys()
            if key != "valid_pixels"
        }
        total_average_metrics["fps"] = np.average(all_fps).item()
        total_average_metrics["total_time"] = np.average(total_time_per_seq).item()
        total_average_metrics["infer_time"] = np.average(infer_time_per_seq).item()
        logger.info(f"{dataset_name} - Average Videodepth Metrics: {total_average_metrics}")

        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric-{hydra_cfg.align}")
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"

        # 原来的 overall CSV（保持）
        if osp.isfile(statistics_file):
            os.remove(statistics_file)  # 让每次 eval 都是“干净”的 per-seq 文件（避免 append 叠加旧结果）
        write_csv(statistics_file, total_average_metrics)

        # >>> MOD: 新增 per-seq CSV（全新文件，每个 seq 一行）
        per_seq_file = statistics_file.replace(".csv", "-per-seq.csv")
        if osp.isfile(per_seq_file):
            os.remove(per_seq_file)  # 让每次 eval 都是“干净”的 per-seq 文件（避免 append 叠加旧结果）

        # 逐行写入（复用你现有的 write_csv：一次写一行）
        for row in per_seq_rows:
            write_csv(per_seq_file, row)

        logger.info(f"{dataset_name} - Saved per-sequence metrics to: {per_seq_file}")


if __name__ == "__main__":
    set_default_arg("evaluation", "videodepth")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
