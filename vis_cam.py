"""
python vis_cam.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/vggt*cam*best*/*.csv" \
    /home/nan.huang/code/vggt/eval/outputs_vggt/cam_pretrain/*.csv \
  --out_dir figure/plots_cam_bedlam_vggt

python vis_cam.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/pi3*cam*best*/*.csv" \
    /home/nan.huang/code/vggt/eval/outputs/pi3_cam_pretrain/*.csv \
  --out_dir figure/plots_cam_bedlam_pi3

python vis_cam.py \
  --out_dir figure/plots_c_0220\
  --exp_names cam_exp_frcam_alignt_sparse cam_exp_frcam_aligns_sparse cam_bot_995_frcam_prob_sparse cam_random_1_frcam_sparse  cam_bot_995_loss_frcam cam_bot_995_frcam cam_pretrain_sparse cam_pretrain

python vis_cam.py \
  --out_dir figure/plots_c_0220\
  --exp_names cam_exp_frcam_alignt_sparse cam_exp_frcam_aligns_sparse cam_bot_995_frcam_prob_sparse cam_random_1_frcam_sparse  cam_bot_995_loss_frcam_sparse cam_bot_995_frcam_sparse cam_pretrain_sparse cam_pretrain

python vis_cam.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/*cam_*sparse*/*.csv" \
  --out_dir figure/plots_cam_pi3

1. ablation
python vis_cam.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/cam3*hoi4d*sparse*/*.csv" \
    /home/nan.huang/code/vggt/eval/outputs/cam3_pretrain_sparse/*metric*.csv \
  --out_dir figure/plots_cam_hoi4d3

python vis_cam.py \
  --inputs $(printf '%s\n' /home/nan.huang/code/vggt/eval/outputs/cam*sparse*/*.csv | grep -v '99' | grep -v 'croponly2') \
    /home/nan.huang/code/vggt/eval/outputs/cam_pretrain_sparse/*metric*.csv \
  --out_dir figure/plots_cam_ablation

python vis_cam.py \
  --inputs $(printf '%s\n' /home/nan.huang/code/vggt/eval/outputs/cam*/*.csv | grep -v 'sparse' | grep -v 'dense') \
    /home/nan.huang/code/vggt/eval/outputs/cam_pretrain/*metric*.csv \
  --out_dir figure/plots_cam_ori

python vis_cam.py \
  --inputs \
    $(printf '%s\n' /home/nan.huang/code/vggt/eval/outputs/cam*robo*/*.csv | grep -v 'sparse' | grep -v 'dense') \
    /home/nan.huang/code/vggt/eval/outputs/cam_pretrain/*metric*.csv \
  --out_dir figure/robo_cam_ori

2. 
python vis_cam.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/cam*99*sparse*/*.csv" \
    /home/nan.huang/code/vggt/eval/outputs/cam_pretrain_sparse/*metric*.csv \
  --out_dir figure/plots_cam_all

3.
python vis_cam.py \
  --out_dir figure/plots_cam_online\
  --exp_names cam_random_995_frcam_sparse cam_exp_frcam_aligns_sparse cam_exp_frcam_alignt_sparse cam_pretrain_sparse

python vis_cam.py \
  --out_dir figure/plots_cam_0221\
  --exp_names cam_random_99_frcam_sparse cam_random_995_frcam_sparse cam_bot_995_frcam_prob_sparse cam_bot_995_random_frcam cam_bot_995_loss_frcam_sparse  cam_bot_995_loss_frcam_sparse cam_bot_995_frcam_sparse cam_top_995_frcam_sparse cam_pretrain_sparse

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import glob
import argparse
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

# Headless-safe backend (clusters without display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def natural_key(s: str):
    """Sort strings with numbers naturally: checkpoint_2 < checkpoint_10."""
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s)))


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def parse_exp_info(exp_full: str):
    """
    exp_full examples:
      - game_ema99_cam_checkpoint_0      -> tea, series_id=game_ema99_cam
      - game_ema999_checkpoint_12        -> tea, series_id=game_ema999
      - stu_game_ema99_checkpoint_5      -> stu, series_id=stu_game_ema99
      - camera_pretrain                 -> pretrain, series_id=pretrain
    """
    # 如果 exp_full 可能是路径，先取 basename
    exp_base = os.path.basename(exp_full)
    name = exp_base.lower()

    # 1) pretrain
    if "pretrain" in name:
        exp_type = "pretrain"
        ema_tag = "pretrain"
        ckpt_num = None
        exp_short = "pretrain"
        series_id = "pretrain"
        return exp_type, exp_short, ema_tag, ckpt_num, series_id

    # 2) student vs teacher
    if name.startswith("stu_"):
        exp_type = "stu"
        prefix = "stu"
    else:
        exp_type = "tea"
        prefix = "tea"

    # 3) parse ema / ckpt
    ema_m = re.search(r"(ema\d+)", name, flags=re.IGNORECASE)
    ckpt_m = re.search(r"checkpoint_(\d+)", name, flags=re.IGNORECASE)

    ema_tag = ema_m.group(1).lower() if ema_m else None
    ckpt_num = int(ckpt_m.group(1)) if ckpt_m else None

    # 4) make exp_short
    if ema_tag is not None and ckpt_num is not None:
        exp_short = f"{prefix}_{ema_tag}_{ckpt_num}"
    elif ema_tag is not None:
        exp_short = f"{prefix}_{ema_tag}"
    else:
        exp_short = exp_base  # fallback

    # ✅ 5) series_id = 实验名（去掉末尾 _checkpoint_x）
    series_id = re.sub(r"_checkpoint_\d+$", "", name)

    return exp_type, exp_short, ema_tag, ckpt_num, series_id


def build_series_color_map(sub: pd.DataFrame) -> Dict[str, object]:
    """
    series_id -> color

    - student (exp_type == 'stu'): Greens gradient (still distinguish different student series)
    - pretrain: fixed grey
    - others (teacher/other/...): high-contrast qualitative palette, avoid green/grey
    """
    series_color: Dict[str, object] = {}

    def _to_rgba(c):
        if isinstance(c, str):
            return c
        if len(c) == 3:
            return (c[0], c[1], c[2], 1.0)
        return c

    def _is_grey(c, tol=0.06):
        if isinstance(c, str):
            return False
        r, g, b, _ = _to_rgba(c)
        return abs(r - g) < tol and abs(g - b) < tol

    def _is_greenish(c):
        if isinstance(c, str):
            return False
        r, g, b, _ = _to_rgba(c)
        return (g > 0.45) and (g - r > 0.18) and (g - b > 0.18)

    def _qual_palette():
        # long, high-contrast pool
        names = ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Dark2", "Accent"]
        pool = []
        for name in names:
            cmap = cm.get_cmap(name)
            if hasattr(cmap, "colors"):
                pool.extend([_to_rgba(x) for x in cmap.colors])
            else:
                n = max(20, getattr(cmap, "N", 256))
                pool.extend([_to_rgba(cmap(i / (n - 1))) for i in range(n)])

        # avoid greys + greens (so they don't clash with pretrain/student)
        filtered = [c for c in pool if (not _is_grey(c)) and (not _is_greenish(c))]
        return filtered if len(filtered) >= 12 else pool

    # 0) pretrain fixed grey (in case caller includes it)
    if (sub["exp_type"] == "pretrain").any():
        series_color["pretrain"] = "0.6"

    # 1) student -> Greens gradient
    stu_df = sub[sub["exp_type"] == "stu"]
    if not stu_df.empty:
        sids = sorted(stu_df["series_id"].unique(), key=natural_key)
        n = len(sids)
        cmap = cm.get_cmap("Greens")
        if n == 1:
            vals = [0.80]
        else:
            lo, hi = 0.55, 0.95
            vals = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        for sid, v in zip(sids, vals):
            series_color[sid] = cmap(v)

    # 2) others -> colorful palette (teacher/other/whatever), excluding pretrain + student
    other_df = sub[(sub["exp_type"] != "stu") & (sub["exp_type"] != "pretrain")]
    if not other_df.empty:
        palette = _qual_palette()
        sids = sorted(other_df["series_id"].unique(), key=natural_key)
        for i, sid in enumerate(sids):
            series_color[sid] = palette[i % len(palette)]

    return series_color


def parse_one_csv(p: Path) -> dict:
    """
    Input path example:
      /home/.../outputs/game_ema99_checkpoint_0/OmniGeo-metric.csv
      /home/.../outputs/cam_game_ema999_checkpoint_3/OmniVideo-metric-something.csv

    Filename:
      <dataset>-metric(.csv)
      <dataset>-metric-<suffix>.csv
    """
    exp_full = p.parent.name
    exp_type, exp_short, ema_tag, ckpt_num, series_id = parse_exp_info(exp_full)

    m = re.match(r"^(?P<dataset>[^-]+)-metric(?:-(?P<suffix>.+))?\.csv$", p.name)
    if not m:
        raise ValueError(f"Filename not match '<dataset>-metric(-suffix).csv': {p.name}")

    dataset = m.group("dataset")
    suffix = m.group("suffix") or ""

    df = pd.read_csv(p)
    # 容错：多行就取 mean（一般是一行）
    if "Auc_15" not in df.columns or "Auc_30" not in df.columns:
        raise KeyError(f"Missing columns Auc_15/Auc_30 in {p}. columns={list(df.columns)}")

    auc15 = float(df["Auc_15"].mean())
    auc30 = float(df["Auc_30"].mean())

    return {
        "path": str(p),
        "exp_full": exp_full,
        "exp_short": exp_short,
        "exp_type": exp_type,
        "ema_tag": ema_tag,
        "ckpt_num": ckpt_num,
        "series_id": series_id,
        "dataset": dataset,
        "suffix": suffix,
        "Auc_15": auc15,
        "Auc_30": auc30,
    }


def plot_line(df_all: pd.DataFrame, out_dir: Path):
    metrics = ["Auc_15", "Auc_30"]

    group_cols = ["dataset", "suffix"]
    for (dataset, suffix), sub in df_all.groupby(group_cols, sort=False):
        sub = sub.copy()

        # 1) 分出 pretrain（水平线）和主曲线（tea/stu/... 的点+线）
        sub_pre  = sub[sub["exp_type"] == "pretrain"].copy()
        sub_main = sub[sub["exp_type"] != "pretrain"].copy()

        # 2) ckpt 轴：优先用主曲线里出现的 ckpt；如果没有主曲线，就默认 ckpt_0..9
        sub_main = sub_main[sub_main["ckpt_num"].notna()].copy()
        if not sub_main.empty:
            sub_main["ckpt_num"] = sub_main["ckpt_num"].astype(int)
            ckpts = sorted(sub_main["ckpt_num"].unique())
        else:
            ckpts = list(range(10))

        xticks = ckpts
        xticklabels = [f"{k}" for k in ckpts]

        # 颜色：只对主曲线做颜色映射；pretrain 固定绿色
        series_color = build_series_color_map(sub_main) if not sub_main.empty else {}

        title_suffix = f" ({suffix})" if suffix else ""
        for metric in metrics:
            plt.figure(figsize=(10, 4.5))

            # 3) 主曲线
            if not sub_main.empty:
                for (exp_type, sid), g in sub_main.groupby(["exp_type", "series_id"], sort=False):
                    g = g.sort_values("ckpt_num")
                    c = series_color.get(sid, "0.4")
                    plt.plot(
                        g["ckpt_num"].tolist(),
                        g[metric].tolist(),
                        marker="o",
                        linewidth=1.6,
                        color=c,
                        alpha=0.95,
                        label=sid,
                    )

            # 4) pretrain：画成水平线（每个 ckpt 都同一个值）
            if not sub_pre.empty:
                y = float(sub_pre[metric].mean())  # 多个 pretrain 结果就取平均
                plt.plot(
                    ckpts,
                    [y] * len(ckpts),
                    color="0.6",
                    linewidth=2.2,
                    linestyle="-",
                    alpha=0.95,
                    label="pretrain",
                )

            plt.title(f"{dataset}{title_suffix} - {metric}")
            plt.xlabel("checkpoint")
            plt.ylabel(metric)
            plt.xticks(xticks, xticklabels)
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

            plt.legend(frameon=True, fontsize=9)
            plt.tight_layout()

            fname = f"line__{dataset}__{sanitize_filename(suffix or 'nosuffix')}__{sanitize_filename(metric)}.png"
            plt.savefig(out_dir / fname, dpi=200)
            plt.close()


def plot_scatter(df_all: pd.DataFrame, out_dir: Path, annotate: bool = True):
    # 以 dataset + suffix 为一组画图
    group_cols = ["dataset", "suffix"]
    for (dataset, suffix), sub in df_all.groupby(group_cols, sort=False):
        sub = sub.copy()

        series_color = build_series_color_map(sub)

        plt.figure(figsize=(6.8, 5.8))
        # 逐点画，方便标注 ckpt
        for _, r in sub.iterrows():
            sid = r["series_id"]
            c = series_color.get(sid, "0.4")
            x = float(r["Auc_15"])
            y = float(r["Auc_30"])
            plt.scatter(x, y, color=c)

            if annotate and pd.notna(r.get("ckpt_num")):
                plt.annotate(
                    str(int(r["ckpt_num"])),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 3),
                    fontsize=8,
                    color=c,
                    alpha=0.95,
                )

        title_suffix = f" ({suffix})" if suffix else ""
        plt.title(f"Scatter: {dataset}{title_suffix}\n(x=Auc_15, y=Auc_30)")
        plt.xlabel("Auc_15 (higher is better)")
        plt.ylabel("Auc_30 (higher is better)")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        # legend：按 series_id
        handles = []
        labels = []
        for sid in sorted(sub["series_id"].unique(), key=natural_key):
            handles.append(plt.Line2D([0], [0], marker='o', linestyle='', color=series_color.get(sid, "0.4")))
            labels.append(sid)

        if len(labels) <= 12:
            plt.legend(handles, labels, frameon=True, fontsize=9)
        else:
            plt.legend(handles, labels, frameon=True, fontsize=8, ncol=2)

        plt.tight_layout()
        fname = f"scatter__{dataset}__{sanitize_filename(suffix or 'nosuffix')}__auc15_vs_auc30.png"
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="CSV paths / dir paths / glob patterns. "
             "e.g. '/.../outputs/*/OmniGeo-metric*.csv' or '/.../outputs/game_ema99_checkpoint_0'",
    )
    ap.add_argument(
        "--exp_names",
        nargs="+",
        default=None,
        help="Experiment name keywords. If provided (and --inputs not provided), "
             "will expand to '<outputs_root>/*{exp}*/*.csv' for each exp.",
    )
    ap.add_argument(
        "--outputs_root",
        default="/home/nan.huang/code/vggt/eval/outputs",
        help="Root folder containing experiment output dirs (used with --exp_names).",
    )
    ap.add_argument("--out_dir", default="plots_cam_metrics", help="Output dir for pngs")
    ap.add_argument("--no_annotate", action="store_true", help="Disable point labels on scatter plots")
    args = ap.parse_args()

    # ---------------------------
    # 1) Decide input patterns
    # ---------------------------
    if args.inputs:
        input_items = args.inputs
    elif args.exp_names:
        root = Path(args.outputs_root)
        input_items = [str(root / f"*{name}*/*.csv") for name in args.exp_names]
    else:
        raise SystemExit("You must provide either --inputs or --exp_names.")

    # ---------------------------
    # 2) Expand globs / dirs / files
    # ---------------------------
    paths = []
    for item in input_items:
        # glob pattern
        if any(ch in item for ch in ["*", "?", "["]):
            paths.extend([Path(p) for p in glob.glob(item, recursive=True)])
            continue

        p = Path(item)
        # directory -> search for csvs inside (recursive)
        if p.exists() and p.is_dir():
            paths.extend([Path(x) for x in glob.glob(str(p / "**" / "*.csv"), recursive=True)])
        else:
            # file path (may or may not exist yet; we'll filter later)
            paths.append(p)

    # filter: only *-metric*.csv
    def is_valid(p: Path) -> bool:
        if (not p.is_file()) or (not p.name.lower().endswith(".csv")):
            return False
        return "-metric" in p.name

    paths = [p for p in paths if is_valid(p)]
    paths = sorted(paths, key=lambda x: natural_key(str(x)))

    if not paths:
        raise SystemExit("No csv files found. Check your --inputs / --exp_names patterns/paths.")

    rows = []
    for p in paths:
        try:
            rows.append(parse_one_csv(p))
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    if not rows:
        raise SystemExit("Parsed 0 valid csvs.")

    df_all = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save merged table for sanity check
    df_all.sort_values(
        ["dataset", "suffix", "series_id", "ckpt_num", "exp_full"],
        key=lambda s: s.map(lambda x: natural_key(str(x))),
    ).to_csv(out_dir / "merged_auc15_auc30.csv", index=False)

    plot_line(df_all, out_dir)
    plot_scatter(df_all, out_dir, annotate=(not args.no_annotate))

    print(f"Done. Outputs saved to: {out_dir.resolve()}")
    print(f"Merged table: {(out_dir / 'merged_auc15_auc30.csv').resolve()}")


if __name__ == "__main__":
    main()

