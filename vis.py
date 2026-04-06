'''
python vis2.py \
  --out_dir figure/plots_vdepth_online\
  --exp_names videodepth_random_995_frcam_sparse videodepth_exp_frcam_aligns_sparse videodepth_exp_frcam_alignt_sparse videodepth_pretrain_sparse   --exp_names videodepth_random_995_frcam_sparse videodepth_exp_frcam_aligns_sparse videodepth_exp_frcam_alignt_sparse videodepth_pretrain


python vis2.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/vggt*depth*best*/*.csv" \
    /home/nan.huang/code/vggt/eval/outputs_vggt/videodepth_pretrain/*-metric-*.csv \
  --out_dir figure/plots_videodepth_bedlam_vggt

python vis2.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/pi3*depth*best*/*.csv" \
    /home/nan.huang/code/vggt/eval/outputs/pi3_videodepth_pretrain/*-metric-*.csv \
  --out_dir figure/plots_videodepth_bedlam_pi3
  
python vis2.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/*vdepth*sparse*/*.csv" \
  --out_dir figure/videodepth_pi3

python vis2.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/videodepth*robo*sparse*/*.csv" \
  --out_dir figure/videodepth_robo

python vis2.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/videodepth*hoi4d*sparse*/*.csv" \
    /home/nan.huang/code/vggt/eval/outputs/videodepth_pretrain_sparse/*-metric-*.csv \
  --out_dir figure/videodepth_hoi4d

python vis2.py \
  --inputs \
    $(printf '%s\n' /home/nan.huang/code/vggt/eval/outputs/videodepth**/*.csv | grep -v 'sparse' | grep -v 'dense') \
    /home/nan.huang/code/vggt/eval/outputs/videodepth_pretrain/*-metric-*.csv \
  --out_dir figure/plots_videodepth_ori

python vis2.py \
  --inputs \
    $(printf '%s\n' /home/nan.huang/code/vggt/eval/outputs/videodepth*robo*/*.csv | grep -v 'sparse' | grep -v 'dense') \
    /home/nan.huang/code/vggt/eval/outputs/videodepth_pretrain/*-metric-*.csv \
  --out_dir figure/robo_videodepth_ori

python vis2.py \
  --inputs \
    $(printf '%s\n' /home/nan.huang/code/vggt/eval/outputs/videodepth*sparse*/*.csv | grep -v '99') \
    /home/nan.huang/code/vggt/eval/outputs/videodepth_pretrain_sparse/*-metric-*.csv \
  --out_dir figure/plots_videodepth_ablation

python vis2.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/videodepth*dense*/*.csv" \
    "/home/nan.huang/code/vggt/eval/outputs/videodepth_pretrain_sparse/*.csv" \
  --out_dir figure/plots_videodepth_dense

python vis2.py \
  --inputs "/home/nan.huang/code/vggt/eval/outputs/videodepth*99*sparse*/*.csv" \
    /home/nan.huang/code/vggt/eval/outputs/videodepth_pretrain_sparse/*-metric-*.csv \
  --out_dir figure/plots_videodepth_all
  
python vis2.py \
  --out_dir figure/plots_d_0221\
  --exp_names videodepth_random_99_frcam_sparse videodepth_random_995_frcam_sparse videodepth_bot_995_frcam_prob_sparse videodepth_bot_995_random_frcam videodepth_bot_995_loss_frcam_sparse  videodepth_bot_995_loss_frcam_sparse videodepth_bot_995_frcam_sparse videodepth_top_995_frcam_sparse videodepth_pretrain_sparse

  --exp_names videodepth_exp_frcam_aligns_sparse videodepth_exp_frcam_alignt_sparse videodepth_bot_995_frcam_prob_sparse videodepth_random_1_frcam_sparse videodepth_bot_995_random_frcam videodepth_bot_995_loss_frcam  videodepth_bot_995_loss_frcam videodepth_bot_995_frcam videodepth_pretrain_sparse videodepth_pretrain

  --exp_names videodepth_exp_frcam_aligns_sparse videodepth_exp_frcam_alignt_sparse videodepth_bot_995_frcam_prob_sparse videodepth_random_1_frcam_sparse  videodepth_bot_995_loss_frcam_sparse videodepth_bot_995_frcam_sparse videodepth_pretrain_sparse

  --exp_names videodepth_nnbot_995_cam0_probaug_loss001_sparse videodepth_nnbot_995_nocam_probaug_loss001_sparse videodepth_nnbot_995_nocam_probaug_loss001_sparse videodepth_nnbot_995_nocam_probaug_loss_sparse videodepth_nnbot_995_nocam_probaug_loss_cropsoft_sparse videodepth_nnbot_995_probaug_loss_cropsoft_sparse videodepth_nnbot_995_nocam_probaug_sparse videodepth_nnbot_995_probaug_sparse videodepth_pretrain_sparse videodepth_nnbot_995_droponly_sparse

  --exp_names videodepth_nnbot_995_loss001_cam0_sparse videodepth_nnbot_995_loss3_sparse videodepth_nnbot_995_nocam_sparse videodepth_nnbot_995_loss3_nocam_sparse videodepth_nnbot_995_loss3_nocaml2_sparse videodepth_nnbot_995_nocam_lossl2_noaug_sparse videodepth_nnbot_995_nocam_loss001_sparse videodepth_nnbot_995_nocam_loss001_augall_sparse cam_pretrain_sparse

  --exp_names videodepth_nnbot_995_croponly3_sparse videodepth_nnbot_995_augonly3_sparse videodepth_nnbot_995_droponly_sparse videodepth_nnbot_995_augdrop_sparse videodepth_pretrain_sparse


  --exp_names videodepth_nnbot_995_nocam_probaug_sparse videodepth_nnbot_995_probaug_sparse videodepth_pretrain_sparse


  --exp_names videodepth_nnbot_995_croponly_sparse videodepth_nnbot_995_augonly_sparse videodepth_nnbot_995_droponly_sparse videodepth_nnbot_995_augdrop_sparse videodepth_nnbot_995_nocam_sparse videodepth_nnbot_995_loss3_nocam_sparse videodepth_nnbot_995_loss3_sparse videodepth_pretrain_sparse\

'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path
import pandas as pd
import glob

# Headless-safe backend (on clusters without display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def natural_key(s: str):
    """Sort strings with numbers naturally: checkpoint_2 < checkpoint_10."""
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s)))


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def build_series_color_map(sub: pd.DataFrame):
    """
    Return dict: series_id -> RGBA color

    - student (stu_*): Greens gradient (all green family)
    - others (tea_* / other_*): high-contrast qualitative palette (avoid green/grey)
    NOTE: pretrain is handled separately (fixed grey) in plotting functions.
    """
    series_color = {}

    def _to_rgba(c):
        if len(c) == 3:
            return (c[0], c[1], c[2], 1.0)
        return c

    def _is_grey(c, tol=0.06):
        r, g, b, _ = _to_rgba(c)
        return abs(r - g) < tol and abs(g - b) < tol

    def _is_greenish(c):
        r, g, b, _ = _to_rgba(c)
        # green is clearly the dominant channel
        return (g > 0.45) and (g - r > 0.18) and (g - b > 0.18)

    def _get_qualitative_palette():
        # A long, high-contrast palette pool
        names = ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Dark2", "Accent"]
        pool = []
        for name in names:
            cmap = cm.get_cmap(name)
            if hasattr(cmap, "colors"):
                pool.extend([_to_rgba(c) for c in cmap.colors])
            else:
                # fallback sampling
                n = max(20, getattr(cmap, "N", 256))
                pool.extend([_to_rgba(cmap(i / (n - 1))) for i in range(n)])

        # remove grey-ish and green-ish colors to avoid collision with pretrain/student
        filtered = [c for c in pool if (not _is_grey(c)) and (not _is_greenish(c))]
        return filtered if len(filtered) >= 12 else pool

    def _assign_gradient(group_df: pd.DataFrame, cmap_name: str, lo=0.45, hi=0.90):
        sids = sorted(group_df["series_id"].unique(), key=natural_key)
        n = len(sids)
        cmap = cm.get_cmap(cmap_name)
        if n == 1:
            vals = [(lo + hi) / 2]
        else:
            vals = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        for sid, v in zip(sids, vals):
            series_color[sid] = cmap(v)

    # 1) student -> Greens
    stu_df = sub[sub["exp_type"] == "stu"]
    if not stu_df.empty:
        _assign_gradient(stu_df, "Greens", lo=0.50, hi=0.95)

    # 2) non-student (tea/other) -> colorful palette
    other_df = sub[sub["exp_type"] != "stu"]
    if not other_df.empty:
        palette = _get_qualitative_palette()
        sids = sorted(other_df["series_id"].unique(), key=natural_key)
        for i, sid in enumerate(sids):
            series_color[sid] = palette[i % len(palette)]

    return series_color


def pick_col(df: pd.DataFrame, kind: str) -> str:
    cols = list(df.columns)

    if kind == "abs_rel":
        for c in cols:
            if c.strip().lower().replace(" ", "") in ["absrel", "abs_rel"]:
                return c
        for c in cols:
            if "abs" in c.lower() and "rel" in c.lower():
                return c
        raise KeyError(f"Cannot find Abs Rel column. Columns={cols}")

    if kind == "delta_1_25":
        for c in cols:
            if "1.25" in c and ("δ" in c or "delta" in c.lower()):
                return c
        for c in cols:
            if "1.25" in c:
                return c
        raise KeyError(f"Cannot find δ < 1.25 column. Columns={cols}")

    raise ValueError(kind)


TYPE_ORDER = {"tea": 0, "stu": 1, "pretrain": 2, "other": 3}


def _extract_exp_name_and_ckpt(exp_dir_wo_stu: str):
    """
    exp_dir_wo_stu examples:
      - videodepth_ema999_bs_checkpoint_3
      - videodepth_ema999_bsmn_checkpoint_2
      - videodepth_game_ema999_checkpoint_0
      - camera_pretrain
    Return: (exp_name, ckpt_num, ema_tag, ema_num)
    """
    name = exp_dir_wo_stu

    # strip leading videodepth_ / video_depth_
    m0 = re.match(r"^(?:videodepth|video_depth)_(.+)$", name, flags=re.IGNORECASE)
    core = m0.group(1) if m0 else name

    # extract trailing _checkpoint_<n>
    ckpt_num = None
    m1 = re.search(r"(?:^|_)checkpoint_(\d+)$", core, flags=re.IGNORECASE)
    if m1:
        ckpt_num = int(m1.group(1))
        exp_name = core[: m1.start()]
        exp_name = exp_name.rstrip("_")
    else:
        exp_name = core

    exp_name = exp_name.strip("_")

    ema_tag = None
    ema_num = None
    m2 = re.search(r"(ema\d+)", exp_name, flags=re.IGNORECASE)
    if m2:
        ema_tag = m2.group(1).lower()
        try:
            ema_num = int(ema_tag.replace("ema", ""))
        except Exception:
            ema_num = None

    return exp_name, ckpt_num, ema_tag, ema_num


def parse_exp_info(exp_full: str):
    """
    Rules:
      - if startswith 'stu_' -> student
      - elif contains 'pretrain' -> pretrain
      - else -> teacher
    exp_name = middle part between (videodepth_) and (_checkpoint_N)
    """
    low = exp_full.lower()

    # pretrain first (so it works even if startswith stu_...pretrain)
    if "pretrain" in low:
        exp_type = "pretrain"
        exp_name, ckpt_num, ema_tag, ema_num = _extract_exp_name_and_ckpt(exp_full)
        exp_short = "pretrain"
        series_id = "pretrain"
        return exp_type, exp_name, exp_short, ema_tag, ema_num, ckpt_num, series_id

    if low.startswith("stu_"):
        exp_type = "stu"
        exp_dir_wo_stu = exp_full[4:]
    else:
        exp_type = "tea"
        exp_dir_wo_stu = exp_full

    exp_name, ckpt_num, ema_tag, ema_num = _extract_exp_name_and_ckpt(exp_dir_wo_stu)

    # short label
    if ckpt_num is not None:
        exp_short = f"{exp_type}_{exp_name}_{ckpt_num}"
    else:
        exp_short = f"{exp_type}_{exp_name}"

    # series grouping: same exp_type + same exp_name -> one color/one line
    series_id = f"{exp_type}_{exp_name}" if exp_name else exp_type

    return exp_type, exp_name, exp_short, ema_tag, ema_num, ckpt_num, series_id


def parse_one_csv(p: Path):
    """
    Example:
      /.../outputs_d/videodepth_ema999_bs_checkpoint_3/gamegeo-metric-scale.csv
        exp_dir  = videodepth_ema999_bs_checkpoint_3
        exp_name = ema999_bs
        dataset  = gamegeo
        align    = scale
    """
    exp_full = p.parent.name
    exp_type, exp_name, exp_short, ema_tag, ema_num, ckpt_num, series_id = parse_exp_info(exp_full)

    m = re.match(r"^(?P<dataset>[^-]+)-metric-(?P<align>.+)\.csv$", p.name)
    if not m:
        raise ValueError(f"Filename not match '<dataset>-metric-<align>.csv': {p.name}")

    dataset = m.group("dataset")
    align = m.group("align")

    df = pd.read_csv(p)
    abs_col = pick_col(df, "abs_rel")
    d125_col = pick_col(df, "delta_1_25")

    abs_rel = float(df[abs_col].mean())
    d125 = float(df[d125_col].mean())

    return {
        "path": str(p),
        "exp_full": exp_full,
        "exp_name": exp_name,
        "exp_short": exp_short,
        "exp_type": exp_type,
        "exp_type_order": TYPE_ORDER.get(exp_type, 999),
        "ema_tag": ema_tag,
        "ema_num": ema_num,
        "ckpt_num": ckpt_num,
        "dataset": dataset,
        "align": align,
        "Abs Rel": abs_rel,
        "δ < 1.25": d125,
        "series_id": series_id,
    }


def plot_line(df_all: pd.DataFrame, out_dir: Path):
    datasets = sorted(df_all["dataset"].unique(), key=natural_key)
    aligns = sorted(df_all["align"].unique(), key=natural_key)
    metrics = ["Abs Rel", "δ < 1.25"]

    for dataset in datasets:
        for align in aligns:
            sub = df_all[(df_all["dataset"] == dataset) & (df_all["align"] == align)].copy()
            if sub.empty:
                continue

            sub_pre = sub[sub["exp_type"] == "pretrain"].copy()
            sub_main = sub[sub["exp_type"] != "pretrain"].copy()

            # ckpt axis
            sub_main = sub_main[sub_main["ckpt_num"].notna()].copy()
            if not sub_main.empty:
                sub_main["ckpt_num"] = sub_main["ckpt_num"].astype(int)
                ckpts = sorted(sub_main["ckpt_num"].unique())
            else:
                ckpts = list(range(10))

            xticks = ckpts
            xticklabels = [f"{k}" for k in ckpts]

            # colors for main curves; pretrain fixed green
            series_color = build_series_color_map(sub_main) if not sub_main.empty else {}

            # avoid overlap at same ckpt: small offsets per exp_type within each exp_type
            offsets = {}
            if not sub_main.empty:
                for exp_type, g in sub_main.groupby("exp_type"):
                    series_list = sorted(g["series_id"].unique(), key=natural_key)
                    n = len(series_list)
                    if n == 1:
                        offset_list = [0.0]
                    else:
                        span = 0.45
                        step = span / (n - 1)
                        offset_list = [-span / 2 + i * step for i in range(n)]
                    for sid, off in zip(series_list, offset_list):
                        offsets[(exp_type, sid)] = off

                sub_main["x"] = sub_main.apply(
                    lambda r: r["ckpt_num"] + offsets.get((r["exp_type"], r["series_id"]), 0.0),
                    axis=1,
                )

            for metric in metrics:
                plt.figure(figsize=(10, 4.5))

                # main curves
                if not sub_main.empty:
                    for (exp_type, sid), g in sub_main.groupby(["exp_type", "series_id"], sort=False):
                        g = g.sort_values("ckpt_num")
                        plt.plot(
                            g["x"].tolist(),
                            g[metric].tolist(),
                            marker="o",
                            linewidth=1.5,
                            color=series_color.get(sid, "0.4"),
                            alpha=0.95,
                            label=sid,
                        )

                # pretrain horizontal line
                if not sub_pre.empty:
                    y = float(sub_pre[metric].mean())
                    plt.plot(
                        ckpts,
                        [y] * len(ckpts),
                        color="0.6",   # grey
                        linewidth=2.2,
                        linestyle="-",
                        alpha=0.95,
                        label="pretrain",
                    )

                plt.title(f"{dataset} + {align} + {metric}")
                plt.xlabel("checkpoint")
                plt.ylabel(metric)
                plt.xticks(xticks, xticklabels)
                plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

                n_series = 0 if sub_main.empty else sub_main["series_id"].nunique()
                n_legend = n_series + (0 if sub_pre.empty else 1)
                if n_legend > 0:
                    if n_legend <= 12:
                        plt.legend(frameon=True, fontsize=9)
                    else:
                        plt.legend(frameon=True, fontsize=8, ncol=2)

                plt.tight_layout()
                fname = f"line__{dataset}__{sanitize_filename(align)}__{sanitize_filename(metric)}.png"
                plt.savefig(out_dir / fname, dpi=200)
                plt.close()


def plot_scatter(df_all: pd.DataFrame, out_dir: Path, annotate: bool = True):
    datasets = sorted(df_all["dataset"].unique(), key=natural_key)
    aligns = sorted(df_all["align"].unique(), key=natural_key)

    for dataset in datasets:
        for align in aligns:
            sub = df_all[(df_all["dataset"] == dataset) & (df_all["align"] == align)].copy()
            if sub.empty:
                continue

            sort_cols = ["exp_type_order", "ema_num", "ckpt_num", "exp_short"]
            sub = sub.sort_values(sort_cols, na_position="last").reset_index(drop=True)

            sub_main = sub[sub["exp_type"] != "pretrain"].copy()
            series_color = build_series_color_map(sub_main) if not sub_main.empty else {}

            def point_color(row):
                if row["exp_type"] == "pretrain":
                    return "0.6"
                return series_color.get(row["series_id"], "0.4")

            plt.figure(figsize=(6.5, 5.5))

            for _, r in sub.iterrows():
                c = point_color(r)
                plt.scatter(r["Abs Rel"], r["δ < 1.25"], color=c)

                if annotate and pd.notna(r.get("ckpt_num")):
                    plt.annotate(
                        str(int(r["ckpt_num"])),
                        (r["Abs Rel"], r["δ < 1.25"]),
                        textcoords="offset points",
                        xytext=(5, 3),
                        fontsize=8,
                        color=c,
                        alpha=0.95,
                    )

            plt.title(f"Scatter: {dataset} + {align}\n(x=Abs Rel, y=δ < 1.25)")
            plt.xlabel("Abs Rel (lower is better)")
            plt.ylabel("δ < 1.25 (higher is better)")
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

            # legend by series_id + pretrain
            handles = []
            labels = []
            if not sub_main.empty:
                for sid in sorted(sub_main["series_id"].unique(), key=natural_key):
                    handles.append(plt.Line2D([0], [0], marker='o', linestyle='',
                                              color=series_color.get(sid, "0.4")))
                    labels.append(sid)
            if (sub["exp_type"] == "pretrain").any():
                handles.append(plt.Line2D([0], [0], marker='o', linestyle='', color="0.6"))
                labels.append("pretrain")

            if len(labels) > 0:
                if len(labels) <= 12:
                    plt.legend(handles, labels, frameon=True, fontsize=9)
                else:
                    plt.legend(handles, labels, frameon=True, fontsize=8, ncol=2)

            plt.tight_layout()
            fname = f"scatter__{dataset}__{sanitize_filename(align)}__absrel_vs_d125.png"
            plt.savefig(out_dir / fname, dpi=200)
            plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="CSV paths or glob patterns, e.g. '/.../outputs_d/*/*-metric-*.csv'. "
             "If a directory is provided, we will search '**/*.csv' inside it.",
    )
    ap.add_argument(
        "--exp_names",
        nargs="+",
        default=None,
        help="Experiment name keywords. If provided (and --inputs not provided), "
             "expand to '<outputs_root>/*{exp}*/*-metric-*.csv' for each exp.",
    )
    ap.add_argument(
        "--outputs_root",
        default="/home/nan.huang/code/vggt/eval/outputs",
        help="Root folder containing experiment output dirs (used with --exp_names).",
    )
    ap.add_argument("--out_dir", default="plots_outputs_d", help="Output dir for pngs")
    ap.add_argument("--no_annotate", action="store_true", help="Disable point labels on scatter plots")
    args = ap.parse_args()

    # 1) Choose input items: prefer --inputs, else use --exp_names
    if args.inputs:
        input_items = args.inputs
    elif args.exp_names:
        root = Path(args.outputs_root)
        input_items = []
        for name in args.exp_names:
            low = name.lower()

            # 1) 如果用户直接给了带 checkpoint 号的完整目录名：xxx_checkpoint_17
            if re.search(r"_checkpoint_\d+$", name, flags=re.IGNORECASE):
                input_items.append(str(root / name / "*-metric-*.csv"))
                continue

            # 2) pretrain 这种通常没有 checkpoint 后缀
            if "pretrain" in low:
                input_items.append(str(root / name / "*-metric-*.csv"))
                continue

            # 3) 常规：严格匹配 <exp_name>_checkpoint_*
            input_items.append(str(root / f"{name}_checkpoint_*/*-metric-*.csv"))
    else:
        raise SystemExit("You must provide either --inputs or --exp_names.")

    # 2) Expand globs / dirs / files
    paths = []
    for item in input_items:
        # glob pattern
        if any(ch in item for ch in ["*", "?", "["]):
            paths.extend([Path(p) for p in glob.glob(item, recursive=True)])
            continue

        p = Path(item)
        # directory -> search recursively for csv
        if p.exists() and p.is_dir():
            paths.extend([Path(x) for x in glob.glob(str(p / "**" / "*.csv"), recursive=True)])
        else:
            # file path
            paths.append(p)

    # 3) Filter out per-seq csvs + keep only *-metric-*.csv
    def is_valid_metric_csv(p: Path) -> bool:
        name = p.name.lower()
        if not p.is_file():
            return False
        if not name.endswith(".csv"):
            return False
        if "per-seq" in name or "per_seq" in name or "perseq" in name:
            return False
        return "-metric-" in name

    paths = [p for p in paths if is_valid_metric_csv(p)]
    paths = sorted(paths, key=lambda x: natural_key(str(x)))

    if not paths:
        raise SystemExit("No csv files found. Check your --inputs / --exp_names / --outputs_root.")

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

    df_all.sort_values(
        ["dataset", "align", "exp_type_order", "ema_num", "ckpt_num", "exp_short"],
        na_position="last",
        key=lambda s: s.map(lambda x: natural_key(str(x))),
    ).to_csv(out_dir / "merged_absrel_d125.csv", index=False)

    plot_line(df_all, out_dir)
    plot_scatter(df_all, out_dir, annotate=(not args.no_annotate))

    print(f"Done. Outputs saved to: {out_dir.resolve()}")
    print(f"Merged table: {(out_dir / 'merged_absrel_d125.csv').resolve()}")


if __name__ == "__main__":
    main()

# python vis2.py --inputs "/home/nan.huang/code/vggt/eval/outputs/videodepth*sparse*/*-metric-*.csv" --out_dir plots_d_d