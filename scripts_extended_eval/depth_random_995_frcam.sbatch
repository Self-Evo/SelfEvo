#!/bin/bash
#SBATCH -J vggt_depth_random_ext
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --array=0-19


eval "$(conda shell.bash hook)"
conda activate vggt
cd ~/code/eval

export PYTHONPATH=/home/nan.huang/code/vggt:$PYTHONPATH

CKPT_DIR="/home/nan.huang/code/vggt/training/logs/random_995_frcam/ckpts"
DATASETS='[sintel, bonn, kitti, droid]'

shopt -s nullglob
ckpts=( "${CKPT_DIR}"/checkpoint_*.pt )
mapfile -t ckpts_sorted < <(printf "%s\n" "${ckpts[@]}" | sort -V)

IDX="${SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge "${#ckpts_sorted[@]}" ]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=$IDX out of range"
  exit 1
fi

CKPT_PATH="${ckpts_sorted[$IDX]}"
CKPT_BASENAME="$(basename "${CKPT_PATH}" .pt)"
EXP_NAME="vdepth_random_995_frcam_${CKPT_BASENAME}"
OUTPUT_DIR="/home/nan.huang/code/eval/0_res_extended/${EXP_NAME}"

echo "============================================================"
echo "[INFO] task_id=${IDX} | ckpt: ${CKPT_PATH}"
echo "============================================================"

python videodepth/infer.py name="${EXP_NAME}" \
  pi3.pretrained_model_name_or_path="${CKPT_PATH}" \
  eval_datasets="${DATASETS}" model=vggt \
  output_dir="${OUTPUT_DIR}"

python videodepth/eval.py name="${EXP_NAME}" \
  pi3.pretrained_model_name_or_path="${CKPT_PATH}" \
  eval_datasets="${DATASETS}" model=vggt \
  output_dir="${OUTPUT_DIR}"

python videodepth/eval.py align=scale name="${EXP_NAME}" \
  pi3.pretrained_model_name_or_path="${CKPT_PATH}" \
  eval_datasets="${DATASETS}" model=vggt \
  output_dir="${OUTPUT_DIR}"
