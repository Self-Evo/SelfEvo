#!/bin/bash
#SBATCH -J vggt_cam_random_ext
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
cd ~/code/eval/relpose

export PYTHONPATH=/home/nan.huang/code/vggt:$PYTHONPATH

CKPT_DIR="/home/nan.huang/code/vggt/training/logs/random_995_frcam/ckpts"
DATASETS='[Re10K]'
OUTPUT_DIR="/home/nan.huang/code/eval/0_res_extended/camera"

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
SAVE_SUFFIX="random_995_frcam_${CKPT_BASENAME}"

echo "============================================================"
echo "[INFO] task_id=${IDX} | ckpt: ${CKPT_PATH}"
echo "============================================================"

python eval_angle.py model=vggt \
  vggt.pretrained_model_name_or_path="${CKPT_PATH}" \
  vggt.use_ema=true eval_datasets="${DATASETS}" save_suffix=${SAVE_SUFFIX} \
  output_dir=${OUTPUT_DIR}
