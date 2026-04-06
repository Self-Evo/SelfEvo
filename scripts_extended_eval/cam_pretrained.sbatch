#!/bin/bash
#SBATCH -J vggt_cam_pre_ext
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


eval "$(conda shell.bash hook)"
conda activate vggt
cd ~/code/eval/relpose

export PYTHONPATH=/home/nan.huang/code/vggt:$PYTHONPATH

DATASETS='[Re10K]'
CKPT_PATH="/home/nan.huang/code/vggt/ckpt/model.pt"
SAVE_SUFFIX="vggt-pretrained"
OUTPUT_DIR="/home/nan.huang/code/eval/0_res_extended/camera"

echo "============================================================"
echo "[INFO] Running vggt-pretrained camera eval on Re10K"
echo "[INFO] ckpt: ${CKPT_PATH}"
echo "============================================================"

python eval_angle.py model=vggt \
  vggt.pretrained_model_name_or_path="${CKPT_PATH}" \
  vggt.use_ema=false eval_datasets="${DATASETS}" save_suffix=${SAVE_SUFFIX} \
  output_dir=${OUTPUT_DIR}
