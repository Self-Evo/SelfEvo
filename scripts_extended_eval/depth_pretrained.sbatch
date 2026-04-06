#!/bin/bash
#SBATCH -J vggt_depth_pre_ext
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


eval "$(conda shell.bash hook)"
conda activate vggt
cd ~/code/eval

export PYTHONPATH=/home/nan.huang/code/vggt:$PYTHONPATH

DATASETS='[sintel, bonn, kitti, droid]'
CKPT_PATH="/home/nan.huang/code/vggt/ckpt/model.pt"
EXP_NAME="vggt-pretrained"
OUTPUT_DIR="/home/nan.huang/code/eval/0_res_extended/${EXP_NAME}"

echo "============================================================"
echo "[INFO] Running vggt-pretrained depth eval on sintel/bonn/kitti/droid"
echo "[INFO] ckpt: ${CKPT_PATH}"
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
