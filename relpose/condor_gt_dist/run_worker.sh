#!/bin/bash
# Condor executable: run worker_gt_dist.py for a single BEDLAM2 session.
# Usage: bash run_worker.sh <session_name>

set -u

SESSION="$1"

echo "============================================================"
echo "[INFO] Session    : ${SESSION}"
echo "[INFO] Start time : $(date)"
echo "[INFO] Hostname   : $(hostname)"
echo "============================================================"

# ---- Environment setup (matches 0_condor_tasks/run_eval_single_vggt_bot.sh) ----
export PATH="/usr/bin:${PATH}"
source /home/pyu/miniforge3/etc/profile.d/conda.sh
module load cuda/12.1
conda activate pi3

export HF_HOME='/is/cluster/pyu/.cache/huggingface'
export LD_LIBRARY_PATH=/is/software/nvidia/cuda-12.1/lib64
export PATH=$PATH:/is/software/nvidia/cuda-12.1/bin
export CUDA_HOME=/is/software/nvidia/cuda-12.1

# Python paths
REPO_ROOT="/is/cluster/fast/groups/ps-pbrgaussians/Uncharted4d/data/Un4D-all"
export PYTHONPATH="${REPO_ROOT}/vggt:${REPO_ROOT}/uncharted4d-Pi3:${PYTHONPATH:-}"

EVAL_DIR="${REPO_ROOT}/eval"
cd "${EVAL_DIR}"

python relpose/condor_gt_dist/worker_gt_dist.py \
    --session "${SESSION}" \
    --bedlam2-dir "/is/cluster/fast/groups/ps-pbrgaussians/Uncharted4d/data/Bedlamv2" \
    --test-selection "${EVAL_DIR}/bedlam2_test_selection_20seq_seed42.json" \
    --output-dir "${EVAL_DIR}/relpose/condor_gt_dist/results"

RC=$?

echo ""
echo "============================================================"
echo "[INFO] Done: ${SESSION}"
echo "[INFO] Exit code : ${RC}"
echo "[INFO] End time  : $(date)"
echo "============================================================"

exit ${RC}
