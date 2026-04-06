#!/bin/bash
cd /home/nan.huang/code/eval
mkdir -p logs

# Depth eval on sintel/bonn/kitti/droid
sbatch scripts_extended_eval/depth_pretrained.sbatch           # pretrained (1 job)
sbatch scripts_extended_eval/depth_alldata_game_eval.sbatch    # alldata_game_eval (20 array)
sbatch scripts_extended_eval/depth_random_995_frcam.sbatch     # random_995_frcam (20 array)

# Camera eval on Re10K
sbatch scripts_extended_eval/cam_pretrained.sbatch             # pretrained (1 job)
sbatch scripts_extended_eval/cam_alldata_game_eval.sbatch      # alldata_game_eval (20 array)
sbatch scripts_extended_eval/cam_random_995_frcam.sbatch       # random_995_frcam (20 array)
