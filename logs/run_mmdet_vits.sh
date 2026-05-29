#!/bin/bash
# MMDetection DINO ViT-S training run — Run 4 (geometry-stratified, no SatStreaks)
# Train:  all_train_run4.json (618 Atwood geometry-stratified + 250 Frigate diversity)
# Val:    val_atwood.json     (133 Atwood geometry-stratified val split)
# Started by Claude Code — runs unsupervised
set -euo pipefail

cd /Users/robert/Argus

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python

echo "[$(date)] Starting MMDetection ViT-S training."

PYTORCH_ENABLE_MPS_FALLBACK=1 \
USE_DEV_SUBSET=false \
TRAIN_ANN_FILE=/Volumes/External/TrainingData/annotations/all_train_run4.json \
VAL_ANN_FILE=/Volumes/External/TrainingData/annotations/val_atwood.json \
ARGUS_NORM=zscore \
ARGUS_ENABLE_PLATE_SOLVE=false \
caffeinate -i "$PYTHON" -m training.train_dino \
  --config models/dino/streak_dinov3_vits_400px_run3.py \
  --work-dir weights/run4_vits_mmdet \
  --val-interval 1 \
  --checkpoint-interval 1

echo "[$(date)] MMDetection ViT-S training complete"
