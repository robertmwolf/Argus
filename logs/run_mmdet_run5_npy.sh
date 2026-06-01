#!/bin/bash
# MMDetection DINO ViT-S — Run 5 resumed with .npy tiles
#
# Resumes from epoch_2.pth using pre-converted .npy tiles instead of raw FITS.
# Expected speedup: data_time ~1.3s → ~0.1s (~10x), cutting epoch time by ~50%.
#
# Annotation: all_train_run5_tiled_npy.json
#   9,115 Atwood tiles as .npy (raw float32, dual-norm works)
#   380   synthetic short-band PNGs (unchanged)
#   0     Frigate (stripped — doubly-virtual path bug, fix in Run 6)
#   = 9,495 total tiles
#
# Evaluate at epoch 7 for diminishing-returns check.
set -euo pipefail

cd /Users/robert/Argus

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python

echo "[$(date)] Resuming OBB Run 5 with .npy tiles from epoch_2.pth"

PYTORCH_ENABLE_MPS_FALLBACK=1 \
USE_DEV_SUBSET=false \
TRAIN_ANN_FILE=/Volumes/External/TrainingData/annotations/all_train_run5_tiled_npy.json \
VAL_ANN_FILE=/Volumes/External/TrainingData/annotations/val_atwood_tiled_400.json \
ARGUS_NORM=zscore \
ARGUS_ENABLE_PLATE_SOLVE=false \
caffeinate -i "$PYTHON" -m training.train_dino \
  --config models/dino/streak_dinov3_vits_400px_run5.py \
  --work-dir weights/run5_vits_mmdet \
  --resume \
  --val-interval 1 \
  --checkpoint-interval 1

echo "[$(date)] OBB Run 5 (.npy) training complete"
