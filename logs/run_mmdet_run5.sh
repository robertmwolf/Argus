#!/bin/bash
# MMDetection DINO ViT-S training run — Run 5 (native-scale tiled Atwood)
#
# Key change vs Run 4: Atwood training images are native-scale 400px tiles
# (built by scripts/build_tiled_brentimages_json.py) instead of full-frame
# FITS resized to 400px. Train and inference now see the same pixel scale.
#
# Train:  all_train_run5_tiled.json
#           6,387 Atwood native-scale tiles (400px, overlap=0.50)
#         +    75 Frigate cluster-2 tiles (110px → 400px, 3.64× magnification)
#         +   380 synthetic short-band (400px PNG)
#         = 6,842 total tiles / 6,664 annotations
#
# Val:    val_atwood_tiled_400.json
#           1,384 Atwood native-scale tiles (240 source images)
#
# Inference: tiled at 400px, overlap=0.50 (matches training distribution)
#
# Started by Claude Code — runs unsupervised
set -euo pipefail

cd /Users/robert/Argus

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python

echo "[$(date)] Starting MMDetection ViT-S Run 5 (native-scale tiled Atwood)."

PYTORCH_ENABLE_MPS_FALLBACK=1 \
USE_DEV_SUBSET=false \
TRAIN_ANN_FILE=/Volumes/External/TrainingData/annotations/all_train_run5_tiled.json \
VAL_ANN_FILE=/Volumes/External/TrainingData/annotations/val_atwood_tiled_400.json \
ARGUS_NORM=zscore \
ARGUS_ENABLE_PLATE_SOLVE=false \
caffeinate -i "$PYTHON" -m training.train_dino \
  --config models/dino/streak_dinov3_vits_400px_run5.py \
  --work-dir weights/run5_vits_mmdet \
  --val-interval 1 \
  --checkpoint-interval 1

echo "[$(date)] MMDetection ViT-S Run 5 training complete"
