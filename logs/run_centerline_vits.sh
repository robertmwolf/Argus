#!/bin/bash
# Orientation-centerline ViT-S training run — Run 4 (geometry-stratified, no SatStreaks)
# Train:  all_train_run4.json (618 Atwood geometry-stratified + 250 Frigate diversity)
# Val:    val_atwood.json     (133 Atwood geometry-stratified val split)
# Started by Claude Code — runs unsupervised
set -euo pipefail

cd /Users/robert/Argus

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python

echo "[$(date)] Starting orientation-centerline ViT-S training"

PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
caffeinate -i "$PYTHON" training/train_dinov3_orientation_centerline.py \
  --train-annotations /Volumes/External/TrainingData/annotations/all_train_run4.json \
  --val-annotations /Volumes/External/TrainingData/annotations/val_atwood.json \
  --work-dir weights/run_dinov3_vits_orientation_centerline_1024 \
  --weights weights/dinov3_vits16_lvd1689m.pth \
  --model-size small \
  --tile-size 2560 \
  --image-size 1024 \
  --positive-train-tiles 1236 \
  --negative-train-tiles 1400 \
  --orientation-bins 18 \
  --decoder-channels 192 \
  --last-layers 4 \
  --centerline-width 2.0 \
  --centerline-sigma 1.4 \
  --catchment-width 14.0 \
  --catchment-sigma 6.0 \
  --neighbor-bin-weight 0.35 \
  --second-neighbor-weight 0.0 \
  --epochs 10 \
  --batch-size 1 \
  --lr 5e-5 \
  --min-lr 1e-5 \
  --weight-decay 1e-4 \
  --pos-weight 60 \
  --dice-weight 1.0 \
  --bce-weight 0.10 \
  --orientation-ce-weight 0.20 \
  --manual-positive-weight 3.0 \
  --catchment-loss-weight 0.35 \
  --catchment-pos-weight 20 \
  --catchment-dice-weight 1.0 \
  --catchment-bce-weight 0.20 \
  --image-loss-weight 0.25 \
  --workers 0 \
  --seed 20260524 \
  --preserve-image-bit-depth \
  --log-interval 200

echo "[$(date)] Orientation-centerline ViT-S training complete"
