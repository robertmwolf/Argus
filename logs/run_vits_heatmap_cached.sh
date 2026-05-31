#!/bin/bash
# ViT-S/16 cached heatmap training — backbone comparison vs ConvNeXt-S
#
# Mirrors run5_convnext_small_s2_heatmap_pretiled exactly:
#   same annotations (all_train_run5.json / val_atwood.json)
#   same tile params (native_tile_size=400, tile_overlap=0.5)
#   same area-fraction filter (min_area_fraction=0.25, 2 neg tiles/img)
#   same head training (50 epochs, batch_size=32, AdamW)
#
# Only difference: backbone = ViT-S/16 instead of ConvNeXt-S stage-2
#
# Purpose: isolate backbone variable for ConvNeXt vs ViT-S comparison.
# Winner goes into the final comparison against the ViT-S OBB (Run 5).
#
# ARGUS_ENABLE_PLATE_SOLVE=false is set; ViT-S backbone is frozen so
# no plate solve can be triggered during training.
set -euo pipefail

cd /Users/robert/Argus

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python

TRAIN_CACHE=/Volumes/External/TrainingData/heatmap_cache/vits_pretiled_train
VAL_CACHE=/Volumes/External/TrainingData/heatmap_cache/vits_pretiled_val
WORK_DIR=weights/run5_vits_heatmap_cached

echo "[$(date)] Step 1/3 — Caching ViT-S/16 train features (9,495 tiles expected)"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
"$PYTHON" scripts/cache_dinov3_heatmap_features.py \
  --annotations data/annotations/all_train_run5.json \
  --output-dir  "$TRAIN_CACHE" \
  --backbone vit --model-size small \
  --image-size 384 \
  --native-tile-size 400 \
  --tile-overlap 0.5 \
  --min-area-fraction 0.25 \
  --neg-tiles-per-image 2 \
  --batch-size 4

echo "[$(date)] Step 2/3 — Caching ViT-S/16 val features"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
"$PYTHON" scripts/cache_dinov3_heatmap_features.py \
  --annotations data/annotations/val_atwood.json \
  --output-dir  "$VAL_CACHE" \
  --backbone vit --model-size small \
  --image-size 384 \
  --native-tile-size 400 \
  --tile-overlap 0.5 \
  --min-area-fraction 0.25 \
  --neg-tiles-per-image 2 \
  --batch-size 4

echo "[$(date)] Step 3/3 — Training ViT-S heatmap head (50 epochs, batch=32)"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
"$PYTHON" -m training.train_dinov3_heatmap_cached \
  --train-cache "$TRAIN_CACHE" \
  --val-cache   "$VAL_CACHE" \
  --work-dir    "$WORK_DIR" \
  --epochs 50 \
  --batch-size 32

echo "[$(date)] ViT-S cached heatmap training complete — weights at $WORK_DIR/best.pt"
