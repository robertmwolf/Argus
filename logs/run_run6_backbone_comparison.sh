#!/bin/bash
# Run 6 — Hard-Negative Backbone Comparison: ConvNeXt-S vs ViT-S
#
# Root cause fix for Run 5 precision catastrophe (0.05% precision):
#   - 59% negative tile ratio (was 3% in Run 5)
#   - Hard negatives from positive images (--hard-neg-per-pos 5)
#   - Frigate cluster-2 from raw FITS (fixes doubly-virtual path bug)
#   - NPY pre-extraction for correct tile loading (fixes blank-image bug)
#
# Caching: --native-tile-size 0 against the NPY annotation (tiles are
# pre-extracted; no on-the-fly tiling needed). Both backbones use identical
# hyperparameters for a controlled comparison.
#
# Success gate: precision > 10% at recall >= 60% on test_atwood.json.
# Winning backbone proceeds to ViT-B / ViT-L for the paper run.
set -euo pipefail

cd /Users/robert/Argus

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python

TRAIN_NPY=/Volumes/External/TrainingData/annotations/all_train_run6_tiled_npy.json
VAL_NPY=/Volumes/External/TrainingData/annotations/val_atwood_tiled_400_npy.json

CONVNEXT_CACHE=/Volumes/External/TrainingData/heatmap_cache/convnext_run6_train
VITS_CACHE=/Volumes/External/TrainingData/heatmap_cache/vits_run6_train
VAL_CONVNEXT_CACHE=/Volumes/External/TrainingData/heatmap_cache/convnext_run6_val
VAL_VITS_CACHE=/Volumes/External/TrainingData/heatmap_cache/vits_run6_val

CONVNEXT_WORK=weights/run6_convnext_heatmap
VITS_WORK=weights/run6_vits_heatmap

# ---------------------------------------------------------------------------
# Step 1: Cache ConvNeXt-S train features
# ---------------------------------------------------------------------------
echo "[$(date)] Step 1/6 — Caching ConvNeXt-S train features (23,886 tiles)"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
"$PYTHON" scripts/cache_dinov3_heatmap_features.py \
  --annotations  "$TRAIN_NPY" \
  --output-dir   "$CONVNEXT_CACHE" \
  --backbone convnext --model-size small --convnext-stage 2 \
  --image-size 384 \
  --batch-size 4

# ---------------------------------------------------------------------------
# Step 2: Cache ConvNeXt-S val features
# ---------------------------------------------------------------------------
echo "[$(date)] Step 2/6 — Caching ConvNeXt-S val features"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
"$PYTHON" scripts/cache_dinov3_heatmap_features.py \
  --annotations  "$VAL_NPY" \
  --output-dir   "$VAL_CONVNEXT_CACHE" \
  --backbone convnext --model-size small --convnext-stage 2 \
  --image-size 384 \
  --batch-size 4

# ---------------------------------------------------------------------------
# Step 3: Train ConvNeXt-S head
# ---------------------------------------------------------------------------
echo "[$(date)] Step 3/6 — Training ConvNeXt-S heatmap head (50 epochs)"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
"$PYTHON" -m training.train_dinov3_heatmap_cached \
  --train-cache "$CONVNEXT_CACHE" \
  --val-cache   "$VAL_CONVNEXT_CACHE" \
  --work-dir    "$CONVNEXT_WORK" \
  --epochs 50 \
  --batch-size 32

echo "[$(date)] ConvNeXt-S done — weights at $CONVNEXT_WORK/best.pt"

# ---------------------------------------------------------------------------
# Step 4: Cache ViT-S train features
# ---------------------------------------------------------------------------
echo "[$(date)] Step 4/6 — Caching ViT-S/16 train features (23,886 tiles)"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
"$PYTHON" scripts/cache_dinov3_heatmap_features.py \
  --annotations  "$TRAIN_NPY" \
  --output-dir   "$VITS_CACHE" \
  --backbone vit --model-size small \
  --image-size 384 \
  --batch-size 4

# ---------------------------------------------------------------------------
# Step 5: Cache ViT-S val features
# ---------------------------------------------------------------------------
echo "[$(date)] Step 5/6 — Caching ViT-S/16 val features"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
"$PYTHON" scripts/cache_dinov3_heatmap_features.py \
  --annotations  "$VAL_NPY" \
  --output-dir   "$VAL_VITS_CACHE" \
  --backbone vit --model-size small \
  --image-size 384 \
  --batch-size 4

# ---------------------------------------------------------------------------
# Step 6: Train ViT-S head
# ---------------------------------------------------------------------------
echo "[$(date)] Step 6/6 — Training ViT-S heatmap head (50 epochs)"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
"$PYTHON" -m training.train_dinov3_heatmap_cached \
  --train-cache "$VITS_CACHE" \
  --val-cache   "$VAL_VITS_CACHE" \
  --work-dir    "$VITS_WORK" \
  --epochs 50 \
  --batch-size 32

echo "[$(date)] ViT-S done — weights at $VITS_WORK/best.pt"

# ---------------------------------------------------------------------------
# Step 7: Evaluate ConvNeXt-S on test_atwood.json
#
# CRITICAL — tile size MUST match training (400px / overlap=0.5).
# The detector defaults to CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=1562, which
# mismatches training scale and produces wrong FP/FN counts.
# ---------------------------------------------------------------------------
echo "[$(date)] Step 7/8 — Evaluating ConvNeXt-S on test_atwood.json (tile=400, overlap=0.5)"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
CONVNEXT_HEATMAP_CHECKPOINT="$CONVNEXT_WORK/best.pt" \
CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=400 \
CONVNEXT_HEATMAP_TILE_OVERLAP=0.5 \
CONVNEXT_HEATMAP_THRESHOLD=0.5 \
"$PYTHON" scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$CONVNEXT_WORK/best.pt" \
  --output      results/run6_convnext_test_atwood/metrics.json \
  --tiled \
  --threshold 0.5

# ---------------------------------------------------------------------------
# Step 8: Evaluate ViT-S on test_atwood.json
# ---------------------------------------------------------------------------
echo "[$(date)] Step 8/8 — Evaluating ViT-S on test_atwood.json (tile=400, overlap=0.5)"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_ENABLE_PLATE_SOLVE=false \
VITS_HEATMAP_NATIVE_TILE_SIZE=400 \
VITS_HEATMAP_TILE_OVERLAP=0.5 \
"$PYTHON" scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$VITS_WORK/best.pt" \
  --output      results/run6_vits_test_atwood/metrics.json \
  --tiled \
  --threshold 0.5

echo ""
echo "=== Run 6 complete ==="
echo "Results:"
echo "  ConvNeXt-S: results/run6_convnext_test_atwood/metrics.json"
echo "  ViT-S:      results/run6_vits_test_atwood/metrics.json"
echo ""
echo "Success gate: precision > 10% at recall >= 60%"
echo "Winning backbone proceeds to ViT-B / ViT-L for the paper run."
