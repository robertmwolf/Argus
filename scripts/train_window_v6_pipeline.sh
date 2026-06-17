#!/usr/bin/env bash
# Train + eval the heatmap head with aggressive hard negative mining (window_v6).
#
# v6 changes vs v5:
#   - Mine with vits_window_v5 as oracle (stronger model → remaining FPs are harder).
#   - Lower mining threshold 0.85 → 0.70: captures borderline FPs, not just confident ones.
#   - Random tile sampling (30/frame) instead of exhaustive grid (~150/frame): 5× faster,
#     broader spatial coverage, and includes the new 20260530_Atwood frames.
#   - Hard neg budget raised from 1/3 → 1/2 of total negatives.
#   - ViT-S only (same as v5).
#
# Usage:
#   bash scripts/train_window_v6_pipeline.sh 2>&1 | tee /tmp/window_v6_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DATA=/Volumes/External/TrainingData
BAL_ANN=$REPO/data/annotations/val_balanced_v1.json
MERGED_ANN=$REPO/data/annotations/all_train_run17_merged.json
HARD_NEGS=$REPO/data/annotations/hard_negatives_vits_window_v5.json
OUT=$REPO/results/window_v6
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

# Focal loss / early stopping (same as v5)
FOCAL_GAMMA=2.0; FOCAL_ALPHA=0.85; EARLY_STOP=10
LR=1e-3; BATCH=32; GEOMW=0.25; HIDDEN=256; SCHED=cosine
VITS_W=$WEIGHTS/dinov3_vits16_lvd1689m.pth
CACHE_ROOT=/Volumes/External/argus_caches

echo "=== Window-v6 pipeline | $(date) | aggressive hard-neg mining ==="

# ── Step 1: Mine hard negatives using v5 as oracle ────────────────────────────
echo "── Step 1: Mining hard negatives (oracle=v5, threshold=0.70, random-tiles=30) ── $(date)"
$PYTHON scripts/mine_hard_negatives.py \
  --checkpoint "$WEIGHTS/vits_window_v5/best.pt" \
  --annotations "$MERGED_ANN" \
  --output "$HARD_NEGS" \
  --peak-threshold 0.70 \
  --margin 400 \
  --tile-size 400 \
  --random-tiles 30 \
  --max-hard-negs 1000
echo "── Mining complete; hard negs saved to $HARD_NEGS ── $(date)"

# ── Step 2: Build v6 dataset ──────────────────────────────────────────────────
echo "── Step 2: Building v6 dataset (hard-neg budget=1/2) ── $(date)"
$PYTHON scripts/build_atwood_window_dataset.py \
  --dataset-root "$DATA" \
  --version 6 \
  --source "$MERGED_ANN" \
  --hard-negs-json "$HARD_NEGS" \
  --neg-frac 0.42 \
  --bg-per-frame 3 \
  --seed 19
TRAIN_ANN="$DATA/train_atwood_synth_window_v6/annotation.json"
echo "── Dataset build complete ── $(date)"

# ── Step 3: Cache ViT-S features ──────────────────────────────────────────────
VITS_CACHE=$CACHE_ROOT/vits_window_v6
echo "── Step 3: Caching ViT-S features ── $(date)"
rm -rf "$VITS_CACHE"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$TRAIN_ANN" \
  --output-dir "$VITS_CACHE/train" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$DATA/val_atwood_window_v6/annotation.json" \
  --output-dir "$VITS_CACHE/val" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
echo "── ViT-S feature cache complete ── $(date)"

# ── Step 4: Train + eval ──────────────────────────────────────────────────────
TAG=vits_window_v6
echo "── Step 4: Training $TAG (focal loss, max 40ep, early_stop=${EARLY_STOP}) ── $(date)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$VITS_CACHE/train" --val-cache "$VITS_CACHE/val" \
  --work-dir "$WEIGHTS/$TAG" \
  --epochs 40 --lr "$LR" --batch-size "$BATCH" \
  --focal-gamma "$FOCAL_GAMMA" --focal-alpha "$FOCAL_ALPHA" \
  --geometry-weight "$GEOMW" --hidden-channels "$HIDDEN" \
  --lr-scheduler "$SCHED" --num-workers 0 \
  --early-stopping-patience "$EARLY_STOP"

echo "── Step 5: Eval $TAG on val_balanced_v1 ── $(date)"
BC=~/argus_${TAG}_balcache
$PYTHON scripts/cache_heatmap_maps.py --annotations "$BAL_ANN" \
  --checkpoint "$WEIGHTS/$TAG/best.pt" --output-dir "$BC" --norm-mode zscore
$PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
  --heatmap-cache "$BC" --annotations "$BAL_ANN" --norm-mode zscore \
  --threshold 0.05 --threshold-sweep 0.70 0.75 0.80 0.85 --peak-floor 0.85 \
  --output "$OUT/$TAG/pf85/metrics_placeholder.json"
$PYTHON scripts/summarize_balanced_eval.py "$OUT/$TAG/pf85"
$PYTHON -m eval.geometry_metrics \
  --predictions "$OUT/$TAG/pf85/predictions_t070.json" \
  --annotations "$BAL_ANN" \
  --output "$OUT/$TAG/pf85/geometry_eval.json"
rm -rf "$BC"

echo "=== Window-v6 complete | $(date) ==="
echo "    Weights: $WEIGHTS/$TAG/best.pt"
echo "    Eval:    $OUT/$TAG/pf85/geometry_eval.json"
echo ""
echo "    Feature cache at: $VITS_CACHE (safe to delete after eval)"
echo "    To clean up: rm -rf $VITS_CACHE"
