#!/usr/bin/env bash
# Train + eval the heatmap head with balanced hard negative mining (window_v7).
#
# v7 changes vs v6:
#   - Reverts hard neg budget from 1/2 → 1/3 (v6's 1/2 caused precision collapse to 40%).
#   - Oracle: vits_window_v5 (same as v6).
#   - Mining threshold: 0.75 (vs v5's 0.85 and v6's 0.70) — captures borderline FPs
#     without flooding the neg pool with near-GT ambiguous tiles.
#   - Keeps random tile sampling (30/frame) from v6 for speed + spatial coverage.
#   - ViT-S only.
#
# v6 post-mortem: changing both the budget (1/2) AND threshold (0.70) simultaneously
# caused a precision collapse (67%→40%). v7 isolates the threshold change only.
#
# Usage:
#   bash scripts/train_window_v7_pipeline.sh 2>&1 | tee /tmp/window_v7_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DATA=/Volumes/External/TrainingData
BAL_ANN=$REPO/data/annotations/val_balanced_v1.json
MERGED_ANN=$REPO/data/annotations/all_train_run17_merged.json
HARD_NEGS=$REPO/data/annotations/hard_negatives_vits_window_v5_t075.json
OUT=$REPO/results/window_v7
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

# Focal loss / early stopping (same as v5/v6)
FOCAL_GAMMA=2.0; FOCAL_ALPHA=0.85; EARLY_STOP=10
LR=1e-3; BATCH=32; GEOMW=0.25; HIDDEN=256; SCHED=cosine
VITS_W=$WEIGHTS/dinov3_vits16_lvd1689m.pth
CACHE_ROOT=/Volumes/External/argus_caches

echo "=== Window-v7 pipeline | $(date) | 1/3 hard-neg budget + t=0.75 threshold ==="

# ── Step 1: Mine hard negatives using v5 as oracle ────────────────────────────
echo "── Step 1: Mining hard negatives (oracle=v5, threshold=0.75, random-tiles=30) ── $(date)"
$PYTHON scripts/mine_hard_negatives.py \
  --checkpoint "$WEIGHTS/vits_window_v5/best.pt" \
  --annotations "$MERGED_ANN" \
  --output "$HARD_NEGS" \
  --peak-threshold 0.75 \
  --margin 400 \
  --tile-size 400 \
  --random-tiles 30 \
  --max-hard-negs 600
echo "── Mining complete; hard negs saved to $HARD_NEGS ── $(date)"

# ── Step 2: Build v7 dataset (hard neg budget = 1/3) ─────────────────────────
echo "── Step 2: Building v7 dataset (hard-neg budget=1/3) ── $(date)"
$PYTHON scripts/build_atwood_window_dataset.py \
  --dataset-root "$DATA" \
  --version 7 \
  --source "$MERGED_ANN" \
  --hard-negs-json "$HARD_NEGS" \
  --neg-frac 0.42 \
  --bg-per-frame 3 \
  --seed 20
TRAIN_ANN="$DATA/train_atwood_synth_window_v7/annotation.json"
echo "── Dataset build complete ── $(date)"

# ── Step 3: Cache ViT-S features ──────────────────────────────────────────────
VITS_CACHE=$CACHE_ROOT/vits_window_v7
echo "── Step 3: Caching ViT-S features ── $(date)"
rm -rf "$VITS_CACHE"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$TRAIN_ANN" \
  --output-dir "$VITS_CACHE/train" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$DATA/val_atwood_window_v7/annotation.json" \
  --output-dir "$VITS_CACHE/val" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
echo "── ViT-S feature cache complete ── $(date)"

# ── Step 4: Train ─────────────────────────────────────────────────────────────
TAG=vits_window_v7
echo "── Step 4: Training $TAG (focal loss, max 40ep, early_stop=${EARLY_STOP}) ── $(date)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$VITS_CACHE/train" --val-cache "$VITS_CACHE/val" \
  --work-dir "$WEIGHTS/$TAG" \
  --epochs 40 --lr "$LR" --batch-size "$BATCH" \
  --focal-gamma "$FOCAL_GAMMA" --focal-alpha "$FOCAL_ALPHA" \
  --geometry-weight "$GEOMW" --hidden-channels "$HIDDEN" \
  --lr-scheduler "$SCHED" --num-workers 0 \
  --early-stopping-patience "$EARLY_STOP"

# ── Step 5: Eval ──────────────────────────────────────────────────────────────
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

echo "=== Window-v7 complete | $(date) ==="
echo "    Weights: $WEIGHTS/$TAG/best.pt"
echo "    Eval:    $OUT/$TAG/pf85/geometry_eval.json"
echo ""
echo "    Feature cache at: $VITS_CACHE (safe to delete after eval)"
echo "    To clean up: rm -rf $VITS_CACHE"
