#!/usr/bin/env bash
# Train + eval window_v8: calibration fix + looser stratification, ViT-S then ViT-B.
#
# v8 changes vs v7:
#   - ~90 training frames had incorrect calibration; those are now corrected on disk.
#     The merged annotation JSON is unchanged (paths are stable); the model just sees
#     better pixel data when features are cached.
#   - val_frac reduced 0.15 → 0.08: gives ~81 more training frames (mostly medium/long)
#     while keeping enough for reliable early-stopping signal.
#   - No hard negative mining: isolate the calibration + stratification effect cleanly.
#     If v8 beats v5, add hard negs back for v9.
#   - ViT-S (40 ep max) then ViT-B (80 ep max), sequential on one MPS GPU.
#
# Usage:
#   bash scripts/train_window_v8_pipeline.sh 2>&1 | tee /tmp/window_v8_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DATA=/Volumes/External/TrainingData
BAL_ANN=$REPO/data/annotations/val_balanced_v1.json
MERGED_ANN=$REPO/data/annotations/all_train_run17_merged.json
OUT=$REPO/results/window_v8
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

FOCAL_GAMMA=2.0; FOCAL_ALPHA=0.85; EARLY_STOP=10
LR=1e-3; BATCH=32; GEOMW=0.25; HIDDEN=256; SCHED=cosine
VITS_W=$WEIGHTS/dinov3_vits16_lvd1689m.pth
VITB_W=$WEIGHTS/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
CACHE_ROOT=/Volumes/External/argus_caches

echo "=== Window-v8 pipeline | $(date) | calibration fix + val_frac=0.08 | ViT-S then ViT-B ==="

# ── Step 1: Build v8 dataset (no hard negs, val_frac=0.08) ───────────────────
echo "── Step 1: Building v8 dataset (val_frac=0.08, no hard negs) ── $(date)"
$PYTHON scripts/build_atwood_window_dataset.py \
  --dataset-root "$DATA" \
  --version 8 \
  --source "$MERGED_ANN" \
  --val-frac 0.08 \
  --neg-frac 0.42 \
  --bg-per-frame 3 \
  --seed 21
TRAIN_ANN="$DATA/train_atwood_synth_window_v8/annotation.json"
echo "── Dataset build complete ── $(date)"

# ── ViT-S arm ────────────────────────────────────────────────────────────────
VITS_TAG=vits_window_v8
VITS_CACHE=$CACHE_ROOT/vits_window_v8

echo "── Step 2a: Caching ViT-S features ── $(date)"
rm -rf "$VITS_CACHE"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$TRAIN_ANN" \
  --output-dir "$VITS_CACHE/train" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$DATA/val_atwood_window_v8/annotation.json" \
  --output-dir "$VITS_CACHE/val" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
echo "── ViT-S feature cache complete ── $(date)"

echo "── Step 2b: Training $VITS_TAG (focal loss, max 40ep) ── $(date)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$VITS_CACHE/train" --val-cache "$VITS_CACHE/val" \
  --work-dir "$WEIGHTS/$VITS_TAG" \
  --epochs 40 --lr "$LR" --batch-size "$BATCH" \
  --focal-gamma "$FOCAL_GAMMA" --focal-alpha "$FOCAL_ALPHA" \
  --geometry-weight "$GEOMW" --hidden-channels "$HIDDEN" \
  --lr-scheduler "$SCHED" --num-workers 0 \
  --early-stopping-patience "$EARLY_STOP"

echo "── Step 2c: Eval $VITS_TAG on val_balanced_v1 ── $(date)"
BC=~/argus_${VITS_TAG}_balcache
$PYTHON scripts/cache_heatmap_maps.py --annotations "$BAL_ANN" \
  --checkpoint "$WEIGHTS/$VITS_TAG/best.pt" --output-dir "$BC" --norm-mode zscore
$PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
  --heatmap-cache "$BC" --annotations "$BAL_ANN" --norm-mode zscore \
  --threshold 0.05 --threshold-sweep 0.70 0.75 0.80 0.85 --peak-floor 0.85 \
  --output "$OUT/$VITS_TAG/pf85/metrics_placeholder.json"
$PYTHON scripts/summarize_balanced_eval.py "$OUT/$VITS_TAG/pf85"
$PYTHON -m eval.geometry_metrics \
  --predictions "$OUT/$VITS_TAG/pf85/predictions_t070.json" \
  --annotations "$BAL_ANN" \
  --output "$OUT/$VITS_TAG/pf85/geometry_eval.json"
rm -rf "$BC"
echo "── $VITS_TAG complete ── $(date)"
rm -rf "$VITS_CACHE"

# ── ViT-B arm ────────────────────────────────────────────────────────────────
VITB_TAG=vitb_window_v8
VITB_CACHE=$CACHE_ROOT/vitb_window_v8

echo "── Step 3a: Caching ViT-B features ── $(date)"
rm -rf "$VITB_CACHE"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$TRAIN_ANN" \
  --output-dir "$VITB_CACHE/train" \
  --backbone vit --model-size base --weights "$VITB_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$DATA/val_atwood_window_v8/annotation.json" \
  --output-dir "$VITB_CACHE/val" \
  --backbone vit --model-size base --weights "$VITB_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
echo "── ViT-B feature cache complete ── $(date)"

echo "── Step 3b: Training $VITB_TAG (focal loss, max 80ep) ── $(date)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$VITB_CACHE/train" --val-cache "$VITB_CACHE/val" \
  --work-dir "$WEIGHTS/$VITB_TAG" \
  --epochs 80 --lr "$LR" --batch-size "$BATCH" \
  --focal-gamma "$FOCAL_GAMMA" --focal-alpha "$FOCAL_ALPHA" \
  --geometry-weight "$GEOMW" --hidden-channels "$HIDDEN" \
  --lr-scheduler "$SCHED" --num-workers 0 \
  --early-stopping-patience "$EARLY_STOP"

echo "── Step 3c: Eval $VITB_TAG on val_balanced_v1 ── $(date)"
BC=~/argus_${VITB_TAG}_balcache
$PYTHON scripts/cache_heatmap_maps.py --annotations "$BAL_ANN" \
  --checkpoint "$WEIGHTS/$VITB_TAG/best.pt" --output-dir "$BC" --norm-mode zscore
$PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
  --heatmap-cache "$BC" --annotations "$BAL_ANN" --norm-mode zscore \
  --threshold 0.05 --threshold-sweep 0.70 0.75 0.80 0.85 --peak-floor 0.85 \
  --output "$OUT/$VITB_TAG/pf85/metrics_placeholder.json"
$PYTHON scripts/summarize_balanced_eval.py "$OUT/$VITB_TAG/pf85"
$PYTHON -m eval.geometry_metrics \
  --predictions "$OUT/$VITB_TAG/pf85/predictions_t070.json" \
  --annotations "$BAL_ANN" \
  --output "$OUT/$VITB_TAG/pf85/geometry_eval.json"
rm -rf "$BC"
echo "── $VITB_TAG complete ── $(date)"
rm -rf "$VITB_CACHE"

echo "=== Window-v8 complete | $(date) ==="
echo "    ViT-S weights: $WEIGHTS/$VITS_TAG/best.pt"
echo "    ViT-B weights: $WEIGHTS/$VITB_TAG/best.pt"
echo "    ViT-S eval:    $OUT/$VITS_TAG/pf85/geometry_eval.json"
echo "    ViT-B eval:    $OUT/$VITB_TAG/pf85/geometry_eval.json"
