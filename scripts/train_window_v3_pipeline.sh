#!/usr/bin/env bash
# Train + eval the heatmap head on the v3 window-crop dataset.
#
# v3 fix vs v2:
#   - Per-annotated-frame background tiles added to negative pool (--bg-per-frame 3)
#   - Closes the train/eval distribution gap: model now sees "background tiles
#     from frames that contain streaks" as negatives, which is what it encounters
#     at inference for the majority of tiles on a full frame.
#   - Corpus neg count reduced proportionally so total neg_frac stays ≈ 0.42.
#
# Same recipe as v2 (Recipe R): lr 1e-3, batch 32, pos_weight 20, geom 0.25,
#   hidden 256, cosine, norm-mode none.  ViT-S 40 epochs, ViT-B 80 epochs.
#
# Usage:
#   bash scripts/train_window_v3_pipeline.sh 2>&1 | tee /tmp/window_v3_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DSROOT=/Volumes/External/TrainingData
BAL_ANN=$REPO/data/annotations/val_balanced_v1.json
VITS_W=$WEIGHTS/dinov3_vits16_lvd1689m.pth
VITB_W=$WEIGHTS/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
OUT=$REPO/results/window_v3
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

LR=1e-3; BATCH=32; POSW=20; GEOMW=0.25; HIDDEN=256; SCHED=cosine
CACHE_ROOT=/Volumes/External/argus_caches  # keep off internal drive
TRAIN_ANN=$DSROOT/train_atwood_synth_window_v3/annotation.json
VAL_ANN=$DSROOT/val_atwood_window_v3/annotation.json

echo "=== Window-v3 build | $(date) ==="
if [ -f "$TRAIN_ANN" ] && [ -f "$VAL_ANN" ]; then
  echo "── v3 dataset already built — skipping build step ── $(date)"
else
  $PYTHON scripts/build_atwood_window_dataset.py \
    --version 3 --bg-per-frame 3 --neg-frac 0.42 \
    --n-synth-short 400 --pos-tiles-per-window 3.0 --neg-tiles-per-image 4
fi
echo "── v3 dataset ready ── $(date)"

# arm: $1=tag  $2=model_size  $3=backbone_weights  $4=cache_dir  $5=epochs
# Val cache is launched in the background immediately after train cache finishes
# so training and val caching overlap.  Training polls until val manifest exists
# before the first validation epoch needs it.
run_arm () {
  local TAG=$1 SIZE=$2 BW=$3 CACHE=$4 EPOCHS=$5
  if [ -f "$CACHE/train/manifest.json" ] && [ -f "$CACHE/val/manifest.json" ]; then
    echo "── $TAG cache present at $CACHE — skipping cache ── $(date)"
  else
    if [ ! -f "$CACHE/train/manifest.json" ]; then
      echo "── Caching $TAG train features ── $(date)"
      $PYTHON scripts/cache_dinov3_heatmap_features.py \
        --annotations "$TRAIN_ANN" --output-dir "$CACHE/train" \
        --backbone vit --model-size "$SIZE" --weights "$BW" \
        --image-size 518 --num-workers 0 --native-tile-size 400 \
        --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
    fi
    if [ ! -f "$CACHE/val/manifest.json" ]; then
      echo "── Caching $TAG val features in background ── $(date)"
      $PYTHON scripts/cache_dinov3_heatmap_features.py \
        --annotations "$VAL_ANN" --output-dir "$CACHE/val" \
        --backbone vit --model-size "$SIZE" --weights "$BW" \
        --image-size 518 --num-workers 0 --native-tile-size 400 \
        --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4 \
        >> /tmp/window_v3_valcache_${TAG}.log 2>&1 &
      local VAL_PID=$!
      echo "── Val cache PID=$VAL_PID — training will start now and wait for val manifest ── $(date)"
    fi
  fi
  # Wait for val cache before training reads it (poll; typically <10 min)
  until [ -f "$CACHE/val/manifest.json" ]; do
    echo "── Waiting for $TAG val cache (checking every 30s) ── $(date)"
    sleep 30
  done
  echo "── Training $TAG (Recipe R, ${EPOCHS}ep) ── $(date)"
  $PYTHON training/train_dinov3_heatmap_cached.py \
    --train-cache "$CACHE/train" --val-cache "$CACHE/val" \
    --work-dir "$WEIGHTS/$TAG" \
    --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH" \
    --pos-weight "$POSW" --geometry-weight "$GEOMW" --hidden-channels "$HIDDEN" \
    --lr-scheduler "$SCHED" --num-workers 0
  echo "── Eval $TAG on val_balanced_v1 ── $(date)"
  local BC=~/argus_${TAG}_balcache
  $PYTHON scripts/cache_heatmap_maps.py --annotations "$BAL_ANN" \
    --checkpoint "$WEIGHTS/$TAG/best.pt" --output-dir "$BC" --norm-mode zscore
  $PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
    --heatmap-cache "$BC" --annotations "$BAL_ANN" --norm-mode zscore \
    --threshold 0.05 --threshold-sweep 0.5 0.6 0.7 0.8 --peak-floor 0.85 \
    --output "$OUT/$TAG/pf85/metrics_placeholder.json"
  $PYTHON scripts/summarize_balanced_eval.py "$OUT/$TAG/pf85"
  # Geometry eval (T2 raw OBB only — no --images-dir since T3/Radon degrades results)
  $PYTHON -m eval.geometry_metrics \
    --predictions "$OUT/$TAG/pf85/predictions_t070.json" \
    --annotations "$BAL_ANN" \
    --output "$OUT/$TAG/pf85/geometry_eval.json"
  rm -rf "$BC"
}

echo "=== Window-v3 training | $(date) | lr=$LR posw=$POSW hidden=$HIDDEN $SCHED ==="
run_arm vits_window_v3 small "$VITS_W" "$CACHE_ROOT/vits_window_v3" 40
run_arm vitb_window_v3 base  "$VITB_W" "$CACHE_ROOT/vitb_window_v3" 80

echo "=== Window-v3 training complete | $(date) ==="
echo "    ViT-S: $WEIGHTS/vits_window_v3/history.json | eval $OUT/vits_window_v3/pf85"
echo "    ViT-B: $WEIGHTS/vitb_window_v3/history.json | eval $OUT/vitb_window_v3/pf85"
echo "    Feature caches at $CACHE_ROOT — safe to delete after eval"
