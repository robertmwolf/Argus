#!/usr/bin/env bash
# Train + eval the heatmap head on the CORRECTED window-crop dataset
# (train_atwood_synth_window_v1 / val_atwood_window_v1), for BOTH backbones.
#
# Context: every prior ViT-B run (17/18/19) and the Run 20 control trained on
# train_run18.json, whose targets were ~1800px off the real streaks (window-local
# obb coords against full-frame file_names — see agent_docs/dataset_naming.md).
# Both ViT-S and ViT-B capped at val_dice ~0.12 there. This dataset materialises
# each window as a crop so pixels and obb coords share one frame (validated:
# streak-on-target line contrast +16 vs the old -0.15).
#
# Pinned Recipe R (identical to the Run 20 control, so results are comparable):
#   epochs 40, lr 1e-3, batch 32, pos_weight 20, geom 0.25, hidden 256, cosine.
# Backbone is the only difference between the two arms.
#
# SUCCESS: val_dice should recover from ~0.12 toward ViT-S's historical ~0.77.
# If it does, the data fix is confirmed AND we finally get a fair ViT-S vs ViT-B
# comparison on correct data (the original project goal).
#
# Usage:
#   bash scripts/train_window_v1_pipeline.sh 2>&1 | tee /tmp/window_v1_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DSROOT=/Volumes/External/TrainingData
TRAIN_ANN=$DSROOT/train_atwood_synth_window_v1/annotation.json
VAL_ANN=$DSROOT/val_atwood_window_v1/annotation.json
BAL_ANN=$REPO/data/annotations/val_balanced_v1.json
VITS_W=$WEIGHTS/dinov3_vits16_lvd1689m.pth
VITB_W=$WEIGHTS/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
OUT=$REPO/results/window_v1
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

EPOCHS=40; LR=1e-3; BATCH=32; POSW=20; GEOMW=0.25; HIDDEN=256; SCHED=cosine

echo "=== Window-v1 training | $(date) | recipe: ep=$EPOCHS lr=$LR posw=$POSW hidden=$HIDDEN $SCHED ==="
[ -f "$TRAIN_ANN" ] || { echo "ERROR missing $TRAIN_ANN (run build_atwood_window_dataset.py --version 1)"; exit 1; }
[ -f "$VAL_ANN" ]   || { echo "ERROR missing $VAL_ANN"; exit 1; }

# arm: $1=tag  $2=model_size  $3=backbone_weights  $4=cache_dir
run_arm () {
  local TAG=$1 SIZE=$2 BW=$3 CACHE=$4
  if [ -f "$CACHE/train/manifest.json" ] && [ -f "$CACHE/val/manifest.json" ]; then
    echo "── $TAG cache present at $CACHE — skipping cache ── $(date)"
  else
    for SPLIT in train val; do
      local ANN=$TRAIN_ANN; [ "$SPLIT" = val ] && ANN=$VAL_ANN
      echo "── Caching $TAG $SPLIT features ── $(date)"
      $PYTHON scripts/cache_dinov3_heatmap_features.py \
        --annotations "$ANN" --output-dir "$CACHE/$SPLIT" \
        --backbone vit --model-size "$SIZE" --weights "$BW" \
        --image-size 518 --num-workers 0 --native-tile-size 400 \
        --tile-overlap 0.0 --norm-mode zscore --neg-tiles-per-image 4
    done
  fi
  echo "── Training $TAG (Recipe R) ── $(date)"
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
  rm -rf "$BC"
}

run_arm vits_window_v1 small "$VITS_W" ~/argus_vits_window_v1_cache
run_arm vitb_window_v1 base  "$VITB_W" ~/argus_vitb_window_v1_cache

echo "=== Window-v1 training complete | $(date) ==="
echo "    ViT-S: $WEIGHTS/vits_window_v1/history.json | eval $OUT/vits_window_v1/pf85"
echo "    ViT-B: $WEIGHTS/vitb_window_v1/history.json | eval $OUT/vitb_window_v1/pf85"
echo "    caches kept: ~/argus_vits_window_v1_cache ~/argus_vitb_window_v1_cache"