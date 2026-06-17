#!/usr/bin/env bash
# Train + eval the heatmap head with focal loss (window_v4).
#
# v4 changes vs v3:
#   - Focal loss (gamma=2.0, alpha=0.85) replaces BCE+pos_weight=20.
#     The (1-p_t)^gamma term down-weights easy examples, penalising the head
#     more for confident false activations on background tiles.
#   - Early stopping (patience=10 on val_loss) replaces fixed epoch counts.
#     ViT-S runs up to 40 epochs; ViT-B up to 80 but stops when val_loss
#     diverges (expected ~ep40-50 based on v3 history).
#   - Reuses v3 feature caches — no dataset build or re-caching needed.
#
# Usage:
#   bash scripts/train_window_v4_pipeline.sh 2>&1 | tee /tmp/window_v4_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
BAL_ANN=$REPO/data/annotations/val_balanced_v1.json
OUT=$REPO/results/window_v4
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

CACHE_ROOT=/Volumes/External/argus_caches  # keep off internal drive

# Reuse v3 feature caches — dataset and backbone features are identical.
VITS_CACHE=$CACHE_ROOT/vits_window_v3
VITB_CACHE=$CACHE_ROOT/vitb_window_v3

# Focal loss hyperparameters
FOCAL_GAMMA=2.0
FOCAL_ALPHA=0.85
EARLY_STOP=10

# All other recipe params unchanged from v3 (Recipe R)
LR=1e-3; BATCH=32; GEOMW=0.25; HIDDEN=256; SCHED=cosine

echo "=== Window-v4 training | $(date) | focal gamma=$FOCAL_GAMMA alpha=$FOCAL_ALPHA early_stop=$EARLY_STOP ==="

run_arm () {
  local TAG=$1 CACHE=$2 EPOCHS=$3
  echo "── Training $TAG (focal loss, max ${EPOCHS}ep, early_stop=${EARLY_STOP}) ── $(date)"
  $PYTHON training/train_dinov3_heatmap_cached.py \
    --train-cache "$CACHE/train" --val-cache "$CACHE/val" \
    --work-dir "$WEIGHTS/$TAG" \
    --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH" \
    --focal-gamma "$FOCAL_GAMMA" --focal-alpha "$FOCAL_ALPHA" \
    --geometry-weight "$GEOMW" --hidden-channels "$HIDDEN" \
    --lr-scheduler "$SCHED" --num-workers 0 \
    --early-stopping-patience "$EARLY_STOP"
  echo "── Eval $TAG on val_balanced_v1 ── $(date)"
  local BC=~/argus_${TAG}_balcache
  $PYTHON scripts/cache_heatmap_maps.py --annotations "$BAL_ANN" \
    --checkpoint "$WEIGHTS/$TAG/best.pt" --output-dir "$BC" --norm-mode zscore
  $PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
    --heatmap-cache "$BC" --annotations "$BAL_ANN" --norm-mode zscore \
    --threshold 0.05 --threshold-sweep 0.5 0.6 0.7 0.8 --peak-floor 0.85 \
    --output "$OUT/$TAG/pf85/metrics_placeholder.json"
  $PYTHON scripts/summarize_balanced_eval.py "$OUT/$TAG/pf85"
  $PYTHON -m eval.geometry_metrics \
    --predictions "$OUT/$TAG/pf85/predictions_t070.json" \
    --annotations "$BAL_ANN" \
    --output "$OUT/$TAG/pf85/geometry_eval.json"
  rm -rf "$BC"
  echo "── $TAG complete ── $(date)"
}

run_arm vits_window_v4 "$VITS_CACHE" 40
run_arm vitb_window_v4 "$VITB_CACHE" 80

echo "=== Window-v4 training complete | $(date) ==="
echo "    ViT-S: $WEIGHTS/vits_window_v4/history.json | eval $OUT/vits_window_v4/pf85"
echo "    ViT-B: $WEIGHTS/vitb_window_v4/history.json | eval $OUT/vitb_window_v4/pf85"
