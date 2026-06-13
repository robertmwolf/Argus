#!/usr/bin/env bash
# Run 18 — ViT-B heatmap RETRAIN on the consistent real-FITS split.
#
# Fixes Run 17's root cause: training mixed FITS+PNG+synthetic-NPY (70% synth)
# while validation was pure FITS -> head underfit (train_dice 0.37). Run 18
# trains and validates on ONE consistent zscore-FITS distribution
# (train_run18.json / val_run18.json, built by build_run18_split.py).
#
# Hardware is slow (M3 MPS), so epochs are reduced and checkpoints are written
# every epoch (best.pt + latest.pt). Stop anytime with Ctrl-C and use best.pt;
# resume with --resume (latest.pt). Caching is the slow part (backbone pass).
#
# Usage:
#   bash scripts/run18_vitb_pipeline.sh 2>&1 | tee /tmp/run18_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
VITB_WEIGHTS=$WEIGHTS/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
CACHE=~/argus_run18_cache
TRAIN_ANN=$REPO/data/annotations/train_run18.json
VAL_ANN=$REPO/data/annotations/val_run18.json
EPOCHS=${EPOCHS:-22}            # reduced for slow MPS; best.pt saved every epoch
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "=== Run 18 ViT-B retrain | $(date) | epochs=$EPOCHS ==="
[ -f "$VITB_WEIGHTS" ] || { echo "ERROR missing $VITB_WEIGHTS"; exit 1; }

# ── Step 1: Cache ViT-B features (consistent zscore FITS; 38% neg tiles) ──────
# --neg-tiles-per-image 4 must match build_run18_split.py's neg sizing.
echo "── Caching ViT-B TRAIN features ── $(date)"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$TRAIN_ANN" --output-dir "$CACHE/train" \
  --backbone vit --model-size base --weights "$VITB_WEIGHTS" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode zscore --neg-tiles-per-image 4

echo "── Caching ViT-B VAL features ── $(date)"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$VAL_ANN" --output-dir "$CACHE/val" \
  --backbone vit --model-size base --weights "$VITB_WEIGHTS" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode zscore --neg-tiles-per-image 4

# ── Step 2: Train cached head (frozen backbone; checkpoints every epoch) ──────
# pos_weight 20 / cosine kept from Run 15 (proven). If train_dice climbs past
# ~0.6 the consistency fix worked. Stop early anytime -> best.pt is the best so far.
echo "── Training ViT-B head ── $(date)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$CACHE/train" --val-cache "$CACHE/val" \
  --work-dir "$WEIGHTS/run18_vitb" \
  --epochs "$EPOCHS" --batch-size 32 --pos-weight 20 \
  --num-workers 0 --lr-scheduler cosine

# ── Step 3: Eval on val_balanced_v1 (the gate) ───────────────────────────────
echo "── Eval on val_balanced_v1 ── $(date)"
$PYTHON scripts/cache_heatmap_maps.py \
  --annotations data/annotations/val_balanced_v1.json \
  --checkpoint "$WEIGHTS/run18_vitb/best.pt" \
  --output-dir ~/argus_run18_bal_cache --norm-mode zscore
$PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
  --heatmap-cache ~/argus_run18_bal_cache \
  --annotations data/annotations/val_balanced_v1.json --norm-mode zscore \
  --threshold 0.05 --threshold-sweep 0.5 0.6 0.7 0.8 --peak-floor 0.85 \
  --output results/run18_vitb/balanced_v1/pf85/metrics_placeholder.json
$PYTHON scripts/summarize_balanced_eval.py results/run18_vitb/balanced_v1/pf85

# ── Step 4: Free SSD ─────────────────────────────────────────────────────────
rm -rf "$CACHE" ~/argus_run18_bal_cache
echo "=== Run 18 complete | $(date) | gate: beat ViT-S short recall 0.746 ==="
