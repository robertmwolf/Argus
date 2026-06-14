#!/usr/bin/env bash
# Run 19 — ViT-B heatmap retrain, CORRECTED TRAINING BUDGET.
#
# Root cause of Run 17/18 (found 2026-06-13, see agent_docs/run18_vitb_handoff.md):
# NOT a mechanical bug and NOT a feature deficiency. Proven by:
#   - ViT-B checkpoint loads 0 missing / 0 unexpected keys; features sane.
#   - Head in_channels auto-inferred to 768; zscore norm parity holds.
#   - OVERFIT PROBE: a frozen ViT-B head fits 12 streak tiles to train_dice 0.95
#     (BETTER than ViT-S 0.68) given enough steps -> features carry the signal.
# Run 17/18 simply UNDERTRAINED: ViT-B's 768-d head converges slower per step and
# Run 18's cosine(T_max=22) strangled the LR before the head could fit (train_dice
# stuck 0.13). Fix = MORE training + hotter LR with warmup, not unfreezing.
#
# Recipe deltas vs Run 18: epochs 22->80, lr default->3e-3, +4-epoch linear warmup.
# Reuses the (correct) consistent real-FITS split train_run18.json/val_run18.json.
#
# Head training is CHEAP (frozen backbone, cached features); caching is the slow
# part (~3.5 hr backbone pass). So we KEEP the train/val cache on exit to allow
# cheap LR/epoch re-sweeps via --train-cache without re-caching.
#
# Checkpoints (best.pt + latest.pt + history.json) written every epoch. Stop with
# Ctrl-C and use best.pt, or continue with --resume (latest.pt).
#
# Usage:
#   bash scripts/run19_vitb_pipeline.sh 2>&1 | tee /tmp/run19_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
VITB_WEIGHTS=$WEIGHTS/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
CACHE=~/argus_run19_cache
TRAIN_ANN=$REPO/data/annotations/train_run18.json
VAL_ANN=$REPO/data/annotations/val_run18.json
EPOCHS=${EPOCHS:-80}
LR=${LR:-3e-3}
WARMUP=${WARMUP:-4}
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "=== Run 19 ViT-B retrain | $(date) | epochs=$EPOCHS lr=$LR warmup=$WARMUP ==="
[ -f "$VITB_WEIGHTS" ] || { echo "ERROR missing $VITB_WEIGHTS"; exit 1; }

# ── Step 1: Cache ViT-B features (consistent zscore FITS; 38% neg tiles) ──────
# Params MUST match build_run18_split.py neg sizing and Run 18 cache geometry.
if [ -f "$CACHE/train/manifest.json" ] && [ -f "$CACHE/val/manifest.json" ]; then
  echo "── Cache already present at $CACHE — skipping caching ── $(date)"
else
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
fi

# ── Step 2: Train cached head (hotter LR + warmup + long budget) ──────────────
# Success signal: train_dice must cross ~0.6 then ~0.8 (ViT-S baseline 0.81).
# If it plateaus well below that, THEN consider larger head / backbone unfreeze.
echo "── Training ViT-B head ── $(date)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$CACHE/train" --val-cache "$CACHE/val" \
  --work-dir "$WEIGHTS/run19_vitb" \
  --epochs "$EPOCHS" --batch-size 32 --pos-weight 20 \
  --lr "$LR" --lr-scheduler cosine --warmup-epochs "$WARMUP" \
  --num-workers 0

# ── Step 3: Eval on val_balanced_v1 (the gate) ───────────────────────────────
echo "── Eval on val_balanced_v1 ── $(date)"
$PYTHON scripts/cache_heatmap_maps.py \
  --annotations data/annotations/val_balanced_v1.json \
  --checkpoint "$WEIGHTS/run19_vitb/best.pt" \
  --output-dir ~/argus_run19_bal_cache --norm-mode zscore
$PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
  --heatmap-cache ~/argus_run19_bal_cache \
  --annotations data/annotations/val_balanced_v1.json --norm-mode zscore \
  --threshold 0.05 --threshold-sweep 0.5 0.6 0.7 0.8 --peak-floor 0.85 \
  --output results/run19_vitb/balanced_v1/pf85/metrics_placeholder.json
$PYTHON scripts/summarize_balanced_eval.py results/run19_vitb/balanced_v1/pf85

# ── Step 4: Free only the balanced-eval cache; KEEP train/val cache for re-sweeps.
rm -rf ~/argus_run19_bal_cache
echo "=== Run 19 complete | $(date) | gate: beat ViT-S short recall 0.746 ==="
echo "    train/val feature cache kept at $CACHE (delete manually when done)"
