#!/usr/bin/env bash
# Run 12 — cache → train → eval restart script.
#
# Use this after Steps 1–7 (NPY + synthetic + merge) are already complete on
# the external drive.  Picks up from feature caching with the fixed
# autostretch normalisation (apply_norm was using per-tile min-max before,
# which made every tile look equally "interesting" and hid the streak signal).
#
# Preconditions:
#   - /tmp/argus_run12_npy is a symlink → /Volumes/External/TrainingData/argus_run12_npy
#     (all NPY present: atwood1800 101G, val1800 14G, synth_short/medium ~7G, frigate 23M)
#   - /Volumes/External/TrainingData/annotations/all_train_run12_npy.json exists
#   - /Volumes/External/TrainingData/annotations/val_run12_1800_npy.json exists
#   - No stale cache dirs for vits_run12_* or convnext_run12_*
#
# Internal drive budget: ≤100 GB at a time (shared machine).
#   - Pre-Step-8: stage selected NPY on internal (~90 GB) for fast cache reads
#   - Steps 10–11: one backbone cache (~30 GB) on internal at a time
#
# Usage:
#   bash scripts/run12_cache_train_eval.sh 2>&1 | tee /tmp/run12_cte_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
EXT_CACHE_DIR=/Volumes/External/TrainingData/heatmap_cache
WEIGHTS_DIR=$REPO/weights

LOCAL_NPY_DIR=/tmp/argus_run12_npy   # symlink → external (set up by earlier pipeline)
LOCAL_CACHE_DIR=/tmp/argus_run12_cache

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 12 — Cache + Train + Eval  $(date)"
echo " Normalization: autostretch (fixed from per-tile min-max)"
echo "================================================================"

# Sanity checks
if [ ! -L "$LOCAL_NPY_DIR" ]; then
  echo "ERROR: $LOCAL_NPY_DIR is not a symlink to external NPY."
  echo "  Fix: ln -s /Volumes/External/TrainingData/argus_run12_npy /tmp/argus_run12_npy"
  exit 1
fi
for f in all_train_run12_npy.json val_run12_1800_npy.json; do
  [ -f "$ANN_DIR/$f" ] || { echo "ERROR: missing $ANN_DIR/$f"; exit 1; }
done
df -h / /Volumes/External

# ── Pre-Step-8: Stage selected NPY on internal SSD ───────────────────────────
# Annotation JSONs store the symlink path (/tmp/argus_run12_npy/...).
# Swapping the symlink for a real directory makes the cacher read from
# internal SSD at full speed, not the slow external drive.
# Only the selected tiles (~90 GB) are rsynced, staying within the 100 GB budget.
echo ""
echo "── Pre-Step-8: Stage selected NPY on internal SSD ──"

rm -f "$LOCAL_NPY_DIR"
mkdir -p \
  "$LOCAL_NPY_DIR/atwood1800" \
  "$LOCAL_NPY_DIR/val1800" \
  "$LOCAL_NPY_DIR/frigate110" \
  "$LOCAL_NPY_DIR/synth_short" \
  "$LOCAL_NPY_DIR/synth_medium"

$PYTHON - <<'PYEOF'
import json, sys
from pathlib import Path

TRAIN_ANN = "/Volumes/External/TrainingData/annotations/all_train_run12_npy.json"
VAL_ANN   = "/Volumes/External/TrainingData/annotations/val_run12_1800_npy.json"
NPY_EXT   = Path("/Volumes/External/TrainingData/argus_run12_npy")
OUT_LIST  = "/tmp/run12_npy_files.txt"
PREFIX    = "/tmp/argus_run12_npy/"

files = set()
missing = 0
for ann_path in [TRAIN_ANN, VAL_ANN]:
    data = json.load(open(ann_path))
    for img in data["images"]:
        fn = img.get("file_name", "")
        if fn.startswith(PREFIX):
            rel = fn[len(PREFIX):]
            if (NPY_EXT / rel).exists():
                files.add(rel)
            else:
                missing += 1

with open(OUT_LIST, "w") as f:
    for rel in sorted(files):
        f.write(rel + "\n")

print(f"Selected {len(files)} NPY files for staging  ({missing} missing on external)")
PYEOF

echo "Rsyncing selected NPY to internal SSD..."
rsync -a --files-from=/tmp/run12_npy_files.txt \
  /Volumes/External/TrainingData/argus_run12_npy/ \
  "$LOCAL_NPY_DIR/"

STAGED=$(du -sh "$LOCAL_NPY_DIR" | cut -f1)
echo "Staged ${STAGED} on internal."
df -h /

STAGED_KB=$(du -s "$LOCAL_NPY_DIR" | cut -f1)
if (( STAGED_KB > 104857600 )); then
  echo "WARNING: staged NPY exceeds 100 GB — continuing but may crowd other users."
fi

# ── Step 8: Cache features (reads from internal SSD, writes .pt to external) ──
echo ""
echo "── Step 8a: Cache ViT-S train features (518px, autostretch) ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run12_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/vits_run12_train" \
  --backbone vit --model-size small \
  --image-size 518 --num-workers 0 \
  --norm-mode autostretch

echo ""
echo "── Step 8b: Cache ViT-S val features (518px, autostretch) ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/val_run12_1800_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/vits_run12_val" \
  --backbone vit --model-size small \
  --image-size 518 --num-workers 0 \
  --norm-mode autostretch

echo ""
echo "── Step 8c: Cache ConvNeXt-S train features (518px, autostretch) ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run12_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/convnext_run12_train" \
  --backbone convnext --model-size small \
  --image-size 518 --num-workers 0 \
  --norm-mode autostretch

echo ""
echo "── Step 8d: Cache ConvNeXt-S val features (518px, autostretch) ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/val_run12_1800_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/convnext_run12_val" \
  --backbone convnext --model-size small \
  --image-size 518 --num-workers 0 \
  --norm-mode autostretch

# ── Step 9: Clear staged NPY from internal SSD ────────────────────────────────
echo ""
echo "── Step 9: Clear staged NPY from internal SSD ──"
rm -rf "$LOCAL_NPY_DIR"
df -h /
echo "Internal freed. External NPY retained."

# ── Step 10: Train ViT-S ──────────────────────────────────────────────────────
echo ""
echo "── Step 10: Copy ViT-S cache to local SSD + train ──"
mkdir -p "$LOCAL_CACHE_DIR"
rsync -a "$EXT_CACHE_DIR/vits_run12_train/" "$LOCAL_CACHE_DIR/vits_run12_train/"
rsync -a "$EXT_CACHE_DIR/vits_run12_val/"   "$LOCAL_CACHE_DIR/vits_run12_val/"
echo "ViT-S cache staged: $(du -sh $LOCAL_CACHE_DIR | cut -f1)"

$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/vits_run12_train" \
  --val-cache   "$LOCAL_CACHE_DIR/vits_run12_val" \
  --work-dir    "$WEIGHTS_DIR/run12_vits" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

rm -rf "$LOCAL_CACHE_DIR/vits_run12_train" "$LOCAL_CACHE_DIR/vits_run12_val"

# ── Step 11: Train ConvNeXt-S ─────────────────────────────────────────────────
echo ""
echo "── Step 11: Copy ConvNeXt-S cache to local SSD + train ──"
rsync -a "$EXT_CACHE_DIR/convnext_run12_train/" "$LOCAL_CACHE_DIR/convnext_run12_train/"
rsync -a "$EXT_CACHE_DIR/convnext_run12_val/"   "$LOCAL_CACHE_DIR/convnext_run12_val/"
echo "ConvNeXt-S cache staged: $(du -sh $LOCAL_CACHE_DIR | cut -f1)"

$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/convnext_run12_train" \
  --val-cache   "$LOCAL_CACHE_DIR/convnext_run12_val" \
  --work-dir    "$WEIGHTS_DIR/run12_convnext_s" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

rm -rf "$LOCAL_CACHE_DIR/convnext_run12_train" "$LOCAL_CACHE_DIR/convnext_run12_val"
rmdir "$LOCAL_CACHE_DIR" 2>/dev/null || true

# ── Step 12: Evaluate both models ─────────────────────────────────────────────
echo ""
echo "── Step 12a: Evaluate ViT-S (t=0.05, tiled 1800px → 518px, stitch) ──"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
VITS_HEATMAP_NATIVE_TILE_SIZE=1800 \
$PYTHON scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run12_vits/best.pt" \
  --output      results/run12_vits/t0.05/metrics.json \
  --tiled --threshold 0.05 --stitch

$PYTHON scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run12_vits/t0.05/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run12_vits/threshold_sweep \
  --thresholds 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "── Step 12b: Evaluate ConvNeXt-S (t=0.05, tiled 1800px → 518px, stitch) ──"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=1800 \
$PYTHON scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run12_convnext_s/best.pt" \
  --backbone convnext \
  --output      results/run12_convnext_s/t0.05/metrics.json \
  --tiled --threshold 0.05 --stitch

$PYTHON scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run12_convnext_s/t0.05/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run12_convnext_s/threshold_sweep \
  --thresholds 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "================================================================"
echo " Run 12 Complete  $(date)"
echo " ViT-S:      results/run12_vits/threshold_sweep/threshold_sweep.json"
echo " ConvNeXt-S: results/run12_convnext_s/threshold_sweep/threshold_sweep.json"
echo "================================================================"
