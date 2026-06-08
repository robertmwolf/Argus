#!/usr/bin/env bash
# Run 15 — ViT-S heatmap at 400px native tile size.
#
# THE KEY CHANGE vs Run 12: training now uses 400px sub-tiles (matching
# the OBB training scale and the default VITS_HEATMAP_NATIVE_TILE_SIZE=400
# used at inference time).  Run 12 trained at full 1800px tile scale, which
# caused short-streak recall to be 0% (a 150px streak = ~3 ViT patches at
# 1800px vs ~9 patches at 400px — below detection threshold at 1800px).
#
# Run 14 = AstroPT-89M (parallel run); Run 15 = ViT-S at correct scale.
#
# Normalization: zscore — consistent with OBB evaluation default and
# inference pipeline (ARGUS_NORM default).  Run 12 used autostretch, which
# also created a train/inference norm mismatch.
#
# Cache destination: internal SSD (/tmp) — external drive is 96% full.
# Budget: ~30–50 GB train + 5 GB val (tile_overlap=0, ~2–3 crops/annotation).
#
# Preconditions:
#   - /tmp/argus_run12_npy is a symlink → /Volumes/External/TrainingData/argus_run12_npy
#     (all_train_run13_npy.json and val_run12_1800_npy.json reference this path)
#   - /Volumes/External/TrainingData/argus_run13_npy/{synth_short,synth_medium} exist
#     (absolute paths baked into all_train_run13_npy.json)
#   - /Volumes/External/TrainingData/annotations/{all_train_run13_npy,val_run12_1800_npy}.json
#
# Usage:
#   bash scripts/run15_pipeline.sh 2>&1 | tee /tmp/run15_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
WEIGHTS_DIR=$REPO/weights

LOCAL_NPY_LINK=/tmp/argus_run12_npy   # symlink to external NPY
LOCAL_CACHE_DIR=/tmp/argus_run15_cache

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 15 — ViT-S Heatmap | 400px native tiles | zscore  $(date)"
echo " Key fix: 400px tiles → short streaks visible (~9 patches)"
echo " Run 14 = AstroPT; Run 15 = ViT-S at correct training scale"
echo "================================================================"

# ── Sanity checks ────────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Verify prerequisites ──"

if [ ! -L "$LOCAL_NPY_LINK" ] && [ ! -d "$LOCAL_NPY_LINK" ]; then
  echo "ERROR: $LOCAL_NPY_LINK does not exist."
  echo "  Fix: ln -s /Volumes/External/TrainingData/argus_run12_npy $LOCAL_NPY_LINK"
  exit 1
fi
echo "  OK: $LOCAL_NPY_LINK"

for f in all_train_run13_npy.json val_run12_1800_npy.json; do
  [ -f "$ANN_DIR/$f" ] || { echo "ERROR: missing $ANN_DIR/$f"; exit 1; }
  echo "  OK: $ANN_DIR/$f"
done

df -h / /Volumes/External

# ── Step 2: Cache ViT-S features (400px tiles, zscore, internal SSD) ─────────
echo ""
echo "── Step 2a: Cache ViT-S train features (400px native tiles, zscore) ──"
echo "  Writing to: $LOCAL_CACHE_DIR/vits_run15_train"
echo "  tile_overlap=0 → ~2–3 crops per annotation (~25K entries, ~80 GB)"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run13_npy.json" \
  --output-dir  "$LOCAL_CACHE_DIR/vits_run15_train" \
  --backbone vit --model-size small \
  --image-size 518 --num-workers 0 \
  --native-tile-size 400 \
  --tile-overlap 0.0 \
  --norm-mode zscore

echo ""
echo "── Step 2b: Cache ViT-S val features (400px native tiles, zscore) ──"
echo "  Writing to: $LOCAL_CACHE_DIR/vits_run15_val"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/val_run12_1800_npy.json" \
  --output-dir  "$LOCAL_CACHE_DIR/vits_run15_val" \
  --backbone vit --model-size small \
  --image-size 518 --num-workers 0 \
  --native-tile-size 400 \
  --tile-overlap 0.0 \
  --norm-mode zscore

echo ""
echo "Cache sizes:"
du -sh "$LOCAL_CACHE_DIR/vits_run15_train" "$LOCAL_CACHE_DIR/vits_run15_val" 2>/dev/null || true
df -h /

# ── Step 3: Train ViT-S ───────────────────────────────────────────────────────
echo ""
echo "── Step 3: Train ViT-S (40 epochs, batch=32, pos_weight=20, cosine LR) ──"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/vits_run15_train" \
  --val-cache   "$LOCAL_CACHE_DIR/vits_run15_val" \
  --work-dir    "$WEIGHTS_DIR/run15_vits" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

# ── Step 4: Free internal SSD ─────────────────────────────────────────────────
echo ""
echo "── Step 4: Free internal SSD cache ──"
rm -rf "$LOCAL_CACHE_DIR/vits_run15_train" "$LOCAL_CACHE_DIR/vits_run15_val"
rmdir "$LOCAL_CACHE_DIR" 2>/dev/null || true
df -h /
echo "Internal SSD freed."

# ── Step 5: Evaluate on val set (default 400px inference, zscore) ─────────────
# VITS_HEATMAP_NATIVE_TILE_SIZE defaults to 400 — do NOT override.
# Using zscore (ARGUS_NORM default) to match training norm.
echo ""
echo "── Step 5: Evaluate ViT-S on val_run12_1800_npy (t=0.05, 400px tiles, stitch) ──"
mkdir -p results/run15_vits/t0.05
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
$PYTHON scripts/evaluate_dinov3_heatmap.py \
  --annotations "$ANN_DIR/val_run12_1800_npy.json" \
  --checkpoint  "$WEIGHTS_DIR/run15_vits/best.pt" \
  --output      results/run15_vits/t0.05/metrics.json \
  --tiled --threshold 0.05 --stitch

# ── Step 6: Post-hoc threshold sweep ──────────────────────────────────────────
echo ""
echo "── Step 6: Threshold sweep ──"
$PYTHON scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run15_vits/t0.05/predictions.json \
  --annotations "$ANN_DIR/val_run12_1800_npy.json" \
  --output-dir  results/run15_vits/threshold_sweep \
  --thresholds 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --iou-threshold 0.10 \
  --stitch || true

echo ""
echo "================================================================"
echo " Run 15 Complete  $(date)"
echo " Results: results/run15_vits/threshold_sweep/threshold_sweep.json"
echo " Compare to OBB:  results/obb_vs_heatmap/obb_sweep/threshold_sweep.json"
echo "================================================================"

# Print best threshold summary
$PYTHON - <<'PYEOF'
import json, sys
from pathlib import Path

sweep = Path("results/run15_vits/threshold_sweep/threshold_sweep.json")
if not sweep.exists():
    print("Threshold sweep not found — check eval step above.")
    sys.exit(0)

data = json.loads(sweep.read_text())
rows = data if isinstance(data, list) else data.get("results", [])

print("\nRun 15 ViT-S threshold sweep (val_run12_1800, IoU>=0.10, stitch):")
print(f"{'thresh':>7}  {'P':>6}  {'R':>6}  {'F1':>6}  {'short_R':>8}  {'med_R':>7}  {'long_R':>7}  {'preds':>6}")
print("-" * 68)
for r in rows:
    t   = r.get("threshold", r.get("t", "?"))
    p   = r.get("precision", r.get("P", 0))
    rc  = r.get("recall",    r.get("R", 0))
    f1  = r.get("f1",        r.get("F1", 0))
    sr  = r.get("short_recall",  r.get("short_R",  0))
    mr  = r.get("medium_recall", r.get("med_R",    0))
    lr  = r.get("long_recall",   r.get("long_R",   0))
    n   = r.get("n_predictions", r.get("preds",    "?"))
    print(f"{t:>7.2f}  {100*p:>5.1f}%  {100*rc:>5.1f}%  {100*f1:>5.1f}%  "
          f"{100*sr:>7.1f}%  {100*mr:>6.1f}%  {100*lr:>6.1f}%  {n:>6}")

# Compare to OBB best
obb_sweep = Path("results/obb_vs_heatmap/obb_sweep/threshold_sweep.json")
if obb_sweep.exists():
    obb_data = json.loads(obb_sweep.read_text())
    obb_rows = obb_data if isinstance(obb_data, list) else obb_data.get("results", [])
    best_obb = max(obb_rows, key=lambda r: r.get("f1", r.get("F1", 0)))
    t  = best_obb.get("threshold", "?")
    f1 = best_obb.get("f1", best_obb.get("F1", 0))
    sr = best_obb.get("short_recall", best_obb.get("short_R", 0))
    print(f"\nOBB best (t={t}): F1={100*f1:.1f}%  short_R={100*sr:.1f}%")
PYEOF
