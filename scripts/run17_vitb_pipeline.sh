#!/usr/bin/env bash
# Run 17 — ViT-B heatmap at 400px native tile size (frozen backbone).
#
# Track A: same architecture as Run 15 (ViT-S) but with the ViT-B backbone
# (768-dim features vs 384-dim).  Everything else is held constant so the
# backbone size is the only variable.
#
# Expected gain: +5–15% F1 over Run 15 (21.9%) from richer frozen features.
# Short-streak recall may stay near 0% — the 16px patch size is unchanged,
# so sub-5-patch streaks are still at the spatial-resolution limit.
#
# Preconditions:
#   - /Volumes/External is mounted (FITS source files live there)
#   - /Volumes/External/TrainingData/annotations/all_train_run5_tiled_ts1800.json exists
#   - /Volumes/External/TrainingData/annotations/val_atwood_tiled_ts1800.json exists
#   - /Volumes/External/TrainingData/annotations/all_train_run13_npy.json exists
#     (for synth NPY paths — argus_run13_npy/synth_short/ and synth_medium/ must be present)
#   - weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth exists
#
# No NPY symlink required. build_run17_annotations.py strips __txN_tyN_tsN tile suffixes
# from the source annotation to recover real FITS paths, then merges with synth NPY.
# The feature cacher loads FITS directly — NPY pre-conversion is not needed.
#
# Cache budget: ViT-B features are 768-dim vs 384-dim (ViT-S), so each .pt
# file is ~2× larger.  Expect ~60–100 GB train + ~10 GB val on internal SSD.
# Check available space before running.
#
# Usage:
#   bash scripts/run17_vitb_pipeline.sh 2>&1 | tee /tmp/run17_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
WEIGHTS_DIR=$REPO/weights
VITB_WEIGHTS=$WEIGHTS_DIR/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

LOCAL_CACHE_DIR=/tmp/argus_run17_vitb_cache
TRAIN_ANN=$ANN_DIR/all_train_run17_merged.json
VAL_ANN=$ANN_DIR/val_run17_fits.json

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 17 — ViT-B Heatmap | 400px native tiles | zscore  $(date)"
echo " Backbone: ViT-B/16 (768-dim) vs Run 15 ViT-S/16 (384-dim)"
echo " All other hyperparameters held constant vs Run 15"
echo "================================================================"

# ── Step 1: Verify prerequisites ─────────────────────────────────────────────
echo ""
echo "── Step 1: Verify prerequisites ──"

[ -f "$VITB_WEIGHTS" ] || { echo "ERROR: missing $VITB_WEIGHTS"; exit 1; }
echo "  OK: $VITB_WEIGHTS"

# Build merged annotations from source FITS files (no NPY dependency)
echo "  Building merged train/val annotations from FITS sources..."
$PYTHON scripts/build_run17_annotations.py || { echo "ERROR: annotation build failed"; exit 1; }

for f in "$TRAIN_ANN" "$VAL_ANN"; do
  [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
  echo "  OK: $f"
done

echo ""
echo "Disk space before caching:"
df -h / /Volumes/External

# ── Step 2: Cache ViT-B features (400px tiles, zscore, internal SSD) ─────────
echo ""
echo "── Step 2a: Cache ViT-B train features (400px native tiles, zscore) ──"
echo "  Writing to: $LOCAL_CACHE_DIR/vitb_run17_train"
echo "  NOTE: ViT-B features are 768-dim (2× ViT-S). Expect ~2× cache size vs Run 15."
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$TRAIN_ANN" \
  --output-dir  "$LOCAL_CACHE_DIR/vitb_run17_train" \
  --backbone vit --model-size base \
  --weights "$VITB_WEIGHTS" \
  --image-size 518 --num-workers 0 \
  --native-tile-size 400 \
  --tile-overlap 0.0 \
  --norm-mode zscore

TRAIN_TILES=$(python3 -c "import json; d=json.load(open('$LOCAL_CACHE_DIR/vitb_run17_train/manifest.json')); print(len(d['manifest']))")
[ "$TRAIN_TILES" -gt 0 ] || { echo "ERROR: train cache wrote 0 tiles — check NPY symlink and external drive"; exit 1; }
echo "  Train cache: $TRAIN_TILES tiles OK"

echo ""
echo "── Step 2b: Cache ViT-B val features (400px native tiles, zscore) ──"
echo "  Writing to: $LOCAL_CACHE_DIR/vitb_run17_val"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$VAL_ANN" \
  --output-dir  "$LOCAL_CACHE_DIR/vitb_run17_val" \
  --backbone vit --model-size base \
  --weights "$VITB_WEIGHTS" \
  --image-size 518 --num-workers 0 \
  --native-tile-size 400 \
  --tile-overlap 0.0 \
  --norm-mode zscore

VAL_TILES=$(python3 -c "import json; d=json.load(open('$LOCAL_CACHE_DIR/vitb_run17_val/manifest.json')); print(len(d['manifest']))")
[ "$VAL_TILES" -gt 0 ] || { echo "ERROR: val cache wrote 0 tiles — external drive missing val1800/ subdirectory or NPY symlink broken"; exit 1; }
echo "  Val cache: $VAL_TILES tiles OK"

echo ""
echo "Cache sizes:"
du -sh "$LOCAL_CACHE_DIR/vitb_run17_train" "$LOCAL_CACHE_DIR/vitb_run17_val" 2>/dev/null || true
df -h /

# ── Step 3: Train ViT-B heatmap head ─────────────────────────────────────────
echo ""
echo "── Step 3: Train ViT-B head (40 epochs, batch=32, pos_weight=20, cosine LR) ──"
echo "  in_channels will be 768 (auto-detected from cache manifest)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/vitb_run17_train" \
  --val-cache   "$LOCAL_CACHE_DIR/vitb_run17_val" \
  --work-dir    "$WEIGHTS_DIR/run17_vitb" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

# ── Step 4: Free internal SSD ─────────────────────────────────────────────────
echo ""
echo "── Step 4: Free internal SSD cache ──"
rm -rf "$LOCAL_CACHE_DIR/vitb_run17_train" "$LOCAL_CACHE_DIR/vitb_run17_val"
rmdir "$LOCAL_CACHE_DIR" 2>/dev/null || true
df -h /
echo "Internal SSD freed."

# ── Step 5: Evaluate on val set (400px inference, zscore) ─────────────────────
echo ""
echo "── Step 5: Evaluate ViT-B on val_run12_1800_npy (t=0.05, 400px tiles, stitch) ──"
mkdir -p results/run17_vitb/t0.05
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
$PYTHON scripts/evaluate_dinov3_heatmap.py \
  --annotations "$VAL_ANN" \
  --checkpoint  "$WEIGHTS_DIR/run17_vitb/best.pt" \
  --output      results/run17_vitb/t0.05/metrics.json \
  --tiled --threshold 0.05 --stitch

# ── Step 6: Post-hoc threshold sweep ──────────────────────────────────────────
echo ""
echo "── Step 6: Threshold sweep ──"
$PYTHON scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run17_vitb/t0.05/predictions.json \
  --annotations "$VAL_ANN" \
  --output-dir  results/run17_vitb/threshold_sweep \
  --thresholds 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --iou-threshold 0.10 \
  --stitch || true

echo ""
echo "================================================================"
echo " Run 17 Complete  $(date)"
echo " Weights: $WEIGHTS_DIR/run17_vitb/best.pt"
echo " Results: results/run17_vitb/threshold_sweep/threshold_sweep.json"
echo "================================================================"

# Print summary and comparison to Run 15
$PYTHON - <<'PYEOF'
import json, sys
from pathlib import Path

sweep17 = Path("results/run17_vitb/threshold_sweep/threshold_sweep.json")
if not sweep17.exists():
    print("Threshold sweep not found — check eval step above.")
    sys.exit(0)

data = json.loads(sweep17.read_text())
rows = data if isinstance(data, list) else data.get("results", [])

print("\nRun 17 ViT-B threshold sweep (val_run12_1800, IoU>=0.10, stitch):")
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

# Compare to Run 15
sweep15 = Path("results/run15_vits/threshold_sweep/threshold_sweep.json")
if sweep15.exists():
    data15 = json.loads(sweep15.read_text())
    rows15 = data15 if isinstance(data15, list) else data15.get("results", [])
    best15 = max(rows15, key=lambda r: r.get("f1", r.get("F1", 0)))
    t15  = best15.get("threshold", "?")
    f115 = best15.get("f1", best15.get("F1", 0))
    mr15 = best15.get("medium_recall", best15.get("med_R", 0))
    lr15 = best15.get("long_recall", best15.get("long_R", 0))
    best17 = max(rows, key=lambda r: r.get("f1", r.get("F1", 0)))
    f117 = best17.get("f1", best17.get("F1", 0))
    mr17 = best17.get("medium_recall", best17.get("med_R", 0))
    lr17 = best17.get("long_recall", best17.get("long_R", 0))
    print(f"\nRun 15 ViT-S best (t={t15}): F1={100*f115:.1f}%  med_R={100*mr15:.1f}%  long_R={100*lr15:.1f}%")
    print(f"Run 17 ViT-B best:           F1={100*f117:.1f}%  med_R={100*mr17:.1f}%  long_R={100*lr17:.1f}%")
    print(f"Delta:                        F1={100*(f117-f115):+.1f}%  med_R={100*(mr17-mr15):+.1f}%  long_R={100*(lr17-lr15):+.1f}%")
PYEOF
