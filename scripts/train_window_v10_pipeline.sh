#!/usr/bin/env bash
# Train + eval window_v10: asl_cldice loss on ViT-B backbone.
#
# v10 goal: compare ViT-B capacity against the v9 ViT-S asl_cldice winner.
# All hyperparameters held constant vs v9 asl_cldice except:
#   - backbone: ViT-B (vs ViT-S)
#   - epochs:   80 max (vs 40) — ViT-B head typically needs more steps
#
# Dataset: v9 dataset reused as-is (no rebuild). The v9 dataset was built with
#   val_frac=0.08, neg_frac=0.42, bg_per_frame=3, seed=42.
#
# Usage:
#   bash scripts/train_window_v10_pipeline.sh 2>&1 | tee /tmp/window_v10_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DATA=/Volumes/External/TrainingData
BAL_ANN=$REPO/data/annotations/val_balanced_v1.json
OUT=$REPO/results/window_v10
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

LR=1e-3; BATCH=32; HIDDEN=256; SCHED=cosine; EARLY_STOP=10
VITB_W=$WEIGHTS/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
CACHE_ROOT=/Volumes/External/argus_caches
VITB_CACHE=$CACHE_ROOT/vitb_window_v10

TAG=vitb_v10_asl_cldice

# v9 dataset (already built)
TRAIN_ANN="$DATA/train_atwood_synth_window_v9/annotation.json"
VAL_ANN="$DATA/val_atwood_window_v9/annotation.json"

echo "=== Window-v10 pipeline | $(date) | asl_cldice | ViT-B ==="

# ── Step 1: Cache ViT-B features ─────────────────────────────────────────────
echo "── Step 1: Caching ViT-B features ── $(date)"
rm -rf "$VITB_CACHE"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$TRAIN_ANN" \
  --output-dir "$VITB_CACHE/train" \
  --backbone vit --model-size base --weights "$VITB_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$VAL_ANN" \
  --output-dir "$VITB_CACHE/val" \
  --backbone vit --model-size base --weights "$VITB_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4

TILES=$($PYTHON -c "import json; d=json.load(open('$VITB_CACHE/train/manifest.json')); print(len(d['manifest']))")
[ "$TILES" -gt 0 ] || { echo "ERROR: train cache wrote 0 tiles"; exit 1; }
echo "── ViT-B feature cache complete ($TILES train tiles) ── $(date)"

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo "── Step 2: Training $TAG (asl_cldice, max 80ep) ── $(date)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$VITB_CACHE/train" --val-cache "$VITB_CACHE/val" \
  --work-dir "$WEIGHTS/$TAG" \
  --epochs 80 --lr "$LR" --batch-size "$BATCH" \
  --hidden-channels "$HIDDEN" \
  --lr-scheduler "$SCHED" --num-workers 0 \
  --early-stopping-patience "$EARLY_STOP" \
  --loss-mode asl_cldice \
  --asl-gamma-neg 4.0 --asl-gamma-pos 0.0 --asl-margin 0.05 --cldice-iters 3
echo "── Training complete ── $(date)"

# ── Step 3: Delete ViT-B cache ────────────────────────────────────────────────
rm -rf "$VITB_CACHE"
echo "── ViT-B cache deleted ── $(date)"

# ── Step 4: Eval on val_balanced_v1 ──────────────────────────────────────────
echo "── Step 4: Eval $TAG on val_balanced_v1 ── $(date)"
BC=/Volumes/External/argus_caches/${TAG}_balcache
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
echo "── Eval complete ── $(date)"

# ── Step 5: Side-by-side vs v9 ViT-S ─────────────────────────────────────────
echo ""
echo "=== v10 ViT-B vs v9 ViT-S asl_cldice (val_balanced_v1, t=0.70, pf=0.85) ==="
$PYTHON - <<'PYEOF'
import json, pathlib

variants = [
    ("vits_v9_asl_cldice (baseline)", "results/window_v9/vits_v9_asl_cldice/pf85/geometry_eval.json"),
    ("vitb_v10_asl_cldice           ", "results/window_v10/vitb_v10_asl_cldice/pf85/geometry_eval.json"),
]
hdr = f"{'Variant':<32}  {'Recall':>7}  {'Prec':>7}  {'Short':>6}  {'Med':>6}  {'Long':>6}  {'Ang°':>6}  {'EndPx':>7}"
print(hdr)
print("-" * len(hdr))
for name, path in variants:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"{name}  (missing)")
        continue
    d = json.loads(p.read_text())
    t1 = d["tier1_detection"]
    t2 = d["tier2_raw_geometry"]
    pb = t1["per_band"]
    print(f"{name}  {t1['detection_recall']:>7.3f}  {t1['detection_precision']:>7.3f}"
          f"  {pb['short']['recall']:>6.3f}  {pb['medium']['recall']:>6.3f}"
          f"  {pb['long']['recall']:>6.3f}"
          f"  {t2['angle_err_deg']['mean']:>6.2f}  {t2['endpoint_err_px']['mean']:>7.1f}")
PYEOF

echo ""
echo "=== Window-v10 complete | $(date) ==="
echo "    Weights: $WEIGHTS/$TAG/best.pt"
echo "    Eval:    $OUT/$TAG/pf85/geometry_eval.json"
