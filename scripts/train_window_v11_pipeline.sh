#!/usr/bin/env bash
# Train + eval vits_v11_asl_cldice on the coordinate-validated v11 dataset.
#
# What changed from v9:
#   - Source annotation: all_train_run17_merged_no_sattrains.json
#     (satellite-train frames excluded; no_sattrains suffix)
#   - Dataset build now validates OBB coordinates against tile bounds:
#     * full-frame coords saved in windowed tiles are translated to tile-local
#     * annotations genuinely outside the tile are dropped
#     Previously ~843 annotations (11% of windowed set) had wrong coords;
#     ~1535 were wrong-tile associations producing mislocated heatmap targets.
#   - Eval: val_balanced_v1_no_sattrains.json (241 annotations, perp_tol=20px)
#
# Baseline to beat: vits_v9_asl_cldice on val_balanced_v1_no_sattrains
#   Recall=0.988  Precision=0.988  short=1.000  med=0.990  long=0.974
#   Ang=0.50°  EndPx=8.9 (mean), 6.8 (median)
#
# Usage:
#   bash scripts/train_window_v11_pipeline.sh 2>&1 | tee /tmp/v11_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DATA=/Volumes/External/TrainingData
ANN_DIR=$DATA/annotations
BAL_ANN=$ANN_DIR/val_balanced_v1_no_sattrains.json
OUT=$REPO/results/window_v11
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

LR=1e-3; BATCH=32; HIDDEN=256; SCHED=cosine; EARLY_STOP=10
VITS_W=$WEIGHTS/dinov3_vits16_lvd1689m.pth
CACHE_ROOT=/Volumes/External/argus_caches
VITS_CACHE=$CACHE_ROOT/vits_v11

TAG=vits_v11_asl_cldice
VERSION=11

TRAIN_ANN="$DATA/train_atwood_synth_window_v${VERSION}/annotation.json"
VAL_ANN="$DATA/val_atwood_window_v${VERSION}/annotation.json"

echo "=== v11 pipeline | $(date) | vits asl_cldice + coord-validated dataset ==="

# ── Step 1: Build dataset v11 ─────────────────────────────────────────────────
echo "── Step 1: Building dataset v${VERSION} ── $(date)"
$PYTHON scripts/build_atwood_window_dataset.py \
  --version "$VERSION" \
  --dataset-root "$DATA" \
  --source "$ANN_DIR/all_train_run17_merged_no_sattrains.json" \
  --eval-frames-json "$BAL_ANN" \
  --val-frac 0.08 --neg-frac 0.42 --bg-per-frame 3 \
  --n-synth-short 400 --seed 42
echo "── Dataset v${VERSION} built ── $(date)"

TRAIN_TILES=$($PYTHON -c "import json; d=json.load(open('$TRAIN_ANN')); print(len(d['images']))")
echo "── Train tiles: $TRAIN_TILES ──"
[ "$TRAIN_TILES" -gt 0 ] || { echo "ERROR: dataset build wrote 0 train tiles"; exit 1; }

# ── Step 2: Cache ViT-S features ──────────────────────────────────────────────
echo "── Step 2: Caching ViT-S features ── $(date)"
rm -rf "$VITS_CACHE"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$TRAIN_ANN" \
  --output-dir "$VITS_CACHE/train" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$VAL_ANN" \
  --output-dir "$VITS_CACHE/val" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4

TILES=$($PYTHON -c "import json; d=json.load(open('$VITS_CACHE/train/manifest.json')); print(len(d['manifest']))")
[ "$TILES" -gt 0 ] || { echo "ERROR: train cache wrote 0 tiles"; exit 1; }
echo "── ViT-S feature cache complete ($TILES train tiles) ── $(date)"

# ── Step 3: Train ─────────────────────────────────────────────────────────────
echo "── Step 3: Training $TAG ── $(date)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$VITS_CACHE/train" --val-cache "$VITS_CACHE/val" \
  --work-dir "$WEIGHTS/$TAG" \
  --epochs 40 --lr "$LR" --batch-size "$BATCH" \
  --hidden-channels "$HIDDEN" --lr-scheduler "$SCHED" --num-workers 0 \
  --early-stopping-patience "$EARLY_STOP" \
  --loss-mode asl_cldice \
  --asl-gamma-neg 4.0 --asl-gamma-pos 0.0 --asl-margin 0.05 --cldice-iters 3
echo "── Training complete ── $(date)"

# ── Step 4: Delete feature cache ──────────────────────────────────────────────
rm -rf "$VITS_CACHE"
echo "── ViT-S cache deleted ── $(date)"

# ── Step 5: Eval on val_balanced_v1_no_sattrains ──────────────────────────────
echo "── Step 5: Eval $TAG ── $(date)"
BC=$CACHE_ROOT/${TAG}_balcache
$PYTHON scripts/cache_heatmap_maps.py \
  --annotations "$BAL_ANN" \
  --checkpoint "$WEIGHTS/$TAG/best.pt" \
  --output-dir "$BC" --norm-mode zscore
$PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
  --heatmap-cache "$BC" --annotations "$BAL_ANN" --norm-mode zscore \
  --threshold 0.05 --threshold-sweep 0.70 0.75 0.80 0.85 --peak-floor 0.85 \
  --profile-peak-fraction 0.85 \
  --output "$OUT/$TAG/pf85/metrics.json"
$PYTHON -m eval.geometry_metrics \
  --predictions "$OUT/$TAG/pf85/predictions_t085.json" \
  --annotations "$BAL_ANN" \
  --output "$OUT/$TAG/pf85/geometry_eval.json"
rm -rf "$BC"
echo "── Eval complete ── $(date)"

# ── Step 6: Side-by-side vs v9 baseline ──────────────────────────────────────
echo ""
echo "=== v11 vs v9 baseline (val_balanced_v1_no_sattrains, t=0.85, pf=0.85, perp_tol=20px) ==="
$PYTHON - <<'PYEOF'
import json, pathlib

variants = [
    ("vits_v9_asl_cldice (v9 baseline)", "results/window_v9/vits_v9_asl_cldice/pf85_no_sattrains/geometry_eval.json"),
    ("vits_v11_asl_cldice              ", "results/window_v11/vits_v11_asl_cldice/pf85/geometry_eval.json"),
]
hdr = f"{'Variant':<34}  {'Recall':>7}  {'Prec':>7}  {'Short':>6}  {'Med':>6}  {'Long':>6}  {'Ang°':>6}  {'EndPx':>7}"
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
echo "=== v11 pipeline complete | $(date) ==="
echo "    Weights: $WEIGHTS/$TAG/best.pt"
echo "    Geometry eval: $OUT/$TAG/pf85/geometry_eval.json"
