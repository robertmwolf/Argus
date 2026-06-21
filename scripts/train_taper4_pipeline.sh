#!/usr/bin/env bash
# Train + eval vits_taper4_asl_cldice: ViT-S asl_cldice with endpoint taper (4px, half of taper8).
#
# Goal: reduce the systematic ~25px too-long endpoint bias measured in v9.
# The GT heatmap target now has a 16px cosine taper at each streak endpoint
# (ramps from 1.0 to 0.0 over the last patch width) to discourage the model
# from extending activation past the true endpoints.
#
# Baseline to beat: vits_v9_asl_cldice
#   Recall=0.979  Precision=0.918  Ang=0.48°  EndPx=21.2  (98% too long)
#
# All other hyperparameters identical to v9 asl_cldice:
#   - ViT-S backbone, image-size 518, native-tile 400, norm-mode none
#   - asl_cldice loss, lr=1e-3 cosine, batch=32, epochs=40, early-stop=10
#
# Usage:
#   bash scripts/train_taper4_pipeline.sh 2>&1 | tee /tmp/taper4_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DATA=/Volumes/External/TrainingData
BAL_ANN=/Volumes/External/TrainingData/annotations/val_balanced_v1.json
OUT=$REPO/results/taper4
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

LR=1e-3; BATCH=32; HIDDEN=256; SCHED=cosine; EARLY_STOP=10
VITS_W=$WEIGHTS/dinov3_vits16_lvd1689m.pth
CACHE_ROOT=/Volumes/External/argus_caches
VITS_CACHE=$CACHE_ROOT/vits_taper4

TAG=vits_taper4_asl_cldice

TRAIN_ANN="$DATA/train_atwood_synth_window_v9/annotation.json"
VAL_ANN="$DATA/val_atwood_window_v9/annotation.json"

echo "=== taper4 pipeline | $(date) | vits asl_cldice + endpoint_taper_px=4 ==="

# ── Step 1: Cache ViT-S features (with taper=16 baked into heatmap targets) ──
echo "── Step 1: Caching ViT-S features with endpoint taper ── $(date)"
rm -rf "$VITS_CACHE"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$TRAIN_ANN" \
  --output-dir "$VITS_CACHE/train" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4 \
  --endpoint-taper-px 4
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$VAL_ANN" \
  --output-dir "$VITS_CACHE/val" \
  --backbone vit --model-size small --weights "$VITS_W" \
  --image-size 518 --num-workers 0 --native-tile-size 400 \
  --tile-overlap 0.0 --norm-mode none --neg-tiles-per-image 4 \
  --endpoint-taper-px 4

TILES=$($PYTHON -c "import json; d=json.load(open('$VITS_CACHE/train/manifest.json')); print(len(d['manifest']))")
[ "$TILES" -gt 0 ] || { echo "ERROR: train cache wrote 0 tiles"; exit 1; }
echo "── ViT-S feature cache complete ($TILES train tiles) ── $(date)"

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo "── Step 2: Training $TAG ── $(date)"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$VITS_CACHE/train" --val-cache "$VITS_CACHE/val" \
  --work-dir "$WEIGHTS/$TAG" \
  --epochs 40 --lr "$LR" --batch-size "$BATCH" \
  --hidden-channels "$HIDDEN" --lr-scheduler "$SCHED" --num-workers 0 \
  --early-stopping-patience "$EARLY_STOP" \
  --loss-mode asl_cldice \
  --asl-gamma-neg 4.0 --asl-gamma-pos 0.0 --asl-margin 0.05 --cldice-iters 3
echo "── Training complete ── $(date)"

# ── Step 3: Delete feature cache ─────────────────────────────────────────────
rm -rf "$VITS_CACHE"
echo "── ViT-S cache deleted ── $(date)"

# ── Step 4: Eval on val_balanced_v1 ──────────────────────────────────────────
echo "── Step 4: Eval $TAG on val_balanced_v1 ── $(date)"
BC=/Volumes/External/argus_caches/${TAG}_balcache
$PYTHON scripts/cache_heatmap_maps.py --annotations "$BAL_ANN" \
  --checkpoint "$WEIGHTS/$TAG/best.pt" --output-dir "$BC" --norm-mode zscore
$PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
  --heatmap-cache "$BC" --annotations "$BAL_ANN" --norm-mode zscore \
  --threshold 0.05 --threshold-sweep 0.70 0.75 0.80 0.85 0.90 0.95 --peak-floor 0.85 \
  --output "$OUT/$TAG/pf85/metrics_placeholder.json"
$PYTHON scripts/summarize_balanced_eval.py "$OUT/$TAG/pf85"

# Geometry eval across all thresholds
for T in 070 075 080 085 090 095; do
  $PYTHON -m eval.geometry_metrics \
    --predictions "$OUT/$TAG/pf85/predictions_t${T}.json" \
    --annotations "$BAL_ANN" \
    --output "$OUT/$TAG/pf85/geometry_eval_t${T}.json"
done
# Canonical eval uses t070 (matches v9 baseline comparison)
cp "$OUT/$TAG/pf85/geometry_eval_t070.json" "$OUT/$TAG/pf85/geometry_eval.json"
rm -rf "$BC"
echo "── Geometry eval complete ── $(date)"

# ── Step 5: Endpoint error analysis across thresholds ────────────────────────
echo "── Step 5: Endpoint error analysis ── $(date)"
for T in 070 075 080 085 090 095; do
  $PYTHON scripts/analyze_endpoint_errors.py \
    --predictions "$OUT/$TAG/pf85/predictions_t${T}.json" \
    --annotations "$BAL_ANN" \
    --output "$OUT/$TAG/pf85/endpoint_error_analysis_t${T}.json" \
    --top-n 20
done
# Canonical endpoint analysis at t085 for comparison with taper16
cp "$OUT/$TAG/pf85/endpoint_error_analysis_t085.json" \
   "$OUT/$TAG/pf85/endpoint_error_analysis.json"
echo "── Endpoint error analysis complete ── $(date)"

# ── Step 6: Threshold sweep summary ──────────────────────────────────────────
echo ""
echo "=== taper4 threshold sweep (val_balanced_v1, pf=0.85) ==="
$PYTHON - <<'PYEOF'
import json, pathlib

thresholds = ["070", "075", "080", "085", "090", "095"]
hdr = f"{'Model / Threshold':<36}  {'Recall':>7}  {'Prec':>7}  {'Short':>6}  {'Med':>6}  {'Long':>6}  {'Ang°':>6}  {'EndPx':>7}"
print(hdr)
print("-" * len(hdr))

# v9 baseline at t070
p = pathlib.Path("results/window_v9/vits_v9_asl_cldice/pf85/geometry_eval.json")
if p.exists():
    d = json.loads(p.read_text())
    t1, t2, pb = d["tier1_detection"], d["tier2_raw_geometry"], d["tier1_detection"]["per_band"]
    print(f"{'vits_v9_asl_cldice   t=0.70 (baseline)':<36}  {t1['detection_recall']:>7.3f}  {t1['detection_precision']:>7.3f}"
          f"  {pb['short']['recall']:>6.3f}  {pb['medium']['recall']:>6.3f}  {pb['long']['recall']:>6.3f}"
          f"  {t2['angle_err_deg']['mean']:>6.2f}  {t2['endpoint_err_px']['mean']:>7.1f}")

print()
for t in thresholds:
    p = pathlib.Path(f"results/taper4/vits_taper4_asl_cldice/pf85/geometry_eval_t{t}.json")
    if not p.exists():
        print(f"  vits_taper4  t=0.{t}  (missing)")
        continue
    d = json.loads(p.read_text())
    t1, t2, pb = d["tier1_detection"], d["tier2_raw_geometry"], d["tier1_detection"]["per_band"]
    ep_path = pathlib.Path(f"results/taper4/vits_taper4_asl_cldice/pf85/endpoint_error_analysis_t{t}.json")
    ep_mean = "n/a"
    if ep_path.exists():
        ep = json.loads(ep_path.read_text())
        ep_mean = f"{ep['symmetric_endpoint_error']['all']['mean']:>7.1f}"
    print(f"  {'vits_taper4_asl_cldice  t=0.' + t:<34}  {t1['detection_recall']:>7.3f}  {t1['detection_precision']:>7.3f}"
          f"  {pb['short']['recall']:>6.3f}  {pb['medium']['recall']:>6.3f}  {pb['long']['recall']:>6.3f}"
          f"  {t2['angle_err_deg']['mean']:>6.2f}  {ep_mean}")
PYEOF

echo ""
echo "=== taper4 pipeline complete | $(date) ==="
echo "    Weights: $WEIGHTS/$TAG/best.pt"
echo "    Geometry evals: $OUT/$TAG/pf85/geometry_eval_t{070,075,080,085,090,095}.json"
echo "    Endpoint analyses: $OUT/$TAG/pf85/endpoint_error_analysis_t{070,075,080,085,090,095}.json"
