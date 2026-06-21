#!/usr/bin/env bash
# Train + eval vits_v10_no_sattrains_asl_cldice: ViT-S asl_cldice on satellite-train-cleaned dataset.
#
# Motivation: 53 BrentImages frames were identified as satellite train events —
#   images where ≥2 annotated streaks satisfy angle_diff < 5° AND perp_dist < 30px.
#   These are Starlink/constellation passes captured in a single frame.  Including
#   them in training is problematic because:
#     (a) Adjacent parallel streaks produce overlapping heatmaps, blurring the
#         model's understanding of individual streak boundaries.
#     (b) They are statistically rare in actual astronomical observing and not
#         representative of the target domain.
#   All 53 source frames are listed in:
#     /Volumes/External/TrainingData/annotations/sat_train_excluded.json
#
# Dataset: v9 tile dataset re-filtered to exclude the 34 sat-train frames
#   that were present at v9 build time.  New tile annotations at:
#     train: /Volumes/External/TrainingData/train_atwood_synth_window_v10_no_sattrains/
#     val:   /Volumes/External/TrainingData/val_atwood_window_v10_no_sattrains/
#   Tiles themselves (.npy files) are the same pre-extracted files; only the
#   annotation manifest has changed (2787→2742 train tiles, 205→202 val tiles).
#
# All hyperparameters identical to v9 asl_cldice (the current production model):
#   - ViT-S backbone, image-size 518, native-tile 400, norm-mode none
#   - asl_cldice loss, lr=1e-3 cosine, batch=32, epochs=40, early-stop=10
#   - NO endpoint taper (post-processing ppf=0.85 handles endpoint bias)
#
# Baseline to beat: vits_v9_asl_cldice
#   Recall=0.877  Precision=0.880  Ang=0.48°  EndPx=21.2  (98% too long, no ppf)
#   With ppf=0.85:  Recall=0.814  Precision=0.833  EndPx=6.6
#
# Usage:
#   bash scripts/train_v10_no_sattrains_pipeline.sh 2>&1 | tee /tmp/v10_no_sattrains_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DATA=/Volumes/External/TrainingData
BAL_ANN=/Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json
OUT=$REPO/results/v10_no_sattrains
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

LR=1e-3; BATCH=32; HIDDEN=256; SCHED=cosine; EARLY_STOP=10
VITS_W=$WEIGHTS/dinov3_vits16_lvd1689m.pth
CACHE_ROOT=/Volumes/External/argus_caches
VITS_CACHE=$CACHE_ROOT/vits_v10_no_sattrains

TAG=vits_v10_no_sattrains_asl_cldice

TRAIN_ANN="$DATA/train_atwood_synth_window_v10/annotation.json"
VAL_ANN="$DATA/val_atwood_window_v10/annotation.json"

echo "=== v10_no_sattrains pipeline | $(date) | vits asl_cldice | sat-train-cleaned dataset ==="
echo "    Train tiles: $(python -c "import json; d=json.load(open('$TRAIN_ANN')); print(len(d['images']))")"
echo "    Val tiles:   $(python -c "import json; d=json.load(open('$VAL_ANN')); print(len(d['images']))")"

# ── Step 1: Cache ViT-S features ─────────────────────────────────────────────
echo "── Step 1: Caching ViT-S features ── $(date)"
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

# ── Step 4: Eval on val_balanced_v1 with ppf ─────────────────────────────────
echo "── Step 4: Eval $TAG on val_balanced_v1 ── $(date)"
BC=/Volumes/External/argus_caches/${TAG}_balcache
$PYTHON scripts/cache_heatmap_maps.py --annotations "$BAL_ANN" \
  --checkpoint "$WEIGHTS/$TAG/best.pt" --output-dir "$BC" --norm-mode zscore

$PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
  --heatmap-cache "$BC" --annotations "$BAL_ANN" --norm-mode zscore \
  --threshold 0.05 --threshold-sweep 0.70 0.75 0.80 0.85 0.90 0.95 \
  --peak-floor 0.85 --profile-peak-fraction 0.85 \
  --output "$OUT/$TAG/pf85/metrics_placeholder.json"

$PYTHON scripts/summarize_balanced_eval.py "$OUT/$TAG/pf85"

# Geometry metrics at canonical threshold t=0.85
$PYTHON -m eval.geometry_metrics \
  --predictions "$OUT/$TAG/pf85/predictions_t085.json" \
  --annotations "$BAL_ANN" \
  --output "$OUT/$TAG/pf85/geometry_eval.json"

rm -rf "$BC"
echo "── Eval complete ── $(date)"

# ── Step 5: Endpoint error analysis at t=0.85 ────────────────────────────────
echo "── Step 5: Endpoint error analysis ── $(date)"
$PYTHON scripts/analyze_endpoint_errors.py \
  --predictions "$OUT/$TAG/pf85/predictions_t085.json" \
  --annotations "$BAL_ANN" \
  --output "$OUT/$TAG/pf85/endpoint_error_analysis.json" \
  --top-n 20
echo "── Endpoint error analysis complete ── $(date)"

# ── Step 6: Comparison vs v9 baseline ────────────────────────────────────────
echo ""
echo "=== v10_no_sattrains vs v9 baseline (val_balanced_v1, t=0.85, pf=0.85, ppf=0.85) ==="
$PYTHON - <<'PYEOF'
import json, pathlib

variants = [
    ("vits_v9_asl_cldice (all data, ppf=0.85, no-sattrains val)",
     "results/window_v9/vits_v9_asl_cldice/pf85_no_sattrains/geometry_eval.json",
     "results/window_v9/vits_v9_asl_cldice/pf85_no_sattrains/endpoint_error_analysis.json"),
    ("vits_v10_no_sattrains_asl_cldice (ppf=0.85)",
     "results/v10_no_sattrains/vits_v10_no_sattrains_asl_cldice/pf85/geometry_eval.json",
     "results/v10_no_sattrains/vits_v10_no_sattrains_asl_cldice/pf85/endpoint_error_analysis.json"),
]
hdr = f"{'Variant':<44}  {'Recall':>7}  {'Prec':>7}  {'Short':>6}  {'Med':>6}  {'Long':>6}  {'Ang°':>6}  {'EndPx':>7}"
print(hdr); print("-" * len(hdr))
for name, geo_path, ep_path in variants:
    p = pathlib.Path(geo_path)
    if not p.exists():
        print(f"{name}  (missing)"); continue
    d = json.loads(p.read_text())
    t1 = d["tier1_detection"]; t2 = d["tier2_raw_geometry"]; pb = t1["per_band"]
    ep_str = "n/a"
    if ep_path:
        ep_p = pathlib.Path(ep_path)
        if ep_p.exists():
            ep = json.loads(ep_p.read_text())
            try:
                ep_str = f"{ep['aggregate']['symmetric_endpoint_err_px']['mean']:>7.1f}"
            except (KeyError, TypeError):
                ep_str = "n/a"
    print(f"{name:<44}  {t1['detection_recall']:>7.3f}  {t1['detection_precision']:>7.3f}"
          f"  {pb['short']['recall']:>6.3f}  {pb['medium']['recall']:>6.3f}  {pb['long']['recall']:>6.3f}"
          f"  {t2['angle_err_deg']['mean']:>6.2f}  {ep_str}")
PYEOF

echo ""
echo "=== v10_no_sattrains pipeline complete | $(date) ==="
echo "    Weights: $WEIGHTS/$TAG/best.pt"
echo "    Geometry eval: $OUT/$TAG/pf85/geometry_eval.json"
echo "    Endpoint analysis: $OUT/$TAG/pf85/endpoint_error_analysis.json"
