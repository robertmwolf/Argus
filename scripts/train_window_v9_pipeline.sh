#!/usr/bin/env bash
# Loss ablation study: 5 loss modes × ViT-S, single shared feature cache.
#
# v9 goal: find the loss function that best reduces false positives.
#
# Variants trained (all other hyperparameters held constant vs v8 ViT-S):
#   focal_dice   — focal(γ=2,α=0.85) + Dice          [v8 ViT-S baseline]
#   asl_dice     — ASL(γ_neg=4,m=0.05) + Dice         [precision-targeting]
#   focal_cldice — focal + clDice(iters=3)             [topology-aware]
#   tversky      — focal + Tversky(α_fp=0.6)           [FP-penalising Dice]
#   asl_cldice   — ASL + clDice                        [combined]
#
# Dataset: same recipe as v8 (calibration fix, val_frac=0.08, no hard negs).
# ViT-S feature cache is built ONCE and shared across all 5 training runs.
# ViT-B is not run here — add a v10 script for the winning variant only.
#
# Usage:
#   bash scripts/train_window_v9_pipeline.sh 2>&1 | tee /tmp/window_v9_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
DATA=/Volumes/External/TrainingData
BAL_ANN=$REPO/data/annotations/val_balanced_v1.json
MERGED_ANN=$REPO/data/annotations/all_train_run17_merged.json
OUT=$REPO/results/window_v9
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"; mkdir -p "$OUT"

FOCAL_GAMMA=2.0; FOCAL_ALPHA=0.85; EARLY_STOP=10
LR=1e-3; BATCH=32; GEOMW=0.25; HIDDEN=256; SCHED=cosine
VITS_W=$WEIGHTS/dinov3_vits16_lvd1689m.pth
CACHE_ROOT=/Volumes/External/argus_caches
VITS_CACHE=$CACHE_ROOT/vits_window_v9

echo "=== Window-v9 pipeline | $(date) | loss ablation (5 modes) | ViT-S only ==="
echo "    Variants: focal_dice  asl_dice  focal_cldice  tversky  asl_cldice"

# ── Step 1: Build v9 dataset ──────────────────────────────────────────────────
# Same recipe as v8: calibration-corrected frames, val_frac=0.08, no hard negs.
# A clean rebuild ensures the ablation is not confounded by dataset differences.
echo "── Step 1: Building v9 dataset (val_frac=0.08, no hard negs) ── $(date)"
$PYTHON scripts/build_atwood_window_dataset.py \
  --dataset-root "$DATA" \
  --version 9 \
  --source "$MERGED_ANN" \
  --val-frac 0.08 \
  --neg-frac 0.42 \
  --bg-per-frame 3 \
  --seed 42
TRAIN_ANN="$DATA/train_atwood_synth_window_v9/annotation.json"
VAL_ANN="$DATA/val_atwood_window_v9/annotation.json"
echo "── Dataset build complete ── $(date)"

# ── Step 2: Cache ViT-S features ONCE (shared across all 5 variants) ─────────
echo "── Step 2: Caching ViT-S features (shared cache) ── $(date)"
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

# Abort early if cache is empty (misconfigured path, etc.)
TILES=$($PYTHON -c "import json; d=json.load(open('$VITS_CACHE/train/manifest.json')); print(len(d['manifest']))")
[ "$TILES" -gt 0 ] || { echo "ERROR: train cache wrote 0 tiles"; exit 1; }
echo "── ViT-S feature cache complete ($TILES train tiles) ── $(date)"

# ── Helper: train one variant ─────────────────────────────────────────────────
# Args: $1=tag  $2=loss-mode  [remaining args passed to trainer]
train_variant() {
  local TAG="$1"; local MODE="$2"; shift 2
  echo "── Training $TAG (loss=$MODE) ── $(date)"
  $PYTHON training/train_dinov3_heatmap_cached.py \
    --train-cache "$VITS_CACHE/train" --val-cache "$VITS_CACHE/val" \
    --work-dir "$WEIGHTS/$TAG" \
    --epochs 40 --lr "$LR" --batch-size "$BATCH" \
    --geometry-weight "$GEOMW" --hidden-channels "$HIDDEN" \
    --lr-scheduler "$SCHED" --num-workers 0 \
    --early-stopping-patience "$EARLY_STOP" \
    --loss-mode "$MODE" \
    "$@"
  echo "── $TAG training complete ── $(date)"
}

# ── Helper: eval one variant ─────────────────────────────────────────────────
eval_variant() {
  local TAG="$1"
  echo "── Eval $TAG on val_balanced_v1 ── $(date)"
  local BC=~/argus_${TAG}_balcache
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
  echo "── $TAG eval complete ── $(date)"
}

# ── Step 3: Train all 5 variants (sequential, one MPS GPU) ───────────────────

# 3a: baseline — same recipe as v8 ViT-S
train_variant vits_v9_focal_dice focal_dice \
  --focal-gamma "$FOCAL_GAMMA" --focal-alpha "$FOCAL_ALPHA"

# 3b: Asymmetric Loss + Dice (hard-negative focus, easy negatives zeroed)
train_variant vits_v9_asl_dice asl_dice \
  --asl-gamma-neg 4.0 --asl-gamma-pos 0.0 --asl-margin 0.05

# 3c: focal + clDice (topology-preserving; rewards linear connectivity)
train_variant vits_v9_focal_cldice focal_cldice \
  --focal-gamma "$FOCAL_GAMMA" --focal-alpha "$FOCAL_ALPHA" --cldice-iters 3

# 3d: focal + Tversky (FP penalised 1.5× more than FN)
train_variant vits_v9_tversky tversky \
  --focal-gamma "$FOCAL_GAMMA" --focal-alpha "$FOCAL_ALPHA" --tversky-alpha 0.6

# 3e: ASL + clDice (combined precision-targeting + topology)
train_variant vits_v9_asl_cldice asl_cldice \
  --asl-gamma-neg 4.0 --asl-gamma-pos 0.0 --asl-margin 0.05 --cldice-iters 3

echo "── All 5 variants trained ── $(date)"

# ── Step 4: Eval all 5 variants ──────────────────────────────────────────────
for TAG in vits_v9_focal_dice vits_v9_asl_dice vits_v9_focal_cldice \
           vits_v9_tversky vits_v9_asl_cldice; do
  eval_variant "$TAG"
done

# ── Step 5: Delete shared ViT-S cache ────────────────────────────────────────
rm -rf "$VITS_CACHE"
echo "── ViT-S cache deleted ── $(date)"

# ── Step 6: Side-by-side comparison ──────────────────────────────────────────
echo ""
echo "=== v9 loss ablation results (val_balanced_v1, t=0.70, pf=0.85) ==="
$PYTHON - <<'PYEOF'
import json, pathlib

variants = [
    ("focal_dice  (baseline)", "results/window_v9/vits_v9_focal_dice/pf85/geometry_eval.json"),
    ("asl_dice              ", "results/window_v9/vits_v9_asl_dice/pf85/geometry_eval.json"),
    ("focal_cldice          ", "results/window_v9/vits_v9_focal_cldice/pf85/geometry_eval.json"),
    ("tversky               ", "results/window_v9/vits_v9_tversky/pf85/geometry_eval.json"),
    ("asl_cldice            ", "results/window_v9/vits_v9_asl_cldice/pf85/geometry_eval.json"),
]
hdr = f"{'Variant':<26}  {'Recall':>7}  {'Prec':>7}  {'Short':>6}  {'Med':>6}  {'Long':>6}  {'Ang°':>6}  {'EndPx':>7}"
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
echo "=== Window-v9 complete | $(date) ==="
echo "    Next: run the winning variant on ViT-B (train_window_v10_pipeline.sh)"
