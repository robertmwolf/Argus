#!/usr/bin/env bash
# Run 10 recovery — restarts from NPY rebuild after the corrupted-cache incident.
#
# Root cause: run10_pipeline.sh deleted LOCAL_NPY_DIR before cache_dinov3_heatmap_features.py
# could read it. All caches produced blank images. Fixed in run10_pipeline.sh (NPY delete
# now happens after Step 4). This script reruns from Step 2 NPY conversion onward.
#
# What is already done and can be skipped:
#   - FITS copy to local SSD: GONE (deleted in Step 3) — we read from external instead
#   - annotation JSON builds (run10a/10b tiled + merged): DONE on external
#   - NPY conversion output JSONs: DONE (all_train_run10a_tiled_npy.json etc.) BUT
#     the .npy files themselves are gone; this script rewrites them to a fresh local dir
#   - Corrupted caches: DELETED before this script
#
# Usage:
#   bash scripts/run10_recovery.sh 2>&1 | tee /tmp/run10_recovery_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
EXT_CACHE_DIR=/Volumes/External/TrainingData/heatmap_cache
WEIGHTS_DIR=$REPO/weights

LOCAL_NPY_DIR=/tmp/argus_run10_npy
LOCAL_CACHE_DIR=/tmp/argus_run10_cache

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 10 Recovery  $(date)"
echo " Starting from NPY rebuild (reading FITS from external drive)"
echo "================================================================"

# ── Step 2 (redo): Build NPY tiles from external FITS ─────────────────────────
echo ""
echo "── Step 2 (redo): Convert 10a tiles to NPY (from external FITS) ──"
mkdir -p "$LOCAL_NPY_DIR/train10a" "$LOCAL_NPY_DIR/train10b"

$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ANN_DIR/all_train_run10a_tiled.json" \
  --output-dir  "$LOCAL_NPY_DIR/train10a" \
  --output-ann  "$ANN_DIR/all_train_run10a_tiled_npy.json"

echo ""
echo "── Step 2 (redo): Convert 10b tiles to NPY (from external FITS) ──"
$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ANN_DIR/all_train_run10b_tiled.json" \
  --output-dir  "$LOCAL_NPY_DIR/train10b" \
  --output-ann  "$ANN_DIR/all_train_run10b_tiled_npy.json"

# ── Step 4: Cache all 4 train feature sets → external ────────────────────────
# NPY files are now live in LOCAL_NPY_DIR — do NOT delete them until after Step 4.
echo ""
echo "── Step 4a: Cache ConvNeXt-S 10a train features → external ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run10a_tiled_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/convnext_run10a_train" \
  --backbone convnext --model-size small --convnext-stage 2 \
  --image-size 384 --num-workers 0

echo ""
echo "── Step 4b: Cache ViT-S 10a train features → external ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run10a_tiled_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/vits_run10a_train" \
  --backbone vit --model-size small \
  --image-size 384 --num-workers 0

echo ""
echo "── Step 4c: Cache ConvNeXt-S 10b train features → external ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run10b_tiled_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/convnext_run10b_train" \
  --backbone convnext --model-size small --convnext-stage 2 \
  --image-size 384 --num-workers 0

echo ""
echo "── Step 4d: Cache ViT-S 10b train features → external ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run10b_tiled_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/vits_run10b_train" \
  --backbone vit --model-size small \
  --image-size 384 --num-workers 0

# ── Delete NPY now that all caches are built ──────────────────────────────────
echo ""
echo "── Delete local NPY tiles ──"
rm -rf "$LOCAL_NPY_DIR"
echo "Local NPY deleted."

echo ""
echo "Val cache: reusing Run 9 (convnext_run9_val, vits_run9_val)"
mkdir -p "$LOCAL_CACHE_DIR"

# ── Train ConvNeXt-S 10a ──────────────────────────────────────────────────────
echo ""
echo "── Train ConvNeXt-S 10a (~38% neg) ──"
rsync -a "$EXT_CACHE_DIR/convnext_run10a_train/" "$LOCAL_CACHE_DIR/convnext_run10a_train/"
rsync -a "$EXT_CACHE_DIR/convnext_run9_val/"     "$LOCAL_CACHE_DIR/convnext_run9_val/"

$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/convnext_run10a_train" \
  --val-cache   "$LOCAL_CACHE_DIR/convnext_run9_val" \
  --work-dir    "$WEIGHTS_DIR/run10a_convnext_s2" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

rm -rf "$LOCAL_CACHE_DIR/convnext_run10a_train" "$LOCAL_CACHE_DIR/convnext_run9_val"

# ── Train ViT-S 10a ───────────────────────────────────────────────────────────
echo ""
echo "── Train ViT-S 10a (~38% neg) ──"
rsync -a "$EXT_CACHE_DIR/vits_run10a_train/" "$LOCAL_CACHE_DIR/vits_run10a_train/"
rsync -a "$EXT_CACHE_DIR/vits_run9_val/"     "$LOCAL_CACHE_DIR/vits_run9_val/"

$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/vits_run10a_train" \
  --val-cache   "$LOCAL_CACHE_DIR/vits_run9_val" \
  --work-dir    "$WEIGHTS_DIR/run10a_vits" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

rm -rf "$LOCAL_CACHE_DIR/vits_run10a_train" "$LOCAL_CACHE_DIR/vits_run9_val"

# ── Train ConvNeXt-S 10b ──────────────────────────────────────────────────────
echo ""
echo "── Train ConvNeXt-S 10b (~42% neg) ──"
rsync -a "$EXT_CACHE_DIR/convnext_run10b_train/" "$LOCAL_CACHE_DIR/convnext_run10b_train/"
rsync -a "$EXT_CACHE_DIR/convnext_run9_val/"     "$LOCAL_CACHE_DIR/convnext_run9_val/"

$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/convnext_run10b_train" \
  --val-cache   "$LOCAL_CACHE_DIR/convnext_run9_val" \
  --work-dir    "$WEIGHTS_DIR/run10b_convnext_s2" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

rm -rf "$LOCAL_CACHE_DIR/convnext_run10b_train" "$LOCAL_CACHE_DIR/convnext_run9_val"

# ── Train ViT-S 10b ───────────────────────────────────────────────────────────
echo ""
echo "── Train ViT-S 10b (~42% neg) ──"
rsync -a "$EXT_CACHE_DIR/vits_run10b_train/" "$LOCAL_CACHE_DIR/vits_run10b_train/"
rsync -a "$EXT_CACHE_DIR/vits_run9_val/"     "$LOCAL_CACHE_DIR/vits_run9_val/"

$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/vits_run10b_train" \
  --val-cache   "$LOCAL_CACHE_DIR/vits_run9_val" \
  --work-dir    "$WEIGHTS_DIR/run10b_vits" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

rm -rf "$LOCAL_CACHE_DIR/vits_run10b_train" "$LOCAL_CACHE_DIR/vits_run9_val"
rmdir "$LOCAL_CACHE_DIR" 2>/dev/null || true

# ── Eval all 4 models ─────────────────────────────────────────────────────────
echo ""
echo "── Eval ConvNeXt-S 10a (t=0.3, stitch) ──"
CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=400 CONVNEXT_HEATMAP_TILE_OVERLAP=0.5 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
"$PYTHON" scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run10a_convnext_s2/best.pt" \
  --output      results/run10a_convnext_s2/t0.3/metrics.json \
  --tiled --threshold 0.3 --stitch

"$PYTHON" scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run10a_convnext_s2/t0.3/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run10a_convnext_s2/threshold_sweep \
  --thresholds 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "── Eval ViT-S 10a (t=0.3, stitch) ──"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
"$PYTHON" scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run10a_vits/best.pt" \
  --output      results/run10a_vits/t0.3/metrics.json \
  --tiled --threshold 0.3 --stitch

"$PYTHON" scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run10a_vits/t0.3/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run10a_vits/threshold_sweep \
  --thresholds 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "── Eval ConvNeXt-S 10b (t=0.3, stitch) ──"
CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=400 CONVNEXT_HEATMAP_TILE_OVERLAP=0.5 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
"$PYTHON" scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run10b_convnext_s2/best.pt" \
  --output      results/run10b_convnext_s2/t0.3/metrics.json \
  --tiled --threshold 0.3 --stitch

"$PYTHON" scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run10b_convnext_s2/t0.3/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run10b_convnext_s2/threshold_sweep \
  --thresholds 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "── Eval ViT-S 10b (t=0.3, stitch) ──"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
"$PYTHON" scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run10b_vits/best.pt" \
  --output      results/run10b_vits/t0.3/metrics.json \
  --tiled --threshold 0.3 --stitch

"$PYTHON" scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run10b_vits/t0.3/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run10b_vits/threshold_sweep \
  --thresholds 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "================================================================"
echo " Run 10 Recovery Complete  $(date)"
echo " Weights: run10a_convnext_s2 / run10a_vits / run10b_convnext_s2 / run10b_vits"
echo " Results: results/run10{a,b}_{convnext_s2,vits}/threshold_sweep/"
echo " Internal SSD: fully freed"
echo "================================================================"
