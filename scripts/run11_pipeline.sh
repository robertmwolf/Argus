#!/usr/bin/env bash
# Run 11 pipeline — ViT-S + synthetic streak augmentation.
#
# Hypothesis: frozen ViT-S recall is bottlenecked by too few positive examples
# with diverse geometry/SNR. Adding synthetic streaks (Gaussian PSF, random
# angle/length/SNR) onto real negative tiles should give the model more signal.
#
# Dataset:
#   Base: Run 10a tiled annotations (9,343 pos + 6,715 neg = 38.3% neg)
#   Synthetic: 1 augmented tile per negative → +6,715 synthetic positives
#   Merged: 16,058 pos + 6,715 neg = 29.5% neg  (near Run 8's 34% sweet spot)
#
# Backbone: ViT-S only (ConvNeXt-S permanently dropped — zero medium recall).
# Val cache: reuse Run 9 vits_run9_val.
#
# Usage:
#   bash scripts/run11_pipeline.sh 2>&1 | tee /tmp/run11_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
EXT_CACHE_DIR=/Volumes/External/TrainingData/heatmap_cache
WEIGHTS_DIR=$REPO/weights

LOCAL_NPY_DIR=/tmp/argus_run11_npy
LOCAL_CACHE_DIR=/tmp/argus_run11_cache

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 11 Pipeline  $(date)"
echo " ViT-S + synthetic streak augmentation"
echo "================================================================"

# ── Step 1: Rebuild NPY tiles from external FITS ──────────────────────────────
echo ""
echo "── Step 1: Convert Run 10a tiles to NPY (from external FITS) ──"
mkdir -p "$LOCAL_NPY_DIR/real"

$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ANN_DIR/all_train_run10a_tiled.json" \
  --output-dir  "$LOCAL_NPY_DIR/real" \
  --output-ann  "$ANN_DIR/all_train_run10a_tiled_npy.json"

# ── Step 2: Generate synthetic streak tiles ───────────────────────────────────
echo ""
echo "── Step 2: Generate synthetic streak tiles (1 per negative) ──"
mkdir -p "$LOCAL_NPY_DIR/synthetic"

$PYTHON scripts/generate_synthetic_streaks.py \
  --neg-annotations "$ANN_DIR/all_train_run10a_tiled_npy.json" \
  --output-dir      "$LOCAL_NPY_DIR/synthetic" \
  --output-ann      "$ANN_DIR/synth_run11_npy.json" \
  --n-per-neg 1 \
  --snr-min 3.0 --snr-max 12.0 \
  --multi-streak-prob 0.15 \
  --seed 42

# ── Step 3: Merge real + synthetic annotations ────────────────────────────────
echo ""
echo "── Step 3: Merge real + synthetic annotations ──"
$PYTHON - <<'PYEOF'
import json
from pathlib import Path

ann_dir = "/Volumes/External/TrainingData/annotations"

with open(f"{ann_dir}/all_train_run10a_tiled_npy.json") as f:
    real = json.load(f)
with open(f"{ann_dir}/synth_run11_npy.json") as f:
    synth = json.load(f)

# Remap IDs to avoid collisions
next_img_id = max(i["id"] for i in real["images"]) + 1
next_ann_id = max((a["id"] for a in real["annotations"]), default=0) + 1

merged_images = list(real["images"])
merged_annotations = list(real["annotations"])

for img in synth["images"]:
    new_img = dict(img)
    old_id = img["id"]
    new_img["id"] = next_img_id
    merged_images.append(new_img)

    for ann in synth["annotations"]:
        if ann["image_id"] == old_id:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = next_img_id
            merged_annotations.append(new_ann)
            next_ann_id += 1

    next_img_id += 1

ann_ids = {a["image_id"] for a in merged_annotations}
neg = len([i for i in merged_images if i["id"] not in ann_ids])
pos = len([i for i in merged_images if i["id"] in ann_ids])
total = len(merged_images)
print(f"Merged: {total} tiles | pos: {pos} | neg: {neg} ({100*neg/total:.1f}% neg)")

out = {
    "info": {"description": "ARGUS Run 11 merged (real + synthetic)", "version": "1.0"},
    "licenses": [],
    "categories": real["categories"],
    "images": merged_images,
    "annotations": merged_annotations,
}
with open(f"{ann_dir}/all_train_run11_npy.json", "w") as f:
    json.dump(out, f)
print(f"Written: {ann_dir}/all_train_run11_npy.json")
PYEOF

# ── Step 4: Cache ViT-S features → external ───────────────────────────────────
# NPY files (real + synthetic) are live in LOCAL_NPY_DIR during this step.
echo ""
echo "── Step 4: Cache ViT-S Run 11 train features → external ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run11_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/vits_run11_train" \
  --backbone vit --model-size small \
  --image-size 384 --num-workers 0

# ── Step 5: Delete local NPY ──────────────────────────────────────────────────
echo ""
echo "── Step 5: Delete local NPY tiles ──"
rm -rf "$LOCAL_NPY_DIR"
echo "Local NPY deleted."

echo "Val cache: reusing vits_run9_val"

# ── Step 6: Copy cache to local SSD and train ─────────────────────────────────
echo ""
echo "── Step 6: Copy ViT-S cache to local SSD ──"
mkdir -p "$LOCAL_CACHE_DIR"
rsync -a "$EXT_CACHE_DIR/vits_run11_train/" "$LOCAL_CACHE_DIR/vits_run11_train/"
rsync -a "$EXT_CACHE_DIR/vits_run9_val/"    "$LOCAL_CACHE_DIR/vits_run9_val/"

echo ""
echo "── Step 6: Train ViT-S Run 11 (cosine LR, 40 epochs) ──"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/vits_run11_train" \
  --val-cache   "$LOCAL_CACHE_DIR/vits_run9_val" \
  --work-dir    "$WEIGHTS_DIR/run11_vits" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

echo ""
echo "── Step 7: Delete local cache ──"
rm -rf "$LOCAL_CACHE_DIR/vits_run11_train" "$LOCAL_CACHE_DIR/vits_run9_val"
rmdir "$LOCAL_CACHE_DIR" 2>/dev/null || true

# ── Step 8: Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "── Step 8: Evaluate ViT-S Run 11 (t=0.3, stitch) ──"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
"$PYTHON" scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run11_vits/best.pt" \
  --output      results/run11_vits/t0.3/metrics.json \
  --tiled --threshold 0.3 --stitch

echo ""
echo "── Step 8b: Threshold sweep ──"
"$PYTHON" scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run11_vits/t0.3/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run11_vits/threshold_sweep \
  --thresholds 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "================================================================"
echo " Run 11 Complete  $(date)"
echo " Weights: $WEIGHTS_DIR/run11_vits/best.pt"
echo " Results: results/run11_vits/threshold_sweep/threshold_sweep.json"
echo "================================================================"
