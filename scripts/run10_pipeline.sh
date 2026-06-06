#!/usr/bin/env bash
# Run 10 pipeline — backbone comparison at 38% and 42% negative ratios.
#
# Goal: find the maximum negative tile ratio at which both ViT-S and
# ConvNeXt-S preserve recall, using cosine LR (proven in Run 9).
#
# Four training runs:
#   10a-convnext: ~38% negatives, ConvNeXt-S
#   10a-vits:     ~38% negatives, ViT-S
#   10b-convnext: ~42% negatives, ConvNeXt-S
#   10b-vits:     ~42% negatives, ViT-S
#
# Val set: reuse Run 9 val cache (32% negatives, honest val_dice).
#
# Data parametres (from assistant_guide.md §Run 10):
#   38%: --neg-tiles-per-image 28 --hard-neg-per-pos 2
#   42%: --neg-tiles-per-image 35 --hard-neg-per-pos 2
#
# Fast-IO order (minimises SSD peak usage):
#   1. Copy source FITS to local SSD
#   2. Build + NPY-convert 10a and 10b annotation files
#   3. Delete local FITS + NPY
#   4. Cache all 4 train feature sets → external drive
#   5–8. For each of the 4 models: copy cache → train → delete local cache
#   9. Eval + threshold sweep all 4 models
#
# Usage:
#   bash scripts/run10_pipeline.sh 2>&1 | tee /tmp/run10_$(date +%Y%m%d_%H%M%S).log
#
# Disk space on internal SSD (peak, step 1):
#   Source FITS: ~100 GB
#   NPY tiles:   ~9 GB  (both 10a and 10b share same positive tiles; negatives differ)
# Disk space per training step (steps 5–8):
#   Feature cache: ~26 GB per backbone (train only; val reused from Run 9)

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
EXT_CACHE_DIR=/Volumes/External/TrainingData/heatmap_cache
WEIGHTS_DIR=$REPO/weights

LOCAL_FITS_DIR=/tmp/argus_run10_fits
LOCAL_NPY_DIR=/tmp/argus_run10_npy
LOCAL_CACHE_DIR=/tmp/argus_run10_cache

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 10 Pipeline  $(date)"
echo " 4 models: ConvNeXt-S + ViT-S × 38% + 42% negatives"
echo "================================================================"

# ── Step 1: Copy source FITS to local SSD ────────────────────────────────────
echo ""
echo "── Step 1: Copy source FITS to local SSD ──"
mkdir -p "$LOCAL_FITS_DIR"
$PYTHON - <<'PYEOF'
import json, shutil, sys
from pathlib import Path
import re

ANN_FILE = "/Users/robert/Argus/data/annotations/all_train_run5.json"
LOCAL_DIR = "/tmp/argus_run10_fits"
TILE_RE = re.compile(r"^(.+?)__tx\d+_ty\d+_ts\d+$")

with open(ANN_FILE) as f:
    coco = json.load(f)

sources = set()
for img in coco["images"]:
    p = Path(img["file_name"])
    m = TILE_RE.match(p.stem)
    real = str(p.parent / (m.group(1) + p.suffix)) if m else img["file_name"]
    sources.add(real)

print(f"Unique source files: {len(sources)}")
total = 0
for i, src in enumerate(sorted(sources), 1):
    src_path = Path(src)
    if not src_path.exists():
        print(f"  MISSING: {src}", file=sys.stderr)
        continue
    dst = Path(LOCAL_DIR) / src_path.name
    if not dst.exists():
        shutil.copy2(src, dst)
    total += 1
    if i % 200 == 0:
        print(f"  Copied {i}/{len(sources)}...")
print(f"Done. {total} files in {LOCAL_DIR}")
PYEOF

# ── Step 2a: Build 10a annotation files (~38% negatives) ─────────────────────
echo ""
echo "── Step 2a: Build Run 10a tiled JSON (~38% negatives) ──"
mkdir -p "$LOCAL_NPY_DIR/train10a" "$LOCAL_NPY_DIR/train10b"

$PYTHON scripts/build_tiled_brentimages_json.py \
  --src data/annotations/all_train_run5.json \
  --out "$ANN_DIR/atwood_train_run10a_tiled.json" \
  --native-tile-size 400 --overlap 0.5 \
  --hard-neg-per-pos 2 --neg-tiles-per-image 28

$PYTHON - <<'PYEOF'
import json

def merge(sources):
    all_images, all_annotations = [], []
    next_img_id = next_ann_id = 1
    for src in sources:
        old_to_new = {}
        for img in src["images"]:
            new_img = dict(img)
            old_to_new[img["id"]] = next_img_id
            new_img["id"] = next_img_id
            all_images.append(new_img)
            next_img_id += 1
        for ann in src.get("annotations", []):
            if ann["image_id"] not in old_to_new:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new[ann["image_id"]]
            all_annotations.append(new_ann)
            next_ann_id += 1
    return {
        "info": {"description": "ARGUS Run 10a merged training split (~38% negatives)", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "streak", "supercategory": "satellite"}],
        "images": all_images,
        "annotations": all_annotations,
    }

ann_dir = "/Volumes/External/TrainingData/annotations"
with open(f"{ann_dir}/atwood_train_run10a_tiled.json") as f:
    atwood = json.load(f)
with open(f"{ann_dir}/frigate_cluster2_run6_ts110.json") as f:
    frigate = json.load(f)

merged = merge([atwood, frigate])
out = f"{ann_dir}/all_train_run10a_tiled.json"
with open(out, "w") as f:
    json.dump(merged, f)
ann_img_ids = {a["image_id"] for a in merged["annotations"]}
neg = len([i for i in merged["images"] if i["id"] not in ann_img_ids])
total = len(merged["images"])
print(f"Run 10a merged: {total} tiles | negative: {neg} ({100*neg/total:.1f}%)")
PYEOF

$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ANN_DIR/all_train_run10a_tiled.json" \
  --output-dir  "$LOCAL_NPY_DIR/train10a" \
  --output-ann  "$ANN_DIR/all_train_run10a_tiled_npy.json" \
  --local-fits-dir "$LOCAL_FITS_DIR"

# ── Step 2b: Build 10b annotation files (~42% negatives) ─────────────────────
echo ""
echo "── Step 2b: Build Run 10b tiled JSON (~42% negatives) ──"

$PYTHON scripts/build_tiled_brentimages_json.py \
  --src data/annotations/all_train_run5.json \
  --out "$ANN_DIR/atwood_train_run10b_tiled.json" \
  --native-tile-size 400 --overlap 0.5 \
  --hard-neg-per-pos 2 --neg-tiles-per-image 35

$PYTHON - <<'PYEOF'
import json

def merge(sources):
    all_images, all_annotations = [], []
    next_img_id = next_ann_id = 1
    for src in sources:
        old_to_new = {}
        for img in src["images"]:
            new_img = dict(img)
            old_to_new[img["id"]] = next_img_id
            new_img["id"] = next_img_id
            all_images.append(new_img)
            next_img_id += 1
        for ann in src.get("annotations", []):
            if ann["image_id"] not in old_to_new:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new[ann["image_id"]]
            all_annotations.append(new_ann)
            next_ann_id += 1
    return {
        "info": {"description": "ARGUS Run 10b merged training split (~42% negatives)", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "streak", "supercategory": "satellite"}],
        "images": all_images,
        "annotations": all_annotations,
    }

ann_dir = "/Volumes/External/TrainingData/annotations"
with open(f"{ann_dir}/atwood_train_run10b_tiled.json") as f:
    atwood = json.load(f)
with open(f"{ann_dir}/frigate_cluster2_run6_ts110.json") as f:
    frigate = json.load(f)

merged = merge([atwood, frigate])
out = f"{ann_dir}/all_train_run10b_tiled.json"
with open(out, "w") as f:
    json.dump(merged, f)
ann_img_ids = {a["image_id"] for a in merged["annotations"]}
neg = len([i for i in merged["images"] if i["id"] not in ann_img_ids])
total = len(merged["images"])
print(f"Run 10b merged: {total} tiles | negative: {neg} ({100*neg/total:.1f}%)")
PYEOF

$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ANN_DIR/all_train_run10b_tiled.json" \
  --output-dir  "$LOCAL_NPY_DIR/train10b" \
  --output-ann  "$ANN_DIR/all_train_run10b_tiled_npy.json" \
  --local-fits-dir "$LOCAL_FITS_DIR"

# ── Step 3: Delete local FITS only (NPY kept until after caching) ─────────────
echo ""
echo "── Step 3: Delete local FITS ──"
rm -rf "$LOCAL_FITS_DIR"
echo "Local FITS deleted. NPY tiles kept for feature caching."

# ── Step 4: Build all 4 feature caches → external ────────────────────────────
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

# ── Step 4e: Delete local NPY now that all caches are built ──────────────────
echo ""
echo "── Step 4e: Delete local NPY tiles ──"
rm -rf "$LOCAL_NPY_DIR"
echo "Local NPY deleted."

# Val cache reused from Run 9 (convnext_run9_val and vits_run9_val)
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
echo "ConvNeXt-S 10a cache deleted."

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
echo "ViT-S 10a cache deleted."

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
echo "ConvNeXt-S 10b cache deleted."

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
echo "ViT-S 10b cache deleted."

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
echo " Run 10 Complete  $(date)"
echo ""
echo " Weights:"
echo "   $WEIGHTS_DIR/run10a_convnext_s2/best.pt"
echo "   $WEIGHTS_DIR/run10a_vits/best.pt"
echo "   $WEIGHTS_DIR/run10b_convnext_s2/best.pt"
echo "   $WEIGHTS_DIR/run10b_vits/best.pt"
echo ""
echo " Results:"
echo "   results/run10a_convnext_s2/threshold_sweep/"
echo "   results/run10a_vits/threshold_sweep/"
echo "   results/run10b_convnext_s2/threshold_sweep/"
echo "   results/run10b_vits/threshold_sweep/"
echo ""
echo " Internal SSD: fully freed"
echo " External: raw FITS + all feature caches preserved"
echo "================================================================"
