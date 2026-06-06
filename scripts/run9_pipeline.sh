#!/usr/bin/env bash
# Run 9 pipeline — fast-IO training + cosine LR scheduler.
#
# Key changes vs Run 8:
#   - Fast-IO: feature cache copied to local SSD for training, deleted after
#   - CosineAnnealingLR: replaces constant LR, expected to converge in ~35 epochs
#   - Reuse Run 8 val set (val_atwood_run8_tiled_npy.json) — already has negatives
#   - Reuse Run 8 train set (all_train_run8_tiled.json) — same negative ratio
#     unless the data strategy changes for Run 9
#
# Fast-IO stage order:
#   1. Copy source FITS to local temp dir
#   2. Build NPY tiles from local FITS
#   3. Build feature cache → external drive
#   4. Delete local FITS + local NPY (free space)
#   5. Copy feature cache external → local SSD
#   6. Train from local SSD
#   7. Delete local feature cache (external copy remains)
#
# Usage:
#   bash scripts/run9_pipeline.sh 2>&1 | tee /tmp/run9_$(date +%Y%m%d_%H%M%S).log
#
# Disk space needed on internal SSD (peak, steps 1–4):
#   Source FITS: ~100 GB (1998 unique parent images × ~50 MB each)
#   NPY tiles:   ~9 GB
#   Total peak:  ~109 GB  (FITS + NPY overlap; FITS deleted before cache copy)
#
# Disk space needed on internal SSD (steps 5–6):
#   Feature cache (ConvNeXt-S): ~26 GB
#   Feature cache (ViT-S):      ~26 GB
#   Only one backbone at a time — train ConvNeXt-S, delete cache, copy ViT-S, train.
#   Peak per backbone: ~26 GB

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
EXT_CACHE_DIR=/Volumes/External/TrainingData/heatmap_cache
EXT_NPY_DIR=/Volumes/External/TrainingData/tiles_npy
WEIGHTS_DIR=$REPO/weights

# Local SSD working directories — adjust if you want a different mount point
LOCAL_FITS_DIR=/tmp/argus_run9_fits
LOCAL_NPY_DIR=/tmp/argus_run9_npy
LOCAL_CACHE_DIR=/tmp/argus_run9_cache

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 9 Pipeline  $(date)"
echo "================================================================"

# ── Steps 1–3: Build data (reuse Run 8 annotation files if strategy unchanged) ──
# If annotation strategy changes, re-run build_tiled_brentimages_json.py first.
# Otherwise, skip to Step 4 using existing Run 8 tiled JSONs.

# ── Step 1: Copy source FITS to local temp ────────────────────────────────────
echo ""
echo "── Step 1: Extract unique FITS paths and copy to local SSD ──"
mkdir -p "$LOCAL_FITS_DIR"
$PYTHON - <<'PYEOF'
import json, os, shutil, sys
from pathlib import Path
import re

ANN_FILE = "/Volumes/External/TrainingData/annotations/all_train_run8_tiled.json"
LOCAL_DIR = "/tmp/argus_run9_fits"
TILE_RE = re.compile(r"^(.+?)__tx\d+_ty\d+_ts\d+$")

with open(ANN_FILE) as f:
    coco = json.load(f)

# Collect unique real source paths from virtual tile file_names
sources = set()
for img in coco["images"]:
    p = Path(img["file_name"])
    m = TILE_RE.match(p.stem)
    if m:
        real = str(p.parent / (m.group(1) + p.suffix))
        sources.add(real)
    else:
        sources.add(img["file_name"])

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
print(f"Done. {total} files copied to {LOCAL_DIR}")
PYEOF

# ── Step 2: Build NPY tiles from local FITS ───────────────────────────────────
echo ""
echo "── Step 2: Build NPY tiles from local FITS ──"
# Build Atwood tiled JSON pointing to local FITS
$PYTHON scripts/build_tiled_brentimages_json.py \
  --src data/annotations/all_train_run5.json \
  --out "$ANN_DIR/atwood_train_run9_tiled.json" \
  --native-tile-size 400 --overlap 0.5 \
  --neg-tiles-per-image 30 \
  --hard-neg-per-pos 3

# Merge with Frigate
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
        "info": {"description": "ARGUS Run 9 merged training split", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "streak", "supercategory": "satellite"}],
        "images": all_images,
        "annotations": all_annotations,
    }

ann_dir = "/Volumes/External/TrainingData/annotations"
with open(f"{ann_dir}/atwood_train_run9_tiled.json") as f:
    atwood = json.load(f)
with open(f"{ann_dir}/frigate_cluster2_run6_ts110.json") as f:
    frigate = json.load(f)

merged = merge([atwood, frigate])
out = f"{ann_dir}/all_train_run9_tiled.json"
with open(out, "w") as f:
    json.dump(merged, f)
ann_img_ids = {a["image_id"] for a in merged["annotations"]}
neg = len([i for i in merged["images"] if i["id"] not in ann_img_ids])
total = len(merged["images"])
print(f"Merged: {total} tiles | negative: {neg} ({100*neg/total:.1f}%)")
PYEOF

mkdir -p "$LOCAL_NPY_DIR/train" "$LOCAL_NPY_DIR/val"

$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ANN_DIR/all_train_run9_tiled.json" \
  --output-dir  "$LOCAL_NPY_DIR/train" \
  --output-ann  "$ANN_DIR/all_train_run9_tiled_npy.json" \
  --local-fits-dir /tmp/argus_run9_fits

# Val set: rebuild with more hard negatives (improves val_dice as FP signal)
$PYTHON scripts/build_tiled_brentimages_json.py \
  --src data/annotations/val_atwood.json \
  --out "$ANN_DIR/val_atwood_run9_tiled.json" \
  --native-tile-size 400 --overlap 0.5 \
  --hard-neg-per-pos 2 \
  --neg-tiles-per-image 10
$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ANN_DIR/val_atwood_run9_tiled.json" \
  --output-dir  "$LOCAL_NPY_DIR/val" \
  --output-ann  "$ANN_DIR/val_atwood_run9_tiled_npy.json" \
  --local-fits-dir /tmp/argus_run9_fits

# ── Step 3: Build feature caches → external drive ────────────────────────────
echo ""
echo "── Step 3: Cache ConvNeXt-S train features → external ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run9_tiled_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/convnext_run9_train" \
  --backbone convnext --model-size small --convnext-stage 2 \
  --image-size 384 --num-workers 0

echo ""
echo "── Step 3: Cache ConvNeXt-S val features → external ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/val_atwood_run9_tiled_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/convnext_run9_val" \
  --backbone convnext --model-size small --convnext-stage 2 \
  --image-size 384 --num-workers 0

echo ""
echo "── Step 3: Cache ViT-S train features → external ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run9_tiled_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/vits_run9_train" \
  --backbone vit --model-size small \
  --image-size 384 --num-workers 0

echo ""
echo "── Step 3: Cache ViT-S val features → external ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/val_atwood_run9_tiled_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/vits_run9_val" \
  --backbone vit --model-size small \
  --image-size 384 --num-workers 0

# ── Step 4: Delete local FITS + NPY (free space) ─────────────────────────────
echo ""
echo "── Step 4: Delete local FITS and NPY tiles ──"
rm -rf "$LOCAL_FITS_DIR" "$LOCAL_NPY_DIR"
echo "Local FITS and NPY deleted."

# ── Steps 5–7: ConvNeXt-S: copy cache → train → delete ───────────────────────
echo ""
echo "── Step 5: Copy ConvNeXt-S cache to local SSD ──"
mkdir -p "$LOCAL_CACHE_DIR"
rsync -a "$EXT_CACHE_DIR/convnext_run9_train/" "$LOCAL_CACHE_DIR/convnext_run9_train/"
rsync -a "$EXT_CACHE_DIR/convnext_run9_val/"   "$LOCAL_CACHE_DIR/convnext_run9_val/"

echo ""
echo "── Step 6: Train ConvNeXt-S from local SSD (cosine LR, 40 epochs) ──"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/convnext_run9_train" \
  --val-cache   "$LOCAL_CACHE_DIR/convnext_run9_val" \
  --work-dir    "$WEIGHTS_DIR/run9_convnext_s2" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

echo ""
echo "── Step 7: Delete local ConvNeXt-S cache ──"
rm -rf "$LOCAL_CACHE_DIR/convnext_run9_train" "$LOCAL_CACHE_DIR/convnext_run9_val"
echo "Local ConvNeXt-S cache deleted."

# ── Steps 5–7: ViT-S: copy cache → train → delete ────────────────────────────
echo ""
echo "── Step 5: Copy ViT-S cache to local SSD ──"
rsync -a "$EXT_CACHE_DIR/vits_run9_train/" "$LOCAL_CACHE_DIR/vits_run9_train/"
rsync -a "$EXT_CACHE_DIR/vits_run9_val/"   "$LOCAL_CACHE_DIR/vits_run9_val/"

echo ""
echo "── Step 6: Train ViT-S from local SSD (cosine LR, 40 epochs) ──"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/vits_run9_train" \
  --val-cache   "$LOCAL_CACHE_DIR/vits_run9_val" \
  --work-dir    "$WEIGHTS_DIR/run9_vits" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

echo ""
echo "── Step 7: Delete local ViT-S cache ──"
rm -rf "$LOCAL_CACHE_DIR/vits_run9_train" "$LOCAL_CACHE_DIR/vits_run9_val"
rmdir "$LOCAL_CACHE_DIR" 2>/dev/null || true
echo "Local ViT-S cache deleted."

echo ""
echo "── Step 8: Evaluate ConvNeXt-S (threshold=0.3, stitch) ──"
# Use threshold=0.3 to capture the full activation distribution for post-hoc sweep.
# Output to t0.3/ subdirectory to keep predictions.json distinct from any prior runs.
# See docs/training_methods.md §6.3 for rationale.
CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=400 CONVNEXT_HEATMAP_TILE_OVERLAP=0.5 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
"$PYTHON" scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run9_convnext_s2/best.pt" \
  --output      results/run9_convnext_s2/t0.3/metrics.json \
  --tiled --threshold 0.3 --stitch

echo ""
echo "── Step 8b: Threshold sweep (ConvNeXt-S) ──"
"$PYTHON" scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run9_convnext_s2/t0.3/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run9_convnext_s2/threshold_sweep \
  --thresholds 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true  # non-zero exit if no row meets gate; don't abort pipeline

echo ""
echo "── Step 9: Evaluate ViT-S (threshold=0.3, stitch) ──"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
"$PYTHON" scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run9_vits/best.pt" \
  --output      results/run9_vits/t0.3/metrics.json \
  --tiled --threshold 0.3 --stitch

echo ""
echo "── Step 9b: Threshold sweep (ViT-S) ──"
"$PYTHON" scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run9_vits/t0.3/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run9_vits/threshold_sweep \
  --thresholds 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "================================================================"
echo " Run 9 Complete  $(date)"
echo " Weights: $WEIGHTS_DIR/run9_convnext_s2/best.pt"
echo "          $WEIGHTS_DIR/run9_vits/best.pt"
echo " Eval:    results/run9_convnext_s2/metrics.json"
echo "          results/run9_vits/metrics.json"
echo " Sweep:   results/run9_convnext_s2/threshold_sweep/threshold_sweep.json"
echo "          results/run9_vits/threshold_sweep/threshold_sweep.json"
echo " Internal SSD: fully freed (back to baseline)"
echo " External: raw FITS + all feature caches preserved"
echo "================================================================"
