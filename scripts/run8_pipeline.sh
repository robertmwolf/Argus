#!/usr/bin/env bash
# Run 8 overnight pipeline — fix val metric + rebalance negatives.
#
# Three-part fix vs Run 7:
#   1. Val set now includes negative tiles (was 97% positive → blind to over-prediction)
#   2. Negative ratio 40–45% (vs Run 6's 60% which collapsed recall, Run 7's 27% which lost precision)
#   3. pos_weight=20 (reverted from Run 7's 50)
#
# Runs both ConvNeXt-S and ViT-S backbones sequentially.
# Expected wall time: ~6–10 hours total depending on hardware.
#
# Usage:
#   bash scripts/run8_pipeline.sh
#   # or with a log:
#   bash scripts/run8_pipeline.sh 2>&1 | tee /tmp/run8_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
NPY_DIR=/Volumes/External/TrainingData/tiles_npy
CACHE_DIR=/Volumes/External/TrainingData/heatmap_cache
WEIGHTS_DIR=$REPO/weights

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 8 Pipeline  $(date)"
echo "================================================================"

# ── Step 1: Val JSON with negative tiles ─────────────────────────────
echo ""
echo "── Step 1: Build val tiled JSON with negative tiles ──"
$PYTHON scripts/build_tiled_brentimages_json.py \
  --src data/annotations/val_atwood.json \
  --out "$ANN_DIR/val_atwood_tiled_400_with_neg.json" \
  --native-tile-size 400 --overlap 0.5 \
  --neg-tiles-per-image 5

# ── Step 2: Train JSON at 40–45% negatives ───────────────────────────
echo ""
echo "── Step 2: Build train tiled JSON (neg-per-image=25, hard-neg-per-pos=1) ──"
$PYTHON scripts/build_tiled_brentimages_json.py \
  --src data/annotations/all_train_run5.json \
  --out "$ANN_DIR/atwood_train_run8_tiled.json" \
  --native-tile-size 400 --overlap 0.5 \
  --neg-tiles-per-image 25 \
  --hard-neg-per-pos 1

# ── Step 3: Merge Atwood + Frigate ───────────────────────────────────
echo ""
echo "── Step 3: Merge Atwood train + Frigate ──"
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
        "info": {"description": "ARGUS Run 8 merged training split", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "streak", "supercategory": "satellite"}],
        "images": all_images,
        "annotations": all_annotations,
    }

ann_dir = "/Volumes/External/TrainingData/annotations"
with open(f"{ann_dir}/atwood_train_run8_tiled.json") as f:
    atwood = json.load(f)
with open(f"{ann_dir}/frigate_cluster2_run6_ts110.json") as f:
    frigate = json.load(f)

merged = merge([atwood, frigate])
out = f"{ann_dir}/all_train_run8_tiled.json"
with open(out, "w") as f:
    json.dump(merged, f)

ann_img_ids = {a["image_id"] for a in merged["annotations"]}
neg = len([i for i in merged["images"] if i["id"] not in ann_img_ids])
total = len(merged["images"])
print(f"Merged: {total} tiles | positive: {total - neg} | negative: {neg} ({100*neg/total:.1f}%)")
print(f"Written: {out}")
PYEOF

# ── Step 4: NPY convert train ─────────────────────────────────────────
echo ""
echo "── Step 4: NPY convert train ──"
$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ANN_DIR/all_train_run8_tiled.json" \
  --output-dir  "$NPY_DIR/run8_train" \
  --output-ann  "$ANN_DIR/all_train_run8_tiled_npy.json"

# ── Step 5: NPY convert val ───────────────────────────────────────────
echo ""
echo "── Step 5: NPY convert val ──"
$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ANN_DIR/val_atwood_tiled_400_with_neg.json" \
  --output-dir  "$NPY_DIR/run8_val" \
  --output-ann  "$ANN_DIR/val_atwood_run8_tiled_npy.json"

# ── Step 6: Cache ConvNeXt-S train features ───────────────────────────
echo ""
echo "── Step 6: Cache ConvNeXt-S train features ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run8_tiled_npy.json" \
  --output-dir  "$CACHE_DIR/convnext_run8_train" \
  --backbone convnext --model-size small --convnext-stage 2 \
  --image-size 384 --num-workers 0

# ── Step 7: Cache ConvNeXt-S val features ────────────────────────────
echo ""
echo "── Step 7: Cache ConvNeXt-S val features ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/val_atwood_run8_tiled_npy.json" \
  --output-dir  "$CACHE_DIR/convnext_run8_val" \
  --backbone convnext --model-size small --convnext-stage 2 \
  --image-size 384 --num-workers 0

# ── Step 8: Cache ViT-S train features ───────────────────────────────
echo ""
echo "── Step 8: Cache ViT-S train features ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run8_tiled_npy.json" \
  --output-dir  "$CACHE_DIR/vits_run8_train" \
  --backbone vit --model-size small \
  --image-size 384 --num-workers 0

# ── Step 9: Cache ViT-S val features ─────────────────────────────────
echo ""
echo "── Step 9: Cache ViT-S val features ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/val_atwood_run8_tiled_npy.json" \
  --output-dir  "$CACHE_DIR/vits_run8_val" \
  --backbone vit --model-size small \
  --image-size 384 --num-workers 0

# ── Step 10: Train ConvNeXt-S ─────────────────────────────────────────
echo ""
echo "── Step 10: Train ConvNeXt-S  (pos_weight=20, 60 epochs) ──"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$CACHE_DIR/convnext_run8_train" \
  --val-cache   "$CACHE_DIR/convnext_run8_val" \
  --work-dir    "$WEIGHTS_DIR/run8_convnext_s2" \
  --epochs 60 --batch-size 32 --pos-weight 20 --num-workers 0

# ── Step 11: Train ViT-S ──────────────────────────────────────────────
echo ""
echo "── Step 11: Train ViT-S  (pos_weight=20, 60 epochs) ──"
$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$CACHE_DIR/vits_run8_train" \
  --val-cache   "$CACHE_DIR/vits_run8_val" \
  --work-dir    "$WEIGHTS_DIR/run8_vits" \
  --epochs 60 --batch-size 32 --pos-weight 20 --num-workers 0

echo ""
echo "================================================================"
echo " Run 8 Complete  $(date)"
echo " Weights:  $WEIGHTS_DIR/run8_convnext_s2/best.pt"
echo "           $WEIGHTS_DIR/run8_vits/best.pt"
echo " Eval with CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=400"
echo "================================================================"
