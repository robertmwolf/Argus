#!/usr/bin/env bash
# Run 12 pipeline — 33/33/33 short/medium/long balance, 518px model input, ViT-S + ConvNeXt-S.
#
# Key changes from Run 11:
#   - Model input: 518px (DINOv2 native resolution, no positional embedding interpolation)
#   - Training tiles: 1800px Atwood tiles → 518px (long streaks appear as medium at model input)
#   - Frigate 110px tiles → 518px (short streaks 38-282px apparent)
#   - Synthetic: short (70-260px drawn on 1800px tiles → 20-75px apparent) and
#                medium (278-694px drawn on 1800px tiles → 80-200px apparent)
#   - Both ViT-S and ConvNeXt-S trained and evaluated
#
# Apparent length formula: apparent_px = drawn_px * (518 / tile_size)
#   1800px tile: scale = 0.288  → short: 70–260px drawn, medium: 278–694px drawn
#   110px tile:  scale = 4.709  → Frigate 8-17px → 38-80px short, 17-42px → 80-200px medium
#
# Dataset targets (positive tiles):
#   Short  (~3000): Frigate 110px real + synthetic short on 1800px neg tiles
#   Medium (~3000): Atwood 1800px real (278–694px native) + synthetic medium
#   Long   (~3000): Atwood 1800px real (>694px native), subsampled
#
# Usage:
#   bash scripts/run12_pipeline.sh 2>&1 | tee /tmp/run12_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
EXT_CACHE_DIR=/Volumes/External/TrainingData/heatmap_cache
WEIGHTS_DIR=$REPO/weights

LOCAL_NPY_DIR=/tmp/argus_run12_npy
LOCAL_CACHE_DIR=/tmp/argus_run12_cache

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 12 Pipeline  $(date)"
echo " 33/33/33 | 518px model input | ViT-S + ConvNeXt-S"
echo "================================================================"

# ── Step 1: Restore frigate_streaks.json from git ─────────────────────────────
echo ""
echo "── Step 1: Restore frigate_streaks.json ──"
git show 25a80a5:data/annotations/frigate_streaks.json \
  > "$ANN_DIR/frigate_streaks.json"
echo "Restored: $ANN_DIR/frigate_streaks.json"

# ── Step 2: Tile Atwood at 1800px ─────────────────────────────────────────────
echo ""
echo "── Step 2: Tile Atwood at 1800px (train) ──"
$PYTHON scripts/build_tiled_brentimages_json.py \
  --src     "$ANN_DIR/all_train_run5.json" \
  --native-tile-size 1800 \
  --overlap 0.5 \
  --neg-tiles-per-image 2 \
  --hard-neg-per-pos 0

ATWOOD1800_JSON="$ANN_DIR/all_train_run5_tiled_ts1800.json"

# ── Step 3: Tile val_atwood at 1800px ─────────────────────────────────────────
echo ""
echo "── Step 3: Tile val_atwood at 1800px ──"
$PYTHON scripts/build_tiled_brentimages_json.py \
  --src     "$ANN_DIR/val_atwood.json" \
  --native-tile-size 1800 \
  --overlap 0.5 \
  --neg-tiles-per-image 1 \
  --hard-neg-per-pos 0

VAL1800_JSON="$ANN_DIR/val_atwood_tiled_ts1800.json"  # produced by build_tiled_brentimages_json.py

# ── Step 4: Tile Frigate at 110px ─────────────────────────────────────────────
echo ""
echo "── Step 4: Tile Frigate at 110px ──"
$PYTHON scripts/build_tiled_frigate_json.py \
  --native-tile-size 110 \
  --overlap 0.0

FRIGATE110_JSON="$ANN_DIR/frigate_tiled_train_ts110.json"

# ── Step 5: Convert all tiles to NPY ──────────────────────────────────────────
echo ""
echo "── Step 5: Convert tiles to NPY ──"
mkdir -p "$LOCAL_NPY_DIR/atwood1800" "$LOCAL_NPY_DIR/frigate110" "$LOCAL_NPY_DIR/val1800"

$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$ATWOOD1800_JSON" \
  --output-dir  "$LOCAL_NPY_DIR/atwood1800" \
  --output-ann  "$ANN_DIR/all_train_run12_atwood1800_npy.json"

$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$VAL1800_JSON" \
  --output-dir  "$LOCAL_NPY_DIR/val1800" \
  --output-ann  "$ANN_DIR/val_run12_1800_npy.json"

$PYTHON scripts/convert_tiles_to_npy.py \
  --annotations "$FRIGATE110_JSON" \
  --output-dir  "$LOCAL_NPY_DIR/frigate110" \
  --output-ann  "$ANN_DIR/frigate_run12_110_npy.json"

# ── Step 6: Generate synthetic short + medium on 1800px neg tiles ─────────────
# Apparent scale factor: 518 / 1800 = 0.2878
# Short target:  20–75px apparent → 70–260px drawn in 1800px tile space
# Medium target: 80–200px apparent → 278–694px drawn in 1800px tile space
echo ""
echo "── Step 6a: Synthetic short streaks (70–260px drawn → 20–75px apparent) ──"
mkdir -p "$LOCAL_NPY_DIR/synth_short" "$LOCAL_NPY_DIR/synth_medium"

$PYTHON scripts/generate_synthetic_streaks.py \
  --neg-annotations "$ANN_DIR/all_train_run12_atwood1800_npy.json" \
  --output-dir      "$LOCAL_NPY_DIR/synth_short" \
  --output-ann      "$ANN_DIR/synth_run12_short_npy.json" \
  --n-per-neg 1 \
  --length-range 70 260 \
  --snr-min 2.0 --snr-max 10.0 \
  --multi-streak-prob 0.10 \
  --seed 42

echo ""
echo "── Step 6b: Synthetic medium streaks (278–694px drawn → 80–200px apparent) ──"
$PYTHON scripts/generate_synthetic_streaks.py \
  --neg-annotations "$ANN_DIR/all_train_run12_atwood1800_npy.json" \
  --output-dir      "$LOCAL_NPY_DIR/synth_medium" \
  --output-ann      "$ANN_DIR/synth_run12_medium_npy.json" \
  --n-per-neg 1 \
  --length-range 278 694 \
  --snr-min 2.0 --snr-max 10.0 \
  --multi-streak-prob 0.10 \
  --seed 43

# ── Step 7: Merge with 33/33/33 balance ───────────────────────────────────────
echo ""
echo "── Step 7: Merge datasets → 33/33/33 balance ──"
$PYTHON - <<'PYEOF'
import json, random, math
from pathlib import Path

random.seed(42)
ANN_DIR = Path("/Volumes/External/TrainingData/annotations")
MODEL_INPUT = 518
TARGET_PER_BAND = 3000

# Scale factors: apparent_length = obb_length * (MODEL_INPUT / tile_size)
SCALE_1800 = MODEL_INPUT / 1800   # 0.2878
SCALE_110  = MODEL_INPUT / 110    # 4.709

def apparent(obb_len, scale):
    return obb_len * scale

def band(app_len):
    if app_len < 80:  return "short"
    if app_len < 200: return "medium"
    return "long"

def load(path):
    with open(path) as f: return json.load(f)

atwood   = load(ANN_DIR / "all_train_run12_atwood1800_npy.json")
frigate  = load(ANN_DIR / "frigate_run12_110_npy.json")
s_short  = load(ANN_DIR / "synth_run12_short_npy.json")
s_medium = load(ANN_DIR / "synth_run12_medium_npy.json")

ann_by_img = {}
for a in atwood["annotations"] + frigate["annotations"]:
    ann_by_img.setdefault(a["image_id"], []).append(a)

# Classify Atwood 1800px positive tiles by apparent band
atwood_by_band = {"short": [], "medium": [], "long": []}
atwood_neg = []
atwood_img = {i["id"]: i for i in atwood["images"]}
for img in atwood["images"]:
    anns = ann_by_img.get(img["id"], [])
    if not anns:
        atwood_neg.append(img)
        continue
    # Use max OBB length across annotations for this tile
    lengths = []
    for a in anns:
        obb = a.get("obb") or {}
        l = obb.get("w") or obb.get("length_px") or a.get("attributes", {}).get("length_px")
        if l: lengths.append(l)
    if not lengths:
        atwood_by_band["long"].append(img)
        continue
    app = apparent(max(lengths), SCALE_1800)
    atwood_by_band[band(app)].append(img)

# Classify Frigate 110px positive tiles
frigate_by_band = {"short": [], "medium": [], "long": []}
frigate_neg = []
frigate_ann_by_img = {}
for a in frigate["annotations"]:
    frigate_ann_by_img.setdefault(a["image_id"], []).append(a)
for img in frigate["images"]:
    anns = frigate_ann_by_img.get(img["id"], [])
    if not anns:
        frigate_neg.append(img)
        continue
    lengths = []
    for a in anns:
        obb = a.get("obb") or {}
        l = obb.get("w") or obb.get("length_px") or a.get("attributes", {}).get("length_px")
        if l: lengths.append(l)
    if not lengths:
        frigate_by_band["short"].append(img)
        continue
    app = apparent(max(lengths), SCALE_110)
    frigate_by_band[band(app)].append(img)

print("Atwood 1800px tile distribution:")
for b, imgs in atwood_by_band.items():
    print(f"  {b:8s}: {len(imgs)}")
print(f"  negative: {len(atwood_neg)}")
print("Frigate 110px tile distribution:")
for b, imgs in frigate_by_band.items():
    print(f"  {b:8s}: {len(imgs)}")
print(f"  negative: {len(frigate_neg)}")

# Synthetic tiles (all are positive by construction)
synth_short_imgs  = s_short["images"]
synth_medium_imgs = s_medium["images"]
print(f"Synthetic short:  {len(synth_short_imgs)}")
print(f"Synthetic medium: {len(synth_medium_imgs)}")

# Pool by band: real first (prefer real), then synthetic top-up
pool_short  = atwood_by_band["short"] + frigate_by_band["short"] + frigate_by_band["medium"]
pool_medium = atwood_by_band["medium"]
pool_long   = atwood_by_band["long"] + frigate_by_band["long"]

# Shuffle all pools
random.shuffle(pool_short);  random.shuffle(pool_medium);  random.shuffle(pool_long)
random.shuffle(synth_short_imgs);  random.shuffle(synth_medium_imgs)

# Sample real + synthetic to hit TARGET_PER_BAND each
def fill(real_pool, synth_pool, target, label):
    selected_real = real_pool[:target]
    need = max(0, target - len(selected_real))
    selected_synth = synth_pool[:need]
    total = len(selected_real) + len(selected_synth)
    print(f"  {label}: {len(selected_real)} real + {len(selected_synth)} synthetic = {total}")
    return selected_real, selected_synth

print("\nSelecting tiles per band:")
short_real,  short_synth  = fill(pool_short,  synth_short_imgs,  TARGET_PER_BAND, "short")
medium_real, medium_synth = fill(pool_medium, synth_medium_imgs, TARGET_PER_BAND, "medium")
long_real,   long_synth   = fill(pool_long,   [],                TARGET_PER_BAND, "long")

# Negative tiles: take all from atwood + frigate (up to ~4500)
all_neg = atwood_neg + frigate_neg
random.shuffle(all_neg)
neg_selected = all_neg[:4500]
print(f"  negatives: {len(neg_selected)}")

# Build merged COCO JSON
# Need to re-map IDs to avoid collisions across sources
# Gather all source JSONs and their annotation maps
sources = [
    (atwood,   "atwood",  SCALE_1800),
    (frigate,  "frigate", SCALE_110),
    (s_short,  "synth_short",  None),
    (s_medium, "synth_medium", None),
]

# Build lookup: (source_name, orig_img_id) → annotations
all_anns_by_src = {}
for (src, name, _) in sources:
    ann_map = {}
    for a in src.get("annotations", []):
        ann_map.setdefault(a["image_id"], []).append(a)
    all_anns_by_src[name] = ann_map

# Collect selected image objects tagged with source
selected_pos = (
    [(img, "atwood")  for img in short_real  if img in atwood_by_band["short"]] +
    [(img, "frigate") for img in short_real  if img in frigate_by_band["short"] + frigate_by_band["medium"]] +
    [(img, "synth_short")  for img in short_synth] +
    [(img, "atwood")  for img in medium_real] +
    [(img, "synth_medium") for img in medium_synth] +
    [(img, "atwood")  for img in long_real] +
    [(img, "atwood")  for img in neg_selected if img in atwood_by_band.get("short",[]) + atwood_neg] +
    [(img, "frigate") for img in neg_selected if img in frigate_neg]
)
# Simpler: rebuild with explicit source tracking
selected_with_src = []
atwood_ids = {i["id"] for i in atwood["images"]}
frigate_ids = {i["id"] for i in frigate["images"]}
ss_ids = {i["id"] for i in s_short["images"]}
sm_ids = {i["id"] for i in s_medium["images"]}

def src_of(img):
    iid = img["id"]
    if iid in atwood_ids:  return "atwood"
    if iid in frigate_ids: return "frigate"
    if iid in ss_ids:      return "synth_short"
    if iid in sm_ids:      return "synth_medium"
    return "atwood"

all_selected_imgs = (
    list(short_real) + list(short_synth) +
    list(medium_real) + list(medium_synth) +
    list(long_real) +
    list(neg_selected)
)

# Deduplicate by id+source
seen = set()
deduped = []
for img in all_selected_imgs:
    key = (img["id"], src_of(img))
    if key not in seen:
        seen.add(key)
        deduped.append((img, src_of(img)))

print(f"\nTotal tiles before dedup: {len(all_selected_imgs)}")
print(f"Total tiles after dedup:  {len(deduped)}")

merged_images = []
merged_annotations = []
next_img_id = 1
next_ann_id = 1

for orig_img, src_name in deduped:
    new_img = dict(orig_img)
    old_id = orig_img["id"]
    new_img["id"] = next_img_id
    new_img["source"] = src_name
    merged_images.append(new_img)

    for a in all_anns_by_src[src_name].get(old_id, []):
        new_ann = dict(a)
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = next_img_id
        merged_annotations.append(new_ann)
        next_ann_id += 1

    next_img_id += 1

ann_img_ids = {a["image_id"] for a in merged_annotations}
n_pos = sum(1 for i in merged_images if i["id"] in ann_img_ids)
n_neg = len(merged_images) - n_pos
print(f"\nFinal dataset: {len(merged_images)} tiles | pos: {n_pos} | neg: {n_neg} ({100*n_neg/len(merged_images):.1f}% neg)")

out = {
    "info": {"description": "ARGUS Run 12 — 33/33/33 short/medium/long, 1800px tiles, 518px model input", "version": "1.0"},
    "licenses": [],
    "categories": atwood["categories"],
    "images": merged_images,
    "annotations": merged_annotations,
}
out_path = ANN_DIR / "all_train_run12_npy.json"
with open(out_path, "w") as f:
    json.dump(out, f)
print(f"Written: {out_path}")
PYEOF

# ── Step 8: Cache ViT-S features (518px) ──────────────────────────────────────
echo ""
echo "── Step 8a: Cache ViT-S train features (518px) ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run12_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/vits_run12_train" \
  --backbone vit --model-size small \
  --image-size 518 --num-workers 0

echo ""
echo "── Step 8b: Cache ViT-S val features (518px) ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/val_run12_1800_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/vits_run12_val" \
  --backbone vit --model-size small \
  --image-size 518 --num-workers 0

echo ""
echo "── Step 8c: Cache ConvNeXt-S train features (518px) ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/all_train_run12_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/convnext_run12_train" \
  --backbone convnext --model-size small \
  --image-size 518 --num-workers 0

echo ""
echo "── Step 8d: Cache ConvNeXt-S val features (518px) ──"
$PYTHON scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ANN_DIR/val_run12_1800_npy.json" \
  --output-dir  "$EXT_CACHE_DIR/convnext_run12_val" \
  --backbone convnext --model-size small \
  --image-size 518 --num-workers 0

# ── Step 9: Delete local NPY (cache lives on external) ────────────────────────
echo ""
echo "── Step 9: Delete local NPY tiles ──"
rm -rf "$LOCAL_NPY_DIR"
echo "Local NPY deleted."

# ── Step 10: Train ViT-S ──────────────────────────────────────────────────────
echo ""
echo "── Step 10: Copy ViT-S cache to local SSD + train ──"
mkdir -p "$LOCAL_CACHE_DIR"
rsync -a "$EXT_CACHE_DIR/vits_run12_train/" "$LOCAL_CACHE_DIR/vits_run12_train/"
rsync -a "$EXT_CACHE_DIR/vits_run12_val/"   "$LOCAL_CACHE_DIR/vits_run12_val/"

$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/vits_run12_train" \
  --val-cache   "$LOCAL_CACHE_DIR/vits_run12_val" \
  --work-dir    "$WEIGHTS_DIR/run12_vits" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

rm -rf "$LOCAL_CACHE_DIR/vits_run12_train" "$LOCAL_CACHE_DIR/vits_run12_val"

# ── Step 11: Train ConvNeXt-S ─────────────────────────────────────────────────
echo ""
echo "── Step 11: Copy ConvNeXt-S cache to local SSD + train ──"
rsync -a "$EXT_CACHE_DIR/convnext_run12_train/" "$LOCAL_CACHE_DIR/convnext_run12_train/"
rsync -a "$EXT_CACHE_DIR/convnext_run12_val/"   "$LOCAL_CACHE_DIR/convnext_run12_val/"

$PYTHON training/train_dinov3_heatmap_cached.py \
  --train-cache "$LOCAL_CACHE_DIR/convnext_run12_train" \
  --val-cache   "$LOCAL_CACHE_DIR/convnext_run12_val" \
  --work-dir    "$WEIGHTS_DIR/run12_convnext_s" \
  --epochs 40 --batch-size 32 --pos-weight 20 --num-workers 0 \
  --lr-scheduler cosine

rm -rf "$LOCAL_CACHE_DIR/convnext_run12_train" "$LOCAL_CACHE_DIR/convnext_run12_val"
rmdir "$LOCAL_CACHE_DIR" 2>/dev/null || true

# ── Step 12: Evaluate both models ─────────────────────────────────────────────
echo ""
# Eval at t=0.05 so the post-hoc sweep has a full picture of the model's score
# distribution.  Using t=0.3 as the base would hard-cap recall: any real streak
# scored 0.05–0.29 would be invisible to every sweep point.  t=0.05 costs a
# slightly larger predictions.json but gives an honest precision-recall curve.
echo "── Step 12a: Evaluate ViT-S (t=0.05, tiled 1800px → 518px, stitch) ──"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
VITS_HEATMAP_NATIVE_TILE_SIZE=1800 \
$PYTHON scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run12_vits/best.pt" \
  --output      results/run12_vits/t0.05/metrics.json \
  --tiled --threshold 0.05 --stitch

$PYTHON scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run12_vits/t0.05/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run12_vits/threshold_sweep \
  --thresholds 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "── Step 12b: Evaluate ConvNeXt-S (t=0.05, tiled 1800px → 518px, stitch) ──"
PYTORCH_ENABLE_MPS_FALLBACK=1 \
CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=1800 \
$PYTHON scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test_atwood.json \
  --checkpoint  "$WEIGHTS_DIR/run12_convnext_s/best.pt" \
  --backbone convnext \
  --output      results/run12_convnext_s/t0.05/metrics.json \
  --tiled --threshold 0.05 --stitch

$PYTHON scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run12_convnext_s/t0.05/predictions.json \
  --annotations data/annotations/test_atwood.json \
  --output-dir  results/run12_convnext_s/threshold_sweep \
  --thresholds 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
  --stitch || true

echo ""
echo "================================================================"
echo " Run 12 Complete  $(date)"
echo " ViT-S:      results/run12_vits/threshold_sweep/threshold_sweep.json"
echo " ConvNeXt-S: results/run12_convnext_s/threshold_sweep/threshold_sweep.json"
echo "================================================================"
