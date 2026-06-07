#!/usr/bin/env bash
# Run 12 resume — restarts from Step 5 (NPY conversion) after internal-disk-full crash.
#
# Preconditions:
#   - /tmp/argus_run12_npy is a symlink → /Volumes/External/TrainingData/argus_run12_npy
#   - Internal drive has ~119 GB free (backup deleted)
#   - External drive has ~310 GB free
#   - Tiled annotation JSONs (Steps 2-4) still exist on external
#
# Usage:
#   bash scripts/run12_resume.sh 2>&1 | tee /tmp/run12_resume_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
ANN_DIR=/Volumes/External/TrainingData/annotations
EXT_CACHE_DIR=/Volumes/External/TrainingData/heatmap_cache
WEIGHTS_DIR=$REPO/weights

LOCAL_NPY_DIR=/tmp/argus_run12_npy   # symlink → /Volumes/External/TrainingData/argus_run12_npy
LOCAL_CACHE_DIR=/tmp/argus_run12_cache

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "================================================================"
echo " Run 12 Resume  $(date)"
echo " Restarting from Step 5 (NPY via external symlink)"
echo "================================================================"

# Verify symlink
if [ ! -L "$LOCAL_NPY_DIR" ]; then
  echo "ERROR: $LOCAL_NPY_DIR is not a symlink. Run:"
  echo "  ln -s /Volumes/External/TrainingData/argus_run12_npy /tmp/argus_run12_npy"
  exit 1
fi
echo "NPY symlink OK: $LOCAL_NPY_DIR → $(readlink $LOCAL_NPY_DIR)"
df -h / /Volumes/External

ATWOOD1800_JSON="$ANN_DIR/all_train_run5_tiled_ts1800.json"
VAL1800_JSON="$ANN_DIR/val_atwood_tiled_ts1800.json"
FRIGATE110_JSON="$ANN_DIR/frigate_tiled_train_ts110.json"

# ── Step 5: Convert tiles to NPY (output → external via symlink) ──────────────
echo ""
echo "── Step 5: Convert tiles to NPY (writing to external via symlink) ──"
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

# ── Step 6: Generate synthetic streaks (output → external via symlink) ─────────
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

atwood_by_band = {"short": [], "medium": [], "long": []}
atwood_neg = []
for img in atwood["images"]:
    anns = ann_by_img.get(img["id"], [])
    if not anns:
        atwood_neg.append(img)
        continue
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

synth_short_imgs  = s_short["images"]
synth_medium_imgs = s_medium["images"]
print(f"Synthetic short:  {len(synth_short_imgs)}")
print(f"Synthetic medium: {len(synth_medium_imgs)}")

pool_short  = atwood_by_band["short"] + frigate_by_band["short"] + frigate_by_band["medium"]
pool_medium = atwood_by_band["medium"]
pool_long   = atwood_by_band["long"] + frigate_by_band["long"]

random.shuffle(pool_short);  random.shuffle(pool_medium);  random.shuffle(pool_long)
random.shuffle(synth_short_imgs);  random.shuffle(synth_medium_imgs)

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

all_neg = atwood_neg + frigate_neg
random.shuffle(all_neg)
neg_selected = all_neg[:4500]
print(f"  negatives: {len(neg_selected)}")

sources = [
    (atwood,   "atwood",  SCALE_1800),
    (frigate,  "frigate", SCALE_110),
    (s_short,  "synth_short",  None),
    (s_medium, "synth_medium", None),
]

all_anns_by_src = {}
for (src, name, _) in sources:
    ann_map = {}
    for a in src.get("annotations", []):
        ann_map.setdefault(a["image_id"], []).append(a)
    all_anns_by_src[name] = ann_map

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

# ── Pre-Step-8: Stage selected NPY on internal SSD ───────────────────────────
# Annotation JSONs store the symlink path (/tmp/argus_run12_npy/...) not the
# resolved external path.  Swapping the symlink for a real directory that
# contains only the selected tiles makes all four caching jobs read from
# internal SSD at full speed without exceeding the 100 GB shared-machine budget.
echo ""
echo "── Pre-Step-8: Stage selected NPY on internal SSD (~90 GB expected) ──"

# Swap symlink → real dir.  NPY writes are done at this point.
rm -f "$LOCAL_NPY_DIR"
mkdir -p \
  "$LOCAL_NPY_DIR/atwood1800" \
  "$LOCAL_NPY_DIR/val1800" \
  "$LOCAL_NPY_DIR/frigate110" \
  "$LOCAL_NPY_DIR/synth_short" \
  "$LOCAL_NPY_DIR/synth_medium"

# Build the list of NPY files referenced by train + val annotation JSONs.
$PYTHON - <<'PYEOF'
import json, sys
from pathlib import Path

TRAIN_ANN = "/Volumes/External/TrainingData/annotations/all_train_run12_npy.json"
VAL_ANN   = "/Volumes/External/TrainingData/annotations/val_run12_1800_npy.json"
NPY_EXT   = Path("/Volumes/External/TrainingData/argus_run12_npy")
OUT_LIST  = "/tmp/run12_npy_files.txt"
SYMLINK_PREFIX = "/tmp/argus_run12_npy/"

files = set()
for ann_path in [TRAIN_ANN, VAL_ANN]:
    data = json.load(open(ann_path))
    for img in data["images"]:
        fn = img.get("file_name", "")
        if fn.startswith(SYMLINK_PREFIX):
            rel = fn[len(SYMLINK_PREFIX):]
            ext_path = NPY_EXT / rel
            if ext_path.exists():
                files.add(rel)
            else:
                print(f"WARNING: missing on external: {rel}", file=sys.stderr)

with open(OUT_LIST, "w") as f:
    for rel in sorted(files):
        f.write(rel + "\n")
print(f"Selected {len(files)} NPY files for staging")
PYEOF

# Rsync only selected files from external to internal.
echo "Rsyncing selected NPY to internal (reads: external, writes: internal)..."
rsync -a --files-from=/tmp/run12_npy_files.txt \
  /Volumes/External/TrainingData/argus_run12_npy/ \
  "$LOCAL_NPY_DIR/"

STAGED=$(du -sh "$LOCAL_NPY_DIR" | cut -f1)
STAGED_GB=$(du -s "$LOCAL_NPY_DIR" | awk '{printf "%.1f", $1/1048576}')
echo "Staged ${STAGED} on internal SSD."
df -h /

# Safety valve: if somehow over 100 GB, warn but continue (cacher will still work).
if (( $(du -s "$LOCAL_NPY_DIR" | cut -f1) > 104857600 )); then
  echo "WARNING: staged NPY exceeds 100 GB — continuing but may crowd other users."
fi

# ── Step 8: Cache features (reads NPY from internal SSD, writes .pt to external) ──
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

# ── Step 9: Clear staged NPY from internal SSD ────────────────────────────────
echo ""
echo "── Step 9: Clear staged NPY from internal SSD ──"
rm -rf "$LOCAL_NPY_DIR"
df -h /
echo "Internal SSD freed. External NPY retained at /Volumes/External/TrainingData/argus_run12_npy"

# ── Step 10: Train ViT-S ──────────────────────────────────────────────────────
echo ""
echo "── Step 10: Copy ViT-S cache to local SSD + train ──"
# Feature cache per backbone: ~30 GB (train + val).  Internal budget: 100 GB.
# Only one backbone cache on internal at a time; deleted before next backbone starts.
mkdir -p "$LOCAL_CACHE_DIR"
rsync -a "$EXT_CACHE_DIR/vits_run12_train/" "$LOCAL_CACHE_DIR/vits_run12_train/"
rsync -a "$EXT_CACHE_DIR/vits_run12_val/"   "$LOCAL_CACHE_DIR/vits_run12_val/"
echo "ViT-S cache staged: $(du -sh $LOCAL_CACHE_DIR | cut -f1)"

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
echo "ConvNeXt-S cache staged: $(du -sh $LOCAL_CACHE_DIR | cut -f1)"

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
