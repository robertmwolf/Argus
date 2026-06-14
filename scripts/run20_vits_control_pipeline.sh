#!/usr/bin/env bash
# Run 20 — CONTROLLED backbone experiment: ViT-S vs ViT-B on the SAME data.
#
# PURPOSE (auditability): Runs 17/18/19 trained ViT-B on train_run18.json and
# failed (val_dice ~0.12, eval recall 0.000). Run 15 trained ViT-S on DIFFERENT
# data (run10 NPY) and succeeded (val_dice 0.77, recall 0.93). That left the
# question "is ViT-B's failure the BACKBONE or the train_run18 DATA?" confounded.
#
# This experiment breaks the confound by holding EVERYTHING constant except the
# backbone:
#   * SAME training data ........ data/annotations/train_run18.json  (+ val_run18)
#   * SAME cache geometry ....... img 518 / native-tile 400 / overlap 0 / zscore
#                                 / neg-tiles-per-image 4   (identical to run19)
#   * SAME training recipe "R" .. epochs 40, lr 1e-3, batch 32, pos_weight 20,
#                                 geometry_weight 0.25, hidden 256, cosine, no warmup
#   * ONLY difference ........... ViT-S/16 (384-d) vs ViT-B/16 (768-d) backbone
#
# READOUT:
#   * ViT-S also lands ~0.12  -> the DATA (train_run18) is the problem; ViT-B exonerated.
#   * ViT-S reaches ~0.7      -> the failure is BACKBONE-specific to ViT-B.
#
# The ViT-B arm retrains from the EXISTING run19 cache (no re-cache) under the
# identical Recipe R, so the two arms are directly comparable. Only the ViT-S
# feature cache is newly computed here.
#
# All inputs/outputs are recorded to results/run20_control/provenance.json for
# reproducibility. Caches are KEPT (not deleted) so the run is re-runnable.
#
# Usage:
#   bash scripts/run20_vits_control_pipeline.sh 2>&1 | tee /tmp/run20_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO=/Users/robert/Argus
WEIGHTS=$REPO/weights
VITS_WEIGHTS=$WEIGHTS/dinov3_vits16_lvd1689m.pth
VITB_CACHE=~/argus_run19_cache                 # existing ViT-B cache (run19), REUSED
VITS_CACHE=~/argus_run20_vits_cache            # new ViT-S cache (this run)
TRAIN_ANN=$REPO/data/annotations/train_run18.json
VAL_ANN=$REPO/data/annotations/val_run18.json
BAL_ANN=$REPO/data/annotations/val_balanced_v1.json
OUT=$REPO/results/run20_control
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"
mkdir -p "$OUT"

# Pinned Recipe R (identical for BOTH arms).
EPOCHS=40; LR=1e-3; BATCH=32; POSW=20; GEOMW=0.25; HIDDEN=256; SCHED=cosine

echo "=== Run 20 controlled backbone experiment | $(date) ==="
echo "    data=train_run18.json  recipe: epochs=$EPOCHS lr=$LR batch=$BATCH posw=$POSW hidden=$HIDDEN sched=$SCHED"
[ -f "$VITS_WEIGHTS" ] || { echo "ERROR missing $VITS_WEIGHTS"; exit 1; }
[ -f "$VITB_CACHE/train/manifest.json" ] || { echo "ERROR missing ViT-B run19 cache at $VITB_CACHE"; exit 1; }

# ── Step 1: Cache ViT-S features on the SAME data/geometry as ViT-B (run19) ────
if [ -f "$VITS_CACHE/train/manifest.json" ] && [ -f "$VITS_CACHE/val/manifest.json" ]; then
  echo "── ViT-S cache present at $VITS_CACHE — skipping ── $(date)"
else
  for SPLIT in train val; do
    ANN=$TRAIN_ANN; [ "$SPLIT" = val ] && ANN=$VAL_ANN
    echo "── Caching ViT-S $SPLIT features ── $(date)"
    $PYTHON scripts/cache_dinov3_heatmap_features.py \
      --annotations "$ANN" --output-dir "$VITS_CACHE/$SPLIT" \
      --backbone vit --model-size small --weights "$VITS_WEIGHTS" \
      --image-size 518 --num-workers 0 --native-tile-size 400 \
      --tile-overlap 0.0 --norm-mode zscore --neg-tiles-per-image 4
  done
fi

# ── Step 2: Train BOTH heads under identical Recipe R ─────────────────────────
train_arm () {  # $1=cache_dir  $2=workdir_name
  echo "── Training $2 (Recipe R) ── $(date)"
  $PYTHON training/train_dinov3_heatmap_cached.py \
    --train-cache "$1/train" --val-cache "$1/val" \
    --work-dir "$WEIGHTS/$2" \
    --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH" \
    --pos-weight "$POSW" --geometry-weight "$GEOMW" --hidden-channels "$HIDDEN" \
    --lr-scheduler "$SCHED" --num-workers 0
}
train_arm "$VITS_CACHE" run20_vits_control      # ViT-S on train_run18
train_arm "$VITB_CACHE" run20_vitb_control      # ViT-B on train_run18 (same recipe)

# ── Step 3: Eval BOTH on val_balanced_v1 (same gate, same pipeline) ───────────
eval_arm () {  # $1=workdir_name  $2=norm
  echo "── Eval $1 on val_balanced_v1 ── $(date)"
  local BC=~/argus_run20_${1}_balcache
  $PYTHON scripts/cache_heatmap_maps.py --annotations "$BAL_ANN" \
    --checkpoint "$WEIGHTS/$1/best.pt" --output-dir "$BC" --norm-mode "$2"
  $PYTHON scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
    --heatmap-cache "$BC" --annotations "$BAL_ANN" --norm-mode "$2" \
    --threshold 0.05 --threshold-sweep 0.5 0.6 0.7 0.8 --peak-floor 0.85 \
    --output "$OUT/$1/pf85/metrics_placeholder.json"
  $PYTHON scripts/summarize_balanced_eval.py "$OUT/$1/pf85"
  rm -rf "$BC"
}
eval_arm run20_vits_control zscore
eval_arm run20_vitb_control zscore

# ── Step 4: Record provenance (model -> data -> recipe -> result paths) ───────
$PYTHON - "$OUT/provenance.json" <<'PY'
import json, sys
prov = {
  "experiment": "run20_control_backbone",
  "purpose": "Isolate backbone (ViT-S vs ViT-B) by holding data+recipe constant.",
  "shared": {
    "train_data": "data/annotations/train_run18.json",
    "val_data":   "data/annotations/val_run18.json",
    "eval_data":  "data/annotations/val_balanced_v1.json",
    "cache_geometry": {"image_size":518,"native_tile_size":400,"tile_overlap":0.0,
                        "norm_mode":"zscore","neg_tiles_per_image":4},
    "recipe_R": {"epochs":40,"lr":1e-3,"batch_size":32,"pos_weight":20,
                  "geometry_weight":0.25,"hidden_channels":256,"lr_scheduler":"cosine",
                  "warmup_epochs":0},
  },
  "arms": {
    "vit_s": {"backbone":"dinov3_vits16_lvd1689m.pth","dim":384,
               "cache":"~/argus_run20_vits_cache","weights":"weights/run20_vits_control",
               "history":"weights/run20_vits_control/history.json",
               "eval":"results/run20_control/run20_vits_control/pf85"},
    "vit_b": {"backbone":"dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth","dim":768,
               "cache":"~/argus_run19_cache (reused from run19)",
               "weights":"weights/run20_vitb_control",
               "history":"weights/run20_vitb_control/history.json",
               "eval":"results/run20_control/run20_vitb_control/pf85"},
  },
}
open(sys.argv[1],"w").write(json.dumps(prov, indent=2))
print("wrote", sys.argv[1])
PY

echo "=== Run 20 complete | $(date) ==="
echo "    ViT-S history: $WEIGHTS/run20_vits_control/history.json"
echo "    ViT-B history: $WEIGHTS/run20_vitb_control/history.json"
echo "    Provenance:    $OUT/provenance.json"
echo "    Caches kept: $VITS_CACHE (ViT-S), $VITB_CACHE (ViT-B)"
