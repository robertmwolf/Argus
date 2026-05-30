#!/bin/bash
# Run 4 full benchmark suite — MMDet OBB + Centerline heatmap
# Launched by Claude Code 2026-05-29
set -euo pipefail
cd /Users/robert/Argus
PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
CKPT_MMDET=weights/run4_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth
CFG_MMDET=models/dino/streak_dinov3_vits_400px_run3.py
CKPT_CL=weights/run_dinov3_vits_orientation_centerline_1024/best.pt

echo "========================================================"
echo "[$(date)] Run 4 evaluation suite starting"
echo "  MMDet  checkpoint: $CKPT_MMDET"
echo "  CL     checkpoint: $CKPT_CL"
echo "========================================================"

# ── 1. MMDet: geometry-stratified test set (primary quality gate) ──────────
echo ""
echo "[$(date)] [1/6] MMDet — test_atwood (geometry-stratified, 133 imgs)"
PYTORCH_ENABLE_MPS_FALLBACK=1 ARGUS_ENABLE_PLATE_SOLVE=false \
  "$PYTHON" scripts/zero_shot_eval.py \
    --annotation data/annotations/test_atwood.json \
    --scope run4_mmdet_test_atwood \
    --label "Run 4 MMDet — Atwood geo-stratified test (133 imgs)" \
    --checkpoint "$CKPT_MMDET" \
    --config "$CFG_MMDET"
echo "[$(date)] [1/6] DONE"

# ── 2. MMDet: zero-shot holdout night 2026-05-27 ──────────────────────────
echo ""
echo "[$(date)] [2/6] MMDet — atwood_20260527 zero-shot (507 pos + 25 neg)"
PYTORCH_ENABLE_MPS_FALLBACK=1 ARGUS_ENABLE_PLATE_SOLVE=false \
  "$PYTHON" scripts/zero_shot_eval.py \
    --annotation data/annotations/atwood_20260527.json \
    --negatives data/annotations/atwood_20260527_negatives.json \
    --scope run4_mmdet_atwood_20260527 \
    --label "Run 4 MMDet — Atwood 2026-05-27 zero-shot holdout (507 imgs)" \
    --checkpoint "$CKPT_MMDET" \
    --config "$CFG_MMDET"
echo "[$(date)] [2/6] DONE"

# ── 3. MMDet: zero-shot holdout night 2026-05-28 ──────────────────────────
echo ""
echo "[$(date)] [3/6] MMDet — atwood_20260528 zero-shot (175 pos + 18 neg)"
PYTORCH_ENABLE_MPS_FALLBACK=1 ARGUS_ENABLE_PLATE_SOLVE=false \
  "$PYTHON" scripts/zero_shot_eval.py \
    --annotation data/annotations/atwood_20260528.json \
    --negatives data/annotations/atwood_20260528_negatives.json \
    --scope run4_mmdet_atwood_20260528 \
    --label "Run 4 MMDet — Atwood 2026-05-28 zero-shot holdout (175 imgs)" \
    --checkpoint "$CKPT_MMDET" \
    --config "$CFG_MMDET"
echo "[$(date)] [3/6] DONE"

# ── 4. MMDet: SatStreaks standard benchmark (secondary) ───────────────────
echo ""
echo "[$(date)] [4/6] MMDet — SatStreaks standard test (308 imgs, secondary)"
PYTORCH_ENABLE_MPS_FALLBACK=1 ARGUS_ENABLE_PLATE_SOLVE=false \
  "$PYTHON" scripts/evaluate_comprehensive.py \
    --checkpoint "$CKPT_MMDET" \
    --config "$CFG_MMDET" \
    --sets test_standard
echo "[$(date)] [4/6] DONE"

# ── 5. Centerline: geometry-stratified test set ───────────────────────────
echo ""
echo "[$(date)] [5/6] Centerline — test_atwood (133 imgs)"
PYTORCH_ENABLE_MPS_FALLBACK=1 ARGUS_ENABLE_PLATE_SOLVE=false \
  "$PYTHON" scripts/evaluate_dinov3_orientation_centerline.py \
    --annotations /Volumes/External/TrainingData/annotations/test_atwood.json \
    --checkpoint "$CKPT_CL" \
    --output results/run4_centerline_test_atwood/metrics.json \
    --preserve-image-bit-depth
echo "[$(date)] [5/6] DONE"

# ── 6. Centerline: zero-shot holdout 2026-05-27 ───────────────────────────
echo ""
echo "[$(date)] [6/6] Centerline — atwood_20260527 zero-shot (507 imgs)"
PYTORCH_ENABLE_MPS_FALLBACK=1 ARGUS_ENABLE_PLATE_SOLVE=false \
  "$PYTHON" scripts/evaluate_dinov3_orientation_centerline.py \
    --annotations /Volumes/External/TrainingData/annotations/atwood_20260527.json \
    --checkpoint "$CKPT_CL" \
    --output results/run4_centerline_zeroshot_20260527/metrics.json \
    --preserve-image-bit-depth
echo "[$(date)] [6/6] DONE"

echo ""
echo "========================================================"
echo "[$(date)] All Run 4 evaluations complete."
echo "========================================================"
