#!/usr/bin/env bash
# Run a threshold × stitch grid eval on Run 8 heatmap checkpoints.
#
# Usage:
#   bash scripts/run_threshold_sweep.sh            # full sweep (28 runs, ~5 hrs)
#   bash scripts/run_threshold_sweep.sh 5          # smoke test on 5 images per run
#
# Results land in results/run8_sweep/<backbone>_t<thresh>_<stitch>/metrics.json
# Summarise afterwards with:
#   python scripts/print_sweep_results.py results/run8_sweep
set -euo pipefail

MAX_SAMPLES_ARG=""
if [[ $# -ge 1 ]]; then
    MAX_SAMPLES_ARG="--max-samples $1"
fi

PYTHON="${PYTHON:-/Users/robert/miniconda3/envs/satid/bin/python}"
THRESHOLDS="0.50 0.60 0.70 0.80 0.85 0.90 0.95"
CONVNEXT_CKPT="weights/run8_convnext_s2/best.pt"
VITS_CKPT="weights/run8_vits/best.pt"
ANN="data/annotations/test_atwood.json"
OUT_BASE="results/run8_sweep"

total=0; done_=0
for _ in $THRESHOLDS; do total=$((total + 4)); done
echo "Starting sweep: $total eval runs  $(date)"
echo ""

for THRESH in $THRESHOLDS; do
    for STITCH_FLAG in "" "--stitch"; do
        STITCH_TAG=$([ -n "$STITCH_FLAG" ] && echo "stitch" || echo "nostitch")

        # ── ConvNeXt-S ──────────────────────────────────────────────────────────
        OUT_DIR="${OUT_BASE}/convnext_t${THRESH}_${STITCH_TAG}"
        echo "[$((done_+1))/$total] convnext  thresh=${THRESH}  stitch=${STITCH_TAG}"
        CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=400 CONVNEXT_HEATMAP_TILE_OVERLAP=0.5 \
        PYTORCH_ENABLE_MPS_FALLBACK=1 \
        "$PYTHON" scripts/evaluate_dinov3_heatmap.py \
            --annotations "$ANN" \
            --checkpoint  "$CONVNEXT_CKPT" \
            --output      "${OUT_DIR}/metrics.json" \
            --tiled --threshold "$THRESH" $STITCH_FLAG $MAX_SAMPLES_ARG
        done_=$((done_+1))

        # ── ViT-S ────────────────────────────────────────────────────────────────
        OUT_DIR="${OUT_BASE}/vits_t${THRESH}_${STITCH_TAG}"
        echo "[$((done_+1))/$total] vits      thresh=${THRESH}  stitch=${STITCH_TAG}"
        PYTORCH_ENABLE_MPS_FALLBACK=1 \
        "$PYTHON" scripts/evaluate_dinov3_heatmap.py \
            --annotations "$ANN" \
            --checkpoint  "$VITS_CKPT" \
            --output      "${OUT_DIR}/metrics.json" \
            --tiled --threshold "$THRESH" $STITCH_FLAG $MAX_SAMPLES_ARG
        done_=$((done_+1))

        echo ""
    done
done

echo "================================================================"
echo " Sweep complete  $(date)"
echo " Results: $OUT_BASE"
echo " Summary: $PYTHON scripts/print_sweep_results.py $OUT_BASE"
echo "================================================================"
