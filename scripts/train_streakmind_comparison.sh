#!/usr/bin/env bash
# train_streakmind_comparison.sh — Full StreakMind methodology-matched comparison.
#
# Trains four YOLO11n-OBB tracks on raw FITS data (GTImages + Frigate negatives)
# and evaluates each at both IoU=0.5 and IoU=0.8 (StreakMind's threshold).
#
# Tracks:
#   real_only            GTImages real annotations only
#   paper_long           GTImages + synthetic long streaks (StreakMind paper style)
#   adapted              GTImages + synthetic short/medium adapted distribution
#   gtimages_plus_frigate GTImages real + Frigate background diversity
#
# Prerequisite annotation files (auto-generated here if missing):
#   data/annotations/gtimages_train_real.json        (from augment_gtimages_synthetic.py)
#   data/annotations/gtimages_train_synth_*.json     (from augment_gtimages_synthetic.py)
#   data/annotations/gtimages_val.json               (from augment_gtimages_synthetic.py)
#   data/annotations/gtimages_test.json              (from augment_gtimages_synthetic.py)
#   data/annotations/frigate_negatives.json          (generated here)
#   data/annotations/gtimages_plus_frigate_train.json (generated here)
#
# Outputs:
#   results/streakmind_yolo/comparison.json
#   results/streakmind_yolo/comparison.md
#   results/streakmind_yolo/<track>/metrics_iou50.json
#   results/streakmind_yolo/<track>/metrics_iou80.json
#
# Usage (Mac M3, ~9–12 hours total for all 4 tracks):
#   bash scripts/train_streakmind_comparison.sh
#
# Override defaults:
#   EPOCHS=10 bash scripts/train_streakmind_comparison.sh
#   TRACKS="real_only" bash scripts/train_streakmind_comparison.sh
#   FRIGATE_RAW=/Volumes/External/frigate/raw bash scripts/train_streakmind_comparison.sh
#   SKIP_TRAIN=1 bash scripts/train_streakmind_comparison.sh  # eval only

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/Users/robert/miniconda3/envs/satid/bin/python}"
EPOCHS="${EPOCHS:-15}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-4}"
YOLO_MODEL="${YOLO_MODEL:-n}"
FRIGATE_RAW="${FRIGATE_RAW:-/Volumes/External/frigate/raw}"
FRIGATE_PROCESSED="${FRIGATE_PROCESSED:-/Volumes/External/frigate/processed}"
TRACKS="${TRACKS:-real_only paper_long adapted gtimages_plus_frigate}"
RESULTS_DIR="${RESULTS_DIR:-results/streakmind_yolo}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

echo "========================================================"
echo " ARGUS StreakMind Methodology-Matched Comparison"
echo " Model:   yolo11${YOLO_MODEL}-obb"
echo " Epochs:  ${EPOCHS}"
echo " Tracks:  ${TRACKS}"
echo " Results: ${RESULTS_DIR}"
echo "========================================================"

# ---------------------------------------------------------------------------
# 1. Check prerequisite annotation files
# ---------------------------------------------------------------------------
MISSING_SYNTH=0
for f in data/annotations/gtimages_train_real.json \
          data/annotations/gtimages_train_synth_paper_long.json \
          data/annotations/gtimages_train_synth_adapted.json \
          data/annotations/gtimages_val.json \
          data/annotations/gtimages_test.json; do
    if [[ ! -f "$f" ]]; then
        echo "Missing: $f"
        MISSING_SYNTH=1
    fi
done

if [[ "$MISSING_SYNTH" == "1" ]]; then
    echo ""
    echo "Generating GTImages annotation splits..."
    ${PYTHON} scripts/augment_gtimages_synthetic.py
    echo "Done."
fi

# ---------------------------------------------------------------------------
# 2. Generate Frigate negative corpus (if needed for combined track)
# ---------------------------------------------------------------------------
if echo "$TRACKS" | grep -q "gtimages_plus_frigate"; then
    if [[ ! -f "data/annotations/gtimages_plus_frigate_train.json" ]]; then
        echo ""
        echo "Generating Frigate negative corpus..."
        if [[ ! -d "$FRIGATE_RAW" ]]; then
            echo "WARNING: Frigate raw dir not found: $FRIGATE_RAW"
            echo "         Skipping gtimages_plus_frigate track."
            TRACKS=$(echo "$TRACKS" | tr ' ' '\n' | grep -v gtimages_plus_frigate | tr '\n' ' ')
        else
            FRIGATE_ARGS="--raw-dir ${FRIGATE_RAW}"
            if [[ -d "$FRIGATE_PROCESSED" ]]; then
                FRIGATE_ARGS="${FRIGATE_ARGS} --processed-dir ${FRIGATE_PROCESSED}"
            else
                FRIGATE_ARGS="${FRIGATE_ARGS} --raw-only"
            fi
            # Cap at 300 frames: 1980 full-res FITS take ~4 h to tile; 300 takes ~37 min.
            ${PYTHON} scripts/annotate_frigate.py \
                ${FRIGATE_ARGS} \
                --max-frames 300 \
                --output data/annotations/frigate_negatives.json

            ${PYTHON} scripts/merge_fits_annotations.py \
                --gtimages data/annotations/gtimages_train_real.json \
                --frigate  data/annotations/frigate_negatives.json \
                --output   data/annotations/gtimages_plus_frigate_train.json
            echo "Done."
        fi
    else
        echo "Reusing existing: data/annotations/gtimages_plus_frigate_train.json"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Run training and evaluation
# ---------------------------------------------------------------------------
echo ""
echo "Starting training (${EPOCHS} epochs per track, MPS/CPU)..."
echo "Estimated time: ~2–3 h/track on Mac M3 (~$((${EPOCHS} * 3)) min total per track)"
echo ""

SKIP_TRAIN_FLAG=""
if [[ "$SKIP_TRAIN" == "1" ]]; then
    SKIP_TRAIN_FLAG="--skip-train"
fi

# PYTORCH_ENABLE_MPS_FALLBACK is required for some YOLO ops on Apple Silicon.
PYTORCH_ENABLE_MPS_FALLBACK=1 \
${PYTHON} scripts/train_compare_streakmind_yolo.py \
    --tracks ${TRACKS} \
    --epochs "${EPOCHS}" \
    --imgsz  "${IMGSZ}" \
    --batch  "${BATCH}" \
    --model  "${YOLO_MODEL}" \
    --data-root data \
    --val-ann  data/annotations/gtimages_val.json \
    --test-ann data/annotations/gtimages_test.json \
    --results-dir "${RESULTS_DIR}" \
    --conf 0.25 \
    --eval-batch 4 \
    ${SKIP_TRAIN_FLAG} \
    --verbose

echo ""
echo "========================================================"
echo " Complete. Results written to: ${RESULTS_DIR}"
echo " Key files:"
echo "   ${RESULTS_DIR}/comparison.md   (human-readable, StreakMind reference included)"
echo "   ${RESULTS_DIR}/comparison.json (machine-readable)"
echo "========================================================"
