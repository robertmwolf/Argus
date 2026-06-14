#!/usr/bin/env bash
# Run evaluate_dinov3_heatmap.py + geometry_metrics.py for a single model
# on val_balanced_v1.json, using a pre-built heatmap cache.
#
# Usage:
#   bash scripts/eval_balanced_v1.sh <run_name> <checkpoint> <backbone> <heatmap_cache_dir>
#
# Outputs land in results/<run_name>/balanced_v1/pf85/

set -euo pipefail

RUN=$1
CHECKPOINT=$2
BACKBONE=$3   # vits or vitb (controls backbone weights arg if needed)
CACHE_DIR=$4

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
ANN=data/annotations/val_balanced_v1.json
OUT_DIR=results/${RUN}/balanced_v1/pf85

mkdir -p "$OUT_DIR"

echo "=== Evaluating ${RUN} on val_balanced_v1 ==="
PYTORCH_ENABLE_MPS_FALLBACK=1 $PYTHON scripts/evaluate_dinov3_heatmap.py \
  --annotations "$ANN" \
  --heatmap-cache "$CACHE_DIR" \
  --tiled --stitch --stitch-max-growth-ratio 3.0 \
  --threshold 0.70 \
  --peak-floor 0.85 \
  --output "${OUT_DIR}/metrics_t070.json"

echo "=== Running geometry eval for ${RUN} ==="
$PYTHON -m eval.geometry_metrics \
  --predictions "${OUT_DIR}/predictions_t070.json" \
  --annotations "$ANN" \
  --output "${OUT_DIR}/geometry_eval.json"

echo "=== Done: ${RUN} ==="
