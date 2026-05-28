#!/usr/bin/env bash
# Run plain-DINOv3 box evaluation on val.json, then compare against heatmap
# centerline proposals using line-segment metrics.
#
# Assumes the box checkpoint already exists at:
#   weights/run_plain_dinov3_box_gauss_384_regen/best.pt
#
# Heatmap proposals sourced from the prior spike work:
#   /Users/robert/Argus-dinov3-heatmap-spike/results/
#     dinov3_vitb_orientation_centerline_input512_catchment_segments_t050_oc055/proposals.json

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
CHECKPOINT=weights/run_plain_dinov3_box_gauss_384_regen/best.pt
VAL_ANNOTATIONS=data/annotations/val.json
HEATMAP_PROPOSALS=results/dinov3_vitb_orientation_centerline_input512_catchment_segments_t050_oc055/proposals.json
BOX_OUTPUT_DIR=results/plain_dinov3_box_gauss_384_regen_val
COMPARISON_OUTPUT=results/heatmap_vs_obb_line_comparison.json

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "=== Step 1: Evaluate box head on val.json ==="
$PYTHON -m scripts.evaluate_dinov3_box \
  --annotations "$VAL_ANNOTATIONS" \
  --checkpoint "$CHECKPOINT" \
  --weights weights/dinov3_vitb16_lvd1689m.pth \
  --output "$BOX_OUTPUT_DIR/metrics_t0.10.json" \
  --threshold 0.10 \
  --batch-size 1

echo "=== Step 2: Compare heatmap centerline vs OBB on val.json ==="
$PYTHON -m scripts.compare_heatmap_centerline_to_obb \
  --annotations "$VAL_ANNOTATIONS" \
  --heatmap-predictions "$HEATMAP_PROPOSALS" \
  --obb-predictions "$BOX_OUTPUT_DIR/predictions.json" \
  --obb-method "plain_dinov3_box_384" \
  --output "$COMPARISON_OUTPUT" \
  --tolerance-px 6.0 \
  --coverage-threshold 0.10

echo "=== Done. Results at: $COMPARISON_OUTPUT ==="
