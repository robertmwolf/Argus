#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/Users/robert/miniconda3/envs/satid/bin/python}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

"${PYTHON_BIN}" training/train_dinov3_orientation_centerline.py \
  --train-annotations data/annotations/train.json \
  --val-annotations data/annotations/val.json \
  --work-dir weights/run_dinov3_vitb_orientation_centerline_input512_catchment \
  --weights weights/dinov3_vitb16_lvd1689m.pth \
  --model-size base \
  --tile-size 2560 \
  --image-size 512 \
  --positive-train-tiles 1236 \
  --negative-train-tiles 1400 \
  --orientation-bins 18 \
  --decoder-channels 192 \
  --last-layers 4 \
  --centerline-width 2.0 \
  --centerline-sigma 1.4 \
  --catchment-width 14.0 \
  --catchment-sigma 6.0 \
  --neighbor-bin-weight 0.35 \
  --second-neighbor-weight 0.0 \
  --epochs 10 \
  --batch-size 1 \
  --lr 5e-5 \
  --min-lr 1e-5 \
  --weight-decay 1e-4 \
  --pos-weight 60 \
  --dice-weight 1.0 \
  --bce-weight 0.10 \
  --orientation-ce-weight 0.20 \
  --manual-positive-weight 3.0 \
  --catchment-loss-weight 0.35 \
  --catchment-pos-weight 20 \
  --catchment-dice-weight 1.0 \
  --catchment-bce-weight 0.20 \
  --image-loss-weight 0.25 \
  --workers 0 \
  --seed 20260524 \
  --preserve-image-bit-depth \
  --log-interval 200
