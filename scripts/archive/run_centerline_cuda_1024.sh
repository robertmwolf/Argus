#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/Users/robert/miniconda3/envs/satid/bin/python}"

"${PYTHON_BIN}" training/train_dinov3_orientation_centerline.py \
  --train-annotations "${TRAIN_ANNOTATIONS:-data/annotations/train.json}" \
  --val-annotations "${VAL_ANNOTATIONS:-data/annotations/val.json}" \
  --work-dir "${WORK_DIR:-weights/run_dinov3_vitb_orientation_centerline_tile2560_input1024_cuda}" \
  --weights "${DINO_WEIGHTS:-weights/dinov3_vitb16_lvd1689m.pth}" \
  --model-size "${MODEL_SIZE:-base}" \
  --tile-size 2560 \
  --image-size 1024 \
  --positive-train-tiles 1236 \
  --negative-train-tiles 1400 \
  --orientation-bins 18 \
  --decoder-channels 192 \
  --last-layers 4 \
  --centerline-width 2.0 \
  --centerline-sigma 1.4 \
  --neighbor-bin-weight 0.35 \
  --second-neighbor-weight 0.0 \
  --epochs "${EPOCHS:-10}" \
  --batch-size 1 \
  --lr 5e-5 \
  --min-lr 1e-5 \
  --weight-decay 1e-4 \
  --pos-weight "${POS_WEIGHT:-60}" \
  --dice-weight 1.0 \
  --bce-weight "${BCE_WEIGHT:-0.10}" \
  --orientation-ce-weight 0.20 \
  --manual-positive-weight "${MANUAL_POSITIVE_WEIGHT:-3.0}" \
  --image-loss-weight "${IMAGE_LOSS_WEIGHT:-0.25}" \
  --workers "${WORKERS:-8}" \
  --seed 20260524 \
  --preserve-image-bit-depth \
  --amp \
  --log-interval 100
