#!/usr/bin/env bash
# Train DINOv3 ViT-B on GT + DM + SatStreaks combined dataset
#
# Dataset: GTImages (synthetic) + SatStreaks (HST FITS) + DarkMatters CDK20
#   Train : data/annotations/dm_merged_train.json  (3 172 images, 3 132 OBBs)
#   Val   : data/annotations/dm_merged_val.json    (  477 images,   490 OBBs)
#
# Annotation files use TRAIN_ANN_FILE / VAL_ANN_FILE env vars (paths relative
# to the data/ directory).  USE_DEV_SUBSET=false activates the full dataset.
# ARGUS_NORM=autostretch is the normalisation used for all real-image runs.
#
# On Mac (no CUDA) the run uses CPU automatically; on A100 it uses CUDA.
# Pass --smoke-test as the first argument to verify setup before a full run.
#
# Usage:
#   bash scripts/train_gt_dm_satstreaks.sh               # full training
#   bash scripts/train_gt_dm_satstreaks.sh --smoke-test  # sanity check only

set -euo pipefail

PYTORCH_ENABLE_MPS_FALLBACK=1 \
USE_DEV_SUBSET=false \
ARGUS_NORM=autostretch \
TRAIN_ANN_FILE=annotations/dm_merged_train.json \
VAL_ANN_FILE=annotations/dm_merged_val.json \
  /Users/robert/miniconda3/envs/satid/bin/python -m training.train_dino \
    --backbone dinov3_vitb \
    --work-dir weights/run_gt_dm_satstreaks_dinov3_vitb \
    "$@"
