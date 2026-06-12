#!/usr/bin/env bash
# train_yolo_obb.sh — Stage dataset to SSD, train YOLO OBB, clean up.
#
# Usage:
#   bash scripts/train_yolo_obb.sh [--epochs N] [--batch N] [--resume]
#
# Defaults: epochs=20, batch=4, no resume.
# Weights are saved to weights/yolo_run17/run/.
# SSD staging is cleaned up on exit (success or failure).

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DATASET=/Volumes/External/TrainingData/yolo_run17_dataset
WEIGHTS=weights/yolo11s-obb.pt
OUT_PROJECT=weights/yolo_run17
OUT_NAME=run
SSD_DIR=/tmp/yolo_run17
EPOCHS=20
BATCH=4
RESUME=false

for arg in "$@"; do
    case $arg in
        --epochs=*) EPOCHS="${arg#*=}" ;;
        --batch=*)  BATCH="${arg#*=}" ;;
        --resume)   RESUME=true ;;
    esac
done

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python

# ── Cleanup trap ──────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Cleaning up SSD staging at $SSD_DIR ..."
    rm -rf "$SSD_DIR"
    echo "Done."
}
trap cleanup EXIT

# ── Stage dataset to SSD ──────────────────────────────────────────────────────
echo "=== Staging dataset to $SSD_DIR ==="
mkdir -p "$SSD_DIR/images/train" "$SSD_DIR/images/val" \
          "$SSD_DIR/labels/train"  "$SSD_DIR/labels/val"

echo "Copying train images..."
while IFS= read -r img; do
    cp "$img" "$SSD_DIR/images/train/"
    stem=$(basename "$img" .png)
    cp "$DATASET/labels/train/${stem}.txt" "$SSD_DIR/labels/train/" 2>/dev/null || true
done < "$DATASET/train_10pct_bg.txt"

echo "Copying val images..."
while IFS= read -r img; do
    cp "$img" "$SSD_DIR/images/val/"
    stem=$(basename "$img" .png)
    cp "$DATASET/labels/val/${stem}.txt" "$SSD_DIR/labels/val/" 2>/dev/null || true
done < "$DATASET/val_annotated.txt"

TRAIN_COUNT=$(ls "$SSD_DIR/images/train/" | wc -l | tr -d ' ')
VAL_COUNT=$(ls "$SSD_DIR/images/val/" | wc -l | tr -d ' ')
echo "Staged: $TRAIN_COUNT train, $VAL_COUNT val"

# ── Write dataset yaml ────────────────────────────────────────────────────────
YAML=$SSD_DIR/dataset.yaml
cat > "$YAML" <<EOF
# YOLO OBB — Run 17 (staged to SSD)
# Train: 21,432 positive + 2,143 background tiles (10% bg ratio)
# Val: 909 annotated tiles
path: $SSD_DIR
train: images/train
val:   images/val

nc: 1
names:
  - streak
EOF

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Training: epochs=$EPOCHS batch=$BATCH resume=$RESUME ==="

RESUME_FLAG=""
if [ "$RESUME" = true ]; then
    LAST_CKPT="$OUT_PROJECT/$OUT_NAME/weights/last.pt"
    if [ -f "$LAST_CKPT" ]; then
        echo "Resuming from $LAST_CKPT"
        RESUME_FLAG="resume=True model=$LAST_CKPT"
    else
        echo "No checkpoint found at $LAST_CKPT, starting fresh."
    fi
fi

$PYTHON -c "
from ultralytics import YOLO
resume = '${RESUME_FLAG}' != ''
if resume:
    model = YOLO('${OUT_PROJECT}/${OUT_NAME}/weights/last.pt')
    model.train(resume=True)
else:
    model = YOLO('${WEIGHTS}')
    model.train(
        task='obb',
        data='${YAML}',
        imgsz=416,
        epochs=${EPOCHS},
        batch=${BATCH},
        device='mps',
        workers=2,
        cos_lr=True,
        degrees=10,
        hsv_s=0,
        hsv_h=0,
        project='${OUT_PROJECT}',
        name='${OUT_NAME}',
        exist_ok=True,
        patience=15,
        save_period=5,
        cache=True,
    )
"

echo ""
echo "=== Training complete. Weights at $OUT_PROJECT/$OUT_NAME/weights/ ==="
