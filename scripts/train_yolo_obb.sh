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

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Config ────────────────────────────────────────────────────────────────────
DATASET=/Volumes/External/TrainingData/yolo_run17_dataset
WEIGHTS="$REPO_ROOT/weights/yolo11s-obb.pt"
OUT_PROJECT="$REPO_ROOT/weights/yolo_run17"
OUT_NAME="${YOLO_OUT_NAME:-run}"
# Training-image list (one png path per line). Override for a curated subset,
# e.g. train_subset3k.txt for a CPU-feasible run.
TRAIN_LIST="${YOLO_TRAIN_LIST:-train_10pct_bg.txt}"
SSD_DIR=/tmp/yolo_run17
EPOCHS=20
BATCH=4
RESUME=false
LAST_CKPT=""   # only set when --resume; default empty so set -u is happy on fresh runs
# --continue-from=PATH warm-restarts from existing weights and trains EPOCHS *new*
# epochs with a fresh LR schedule (unlike --resume, which can't extend past the
# original run length). Use it to train an already-converged model further.
CONTINUE_FROM=""

for arg in "$@"; do
    case $arg in
        --epochs=*)        EPOCHS="${arg#*=}" ;;
        --batch=*)         BATCH="${arg#*=}" ;;
        --resume)          RESUME=true ;;
        --continue-from=*) CONTINUE_FROM="${arg#*=}" ;;
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
done < "$DATASET/$TRAIN_LIST"

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

# ── Device ────────────────────────────────────────────────────────────────────
# WARNING: OBB training is numerically BROKEN on Apple MPS in ultralytics 8.4.46.
# A 25-tile overfit reaches mAP 0.95 on CPU but DIVERGES to mAP 0 on MPS (both
# AdamW and SGD). All three "Run 17" attempts failed for exactly this reason —
# cls_loss never converges, the model emits 0 detections. The proven-working
# streakmind model trained on device=cpu. Do NOT set DEVICE=mps for a real run
# until this is verified fixed on a newer ultralytics/torch. Use cpu or a CUDA box.
DEVICE="${YOLO_DEVICE:-cpu}"
if [ "$DEVICE" = "mps" ]; then
    echo "!!! WARNING: device=mps — OBB training diverges on MPS in this ultralytics" >&2
    echo "!!! version (see memory: yolo-obb-mps-broken). The model will be dead." >&2
fi

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Training: epochs=$EPOCHS batch=$BATCH resume=$RESUME device=$DEVICE ==="

RESUME_FLAG=""
if [ "$RESUME" = true ]; then
    # Check canonical location first; fall back to runs/obb/ prefix that
    # ultralytics uses when project= was a relative path on the original run.
    LAST_CKPT="$OUT_PROJECT/$OUT_NAME/weights/last.pt"
    if [ ! -f "$LAST_CKPT" ]; then
        LAST_CKPT="$REPO_ROOT/runs/obb/weights/yolo_run17/$OUT_NAME/weights/last.pt"
    fi
    if [ -f "$LAST_CKPT" ]; then
        echo "Resuming from $LAST_CKPT"
        RESUME_FLAG="resume=True model=$LAST_CKPT"
    else
        echo "No checkpoint found at $LAST_CKPT, starting fresh."
    fi
fi

if [ -n "$CONTINUE_FROM" ]; then
    echo "Continuing (warm restart) from $CONTINUE_FROM for $EPOCHS new epochs"
fi

$PYTHON -c "
from ultralytics import YOLO
resume = '${RESUME_FLAG}' != ''
continue_from = '${CONTINUE_FROM}'
if resume:
    model = YOLO('${LAST_CKPT}')
    model.train(resume=True)
else:
    # Fresh COCO weights, or warm-restart from an existing checkpoint when
    # --continue-from is given (same fresh config, just different start weights).
    model = YOLO(continue_from if continue_from else '${WEIGHTS}')
    model.train(
        task='obb',
        data='${YAML}',
        imgsz=416,
        epochs=${EPOCHS},
        batch=${BATCH},
        device='${DEVICE}',
        workers=2,
        cos_lr=True,
        # --- cls-head fixes (Run 17 trained a dead objectness head) ---
        warmup_bias_lr=0.1,   # MUST be >0: a stray 0.0 froze the cls bias in
                              # Run 17 → cls_loss never dropped → 0 detections.
        amp=False,            # MPS autocast is unreliable for the cls logits;
                              # the proven recipe ran amp as a CPU no-op.
        degrees=0.0,          # no rotation aug: it corrupts thin-streak OBB
                              # positives and starves the cls head.
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
