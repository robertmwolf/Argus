#!/usr/bin/env bash
# train_yolo_full.sh — Full-dataset YOLO11m-OBB training + evaluation.
#
# Target: ~12 hours on Mac M3 (CPU), faster on GPU.
# Produces:  weights/run_full_yolo_obb/run/weights/best.pt
#            results/full_yolo_obb/yolo_benchmark.json
#            results/full_yolo_obb/yolo_predictions.json
#
# Usage:
#   bash scripts/train_yolo_full.sh              # 50 epochs, yolo11m
#   YOLO_EPOCHS=100 bash scripts/train_yolo_full.sh
#   YOLO_MODEL=l bash scripts/train_yolo_full.sh  # use yolo11l for higher quality

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/Users/robert/miniconda3/envs/satid/bin/python}"
# yolo11n on Mac M3 CPU: ~5-8 hours for 50 epochs (12h budget OK)
# yolo11m on GPU (RTX 5070 Ti): ~2-3 hours for 50 epochs — override with YOLO_MODEL=m
YOLO_MODEL="${YOLO_MODEL:-n}"
# 15 epochs × ~36 min on Mac M3 CPU ≈ 9h; override with YOLO_EPOCHS=50 on GPU
YOLO_EPOCHS="${YOLO_EPOCHS:-15}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
WORK_DIR="${WORK_DIR:-weights/run_full_yolo_obb}"
RESULTS_DIR="${RESULTS_DIR:-results/full_yolo_obb}"

echo "========================================================"
echo " ARGUS YOLO Full-Dataset Training"
echo " Model:   yolo11${YOLO_MODEL}-obb"
echo " Epochs:  ${YOLO_EPOCHS}"
echo " ImgSz:   ${YOLO_IMGSZ}"
echo " WorkDir: ${WORK_DIR}"
echo "========================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Verify data
# ---------------------------------------------------------------------------
if [[ ! -f "data/annotations/train.json" ]]; then
    echo "ERROR: data/annotations/train.json not found."
    echo "Run: python scripts/merge_annotations.py --seed 42 --val-fraction 0.2"
    exit 1
fi

$PYTHON - <<'PY'
import json, pathlib
for split in ["train", "val", "test"]:
    p = pathlib.Path(f"data/annotations/{split}.json")
    if p.exists():
        d = json.loads(p.read_text())
        print(f"  {split}: images={len(d['images'])}, anns={len(d['annotations'])}")
PY
echo ""

# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------
echo "[1/3] Training YOLO11${YOLO_MODEL}-OBB on full dataset..."
USE_DEV_SUBSET=false \
    $PYTHON -m training.train_baseline \
        --model "${YOLO_MODEL}" \
        --imgsz "${YOLO_IMGSZ}" \
        --epochs "${YOLO_EPOCHS}" \
        --work-dir "${WORK_DIR}"

echo ""
echo "Training complete."

# ---------------------------------------------------------------------------
# 3. Locate best checkpoint
# ---------------------------------------------------------------------------
BEST_PT="${WORK_DIR}/run/weights/best.pt"
if [[ ! -f "${BEST_PT}" ]]; then
    echo "ERROR: best.pt not found at ${BEST_PT}"
    exit 1
fi
echo "Best weights: ${BEST_PT}"
echo ""

# ---------------------------------------------------------------------------
# 4. Evaluate on test split
# ---------------------------------------------------------------------------
echo "[2/3] Evaluating on test split..."
mkdir -p "${RESULTS_DIR}"

USE_DEV_SUBSET=false \
    $PYTHON - <<PYEVAL
import json, pathlib, sys
sys.path.insert(0, ".")

from eval.benchmark import run_pipeline_predictions, load_ground_truth
from eval.metrics import evaluate

ann_path = pathlib.Path("data/annotations/test.json")
print(f"Running tiled YOLO inference on {ann_path} ...")
preds = run_pipeline_predictions(ann_path, model="yolo")
gt    = load_ground_truth(ann_path)
metrics = evaluate(preds, gt)

out = pathlib.Path("${RESULTS_DIR}/yolo_benchmark.json")
preds_out = pathlib.Path("${RESULTS_DIR}/yolo_predictions.json")

import json, datetime
result = {
    "date_recorded": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "yolo11${YOLO_MODEL}-obb-full",
    "weights": "${BEST_PT}",
    "training_epochs": ${YOLO_EPOCHS},
    "training_dataset": "full dataset (train.json)",
    "imgsz": ${YOLO_IMGSZ},
    **metrics,
}
out.write_text(json.dumps(result, indent=2))
preds_out.write_text(json.dumps(preds, indent=2))

print(f"\n=== YOLO Full-Dataset Results ===")
print(f"  mAP@0.5:     {metrics['map_50']:.3f}")
print(f"  Precision:   {metrics['precision']:.3f}")
print(f"  Recall:      {metrics['recall']:.3f}")
print(f"  F1:          {metrics['f1']:.3f}")
print(f"  Angle error: {metrics['mean_angle_error_deg']:.2f}°")
print(f"\nResults saved to ${RESULTS_DIR}/")
PYEVAL

echo ""
echo "[3/3] Done. Results in ${RESULTS_DIR}/"
echo ""
echo "Next: commit results and compare against Co-DINO in phase_e_compare.py"
