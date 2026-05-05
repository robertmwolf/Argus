"""Head-to-head benchmark: DINO vs YOLO11-OBB baseline on a test split.

Usage:
    # Run against a pre-computed predictions file (no GPU needed):
    python -m eval.benchmark \\
        --dino-predictions results/dino_predictions.json \\
        --yolo-predictions results/yolo_predictions.json \\
        --annotations data/annotations/test.json \\
        --output results/phase8_benchmark.json

    # Run the pipeline live (requires weights/best.pth):
    python -m eval.benchmark \\
        --run-pipeline \\
        --annotations data/annotations/test.json \\
        --output results/phase8_benchmark.json

Prediction JSON format (list of detections):
    [
      {"image_id": "img001.fits", "confidence": 0.92,
       "obb": {"cx":100,"cy":200,"w":300,"h":12,"angle_deg":5.1},
       "streak_length_px": 300},
      ...
    ]

# Source: StreakMind — evaluation and benchmark methodology
# Ref: agent_docs/streakmind_phases.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from eval.metrics import evaluate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COCO annotation loader
# ---------------------------------------------------------------------------

def load_ground_truth(annotations_path: str | Path) -> list[dict]:
    """Load COCO-format annotations and convert to metrics-compatible format.

    Args:
        annotations_path: Path to COCO JSON annotation file.

    Returns:
        List of dicts with image_id, obb, and streak_length_px.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    ground_truth = []
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        obb_list = ann.get("obb")
        if not obb_list or len(obb_list) < 5:
            continue
        cx, cy, w, h, angle_deg = obb_list[:5]
        ground_truth.append({
            "image_id": id_to_filename.get(ann["image_id"], str(ann["image_id"])),
            "obb": {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle_deg},
            "streak_length_px": max(w, h),
        })

    return ground_truth


# ---------------------------------------------------------------------------
# Pipeline-based prediction runner
# ---------------------------------------------------------------------------

def run_pipeline_predictions(
    annotations_path: str | Path,
    model: str = "dino",
) -> list[dict]:
    """Run the inference pipeline on every image in the annotation file.

    Args:
        annotations_path: Path to COCO JSON annotation file.
        model: "dino" (uses inference/pipeline.py) or "yolo" (uses YOLO baseline).

    Returns:
        List of prediction dicts compatible with evaluate().

    Raises:
        FileNotFoundError: If model weights are not found.
        NotImplementedError: If model is not "dino" or "yolo".
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    # Resolve image directory (annotations live in data/annotations/, images in data/raw/)
    ann_path = Path(annotations_path)
    image_dir = ann_path.parent.parent / "raw"

    predictions = []

    if model == "dino":
        from inference.pipeline import load_model, run as pipeline_run

        logger.info("Loading DINO model (once for all images)…")
        dino_model, dino_device = load_model()

        for img_info in coco["images"]:
            fits_path = image_dir / img_info["file_name"]
            if not fits_path.exists():
                logger.warning("Image not found, skipping: %s", fits_path)
                continue
            logger.info("Running DINO on %s", fits_path.name)
            dets = pipeline_run(
                fits_path=fits_path,
                fast=True,
                model=dino_model,
                inference_device=dino_device,
            )
            for det in dets:
                predictions.append({
                    "image_id": img_info["file_name"],
                    "confidence": det.get("confidence", 0.0),
                    "obb": det["obb"],
                    "streak_length_px": det.get("streak_length_px", 0.0),
                })

    elif model == "yolo":
        # Source: Ultralytics YOLO11-OBB baseline inference
        try:
            from ultralytics import YOLO
            import cv2
        except ImportError as exc:
            raise ImportError("ultralytics and opencv are required for YOLO baseline") from exc

        weights = Path("weights/yolo_baseline.pt")
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")

        yolo = YOLO(str(weights))

        # YOLO cannot read FITS — find the matching PNG in the pre-converted dataset
        png_train = Path("weights/yolo_baseline/dataset/images/train")
        png_val   = Path("weights/yolo_baseline/dataset/images/val")

        for img_info in coco["images"]:
            stem = Path(img_info["file_name"]).stem
            # Search train then val for the converted PNG
            png_path = None
            for png_dir in (png_train, png_val):
                candidate = png_dir / (stem + ".png")
                if candidate.exists():
                    png_path = candidate
                    break
            if png_path is None:
                logger.warning("PNG not found for %s, skipping", stem)
                continue
            logger.info("Running YOLO on %s", png_path.name)
            results = yolo(str(png_path))
            for result in results:
                if result.obb is None:
                    continue
                for box, conf in zip(result.obb.xywhr, result.obb.conf):
                    cx, cy, w, h, angle_rad = box.tolist()
                    predictions.append({
                        "image_id": img_info["file_name"],
                        "confidence": float(conf),
                        "obb": {
                            "cx": cx, "cy": cy,
                            "w": w, "h": h,
                            "angle_deg": float(angle_rad) * 180 / 3.14159265,
                        },
                        "streak_length_px": max(w, h),
                    })
    else:
        raise NotImplementedError(f"Unknown model: {model!r}")

    return predictions


# ---------------------------------------------------------------------------
# Markdown table formatter
# ---------------------------------------------------------------------------

def _fmt(value: float, pct: bool = True) -> str:
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def format_markdown_table(dino_metrics: dict, yolo_metrics: dict | None) -> str:
    """Render benchmark results as a Markdown table.

    Args:
        dino_metrics: Output of evaluate() for DINO.
        yolo_metrics: Output of evaluate() for YOLO, or None.

    Returns:
        Markdown-formatted string.
    """
    rows = [
        ("Precision",         "precision",           True),
        ("Recall",            "recall",              True),
        ("F1",                "f1",                  True),
        ("mAP@0.5",           "map_50",              True),
        ("mAP@0.75",          "map_75",              True),
        ("Angle error (°)",   "mean_angle_error_deg", False),
    ]

    col_yolo = yolo_metrics is not None
    header = "| Metric | DINO (Swin-L)" + (" | YOLO11-OBB" if col_yolo else "") + " | Target |"
    sep    = "|--------|---------------" + ("-|------------" if col_yolo else "") + "-|--------|"
    lines = [header, sep]

    targets = {
        "precision": "≥ 94%",
        "recall":    "≥ 97%",
        "f1":        "—",
        "map_50":    "—",
        "map_75":    "—",
        "mean_angle_error_deg": "—",
    }

    for label, key, pct in rows:
        dino_val = _fmt(dino_metrics.get(key, 0.0), pct)
        target = targets.get(key, "—")
        if col_yolo:
            yolo_val = _fmt(yolo_metrics.get(key, 0.0), pct)
            lines.append(f"| {label} | {dino_val} | {yolo_val} | {target} |")
        else:
            lines.append(f"| {label} | {dino_val} | {target} |")

    # Per-band breakdown
    lines.append("")
    lines.append("### Per-band (DINO)")
    lines.append("| Band | Precision | Recall | F1 |")
    lines.append("|------|-----------|--------|----|")
    for band in ("short", "medium", "long"):
        b = dino_metrics.get("per_band", {}).get(band, {})
        lines.append(
            f"| {band.capitalize()} (<150 / 150–400 / >400 px)"
            f" | {_fmt(b.get('precision', 0))}"
            f" | {_fmt(b.get('recall', 0))}"
            f" | {_fmt(b.get('f1', 0))} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    annotations_path: str | Path,
    dino_predictions: list[dict] | None = None,
    yolo_predictions: list[dict] | None = None,
    output_path: str | Path = "results/phase8_benchmark.json",
    run_pipeline: bool = False,
) -> dict:
    """Run the full benchmark and save results.

    Args:
        annotations_path: Path to COCO JSON annotation file (test split).
        dino_predictions: Pre-computed DINO predictions (skips pipeline run).
        yolo_predictions: Pre-computed YOLO predictions (skips pipeline run).
        output_path: Where to save the JSON results file.
        run_pipeline: If True, run pipeline live (requires weights).

    Returns:
        Results dict saved to output_path.
    """
    ground_truth = load_ground_truth(annotations_path)
    logger.info("Loaded %d ground-truth annotations", len(ground_truth))

    if run_pipeline and dino_predictions is None:
        logger.info("Running DINO pipeline on test set…")
        dino_predictions = run_pipeline_predictions(annotations_path, model="dino")

    if run_pipeline and yolo_predictions is None:
        try:
            logger.info("Running YOLO baseline on test set…")
            yolo_predictions = run_pipeline_predictions(annotations_path, model="yolo")
        except (FileNotFoundError, NotImplementedError) as exc:
            logger.warning("YOLO baseline skipped: %s", exc)
            yolo_predictions = None

    dino_metrics = evaluate(dino_predictions or [], ground_truth)
    yolo_metrics = evaluate(yolo_predictions, ground_truth) if yolo_predictions else None

    results = {
        "date_recorded": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "dino_swin_l",
        "map_50": dino_metrics["map_50"],
        "map_75": dino_metrics["map_75"],
        "recall": dino_metrics["recall"],
        "precision": dino_metrics["precision"],
        "f1": dino_metrics["f1"],
        "mean_angle_error_deg": dino_metrics["mean_angle_error_deg"],
        "per_band": dino_metrics["per_band"],
        "yolo_baseline": {
            "map_50": yolo_metrics["map_50"] if yolo_metrics else 0.0,
            "recall": yolo_metrics["recall"] if yolo_metrics else 0.0,
            "precision": yolo_metrics["precision"] if yolo_metrics else 0.0,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Save per-image predictions to eval/results/
    eval_results_dir = Path("eval/results")
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    if dino_predictions:
        with open(eval_results_dir / "dino_predictions.json", "w") as f:
            json.dump(dino_predictions, f, indent=2)
    if yolo_predictions:
        with open(eval_results_dir / "yolo_predictions.json", "w") as f:
            json.dump(yolo_predictions, f, indent=2)

    # Print markdown table to stdout
    print("\n" + format_markdown_table(dino_metrics, yolo_metrics) + "\n")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="ARGUS Phase 8 benchmark")
    parser.add_argument("--annotations", required=True, help="COCO JSON annotation file")
    parser.add_argument("--dino-predictions", help="Pre-computed DINO predictions JSON")
    parser.add_argument("--yolo-predictions", help="Pre-computed YOLO predictions JSON")
    parser.add_argument("--run-pipeline", action="store_true", help="Run pipeline live (needs weights)")
    parser.add_argument("--output", default="results/phase8_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    dino_preds = None
    if args.dino_predictions:
        with open(args.dino_predictions) as f:
            dino_preds = json.load(f)

    yolo_preds = None
    if args.yolo_predictions:
        with open(args.yolo_predictions) as f:
            yolo_preds = json.load(f)

    run_benchmark(
        annotations_path=args.annotations,
        dino_predictions=dino_preds,
        yolo_predictions=yolo_preds,
        output_path=args.output,
        run_pipeline=args.run_pipeline,
    )
