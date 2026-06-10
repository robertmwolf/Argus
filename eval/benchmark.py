"""Head-to-head benchmark: DINO on a test split.

Usage:
    # Run against a pre-computed predictions file (no GPU needed):
    python -m eval.benchmark \\
        --dino-predictions results/dino_predictions.json \\
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
# Ref: agent_docs/argus_phases.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from eval.metrics import evaluate, confusion_matrix

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
        obb_raw = ann.get("obb")
        if not obb_raw:
            continue
        if isinstance(obb_raw, dict):
            try:
                cx = float(obb_raw["cx"])
                cy = float(obb_raw["cy"])
                w = float(obb_raw["w"])
                h = float(obb_raw["h"])
                angle_deg = float(obb_raw["angle_deg"])
            except (KeyError, TypeError, ValueError):
                continue
        else:
            if len(obb_raw) < 5:
                continue
            cx, cy, w, h, angle_deg = [float(v) for v in obb_raw[:5]]
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
) -> list[dict]:
    """Run the inference pipeline on every image in the annotation file.

    Args:
        annotations_path: Path to COCO JSON annotation file.

    Returns:
        List of prediction dicts compatible with evaluate().

    Raises:
        FileNotFoundError: If model weights are not found.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    # Resolve image directory (annotations live in data/annotations/, images in data/raw/)
    ann_path = Path(annotations_path)
    image_dir = ann_path.parent.parent / "raw"

    predictions = []

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

    return predictions


# ---------------------------------------------------------------------------
# Markdown table formatter
# ---------------------------------------------------------------------------

def _fmt(value: float, pct: bool = True) -> str:
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def format_markdown_table(dino_metrics: dict) -> str:
    """Render benchmark results as a Markdown table.

    Args:
        dino_metrics: Output of evaluate() for DINO.

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

    header = "| Metric | DINO (Swin-L) | Target |"
    sep    = "|--------|---------------|--------|"
    lines = [header, sep]

    targets = {
        "precision": ">= 94%",
        "recall":    ">= 97%",
        "f1":        "--",
        "map_50":    "--",
        "map_75":    "--",
        "mean_angle_error_deg": "--",
    }

    for label, key, pct in rows:
        dino_val = _fmt(dino_metrics.get(key, 0.0), pct)
        target = targets.get(key, "--")
        lines.append(f"| {label} | {dino_val} | {target} |")

    # Per-band breakdown
    lines.append("")
    lines.append("### Per-band (DINO)")
    lines.append("| Band | Precision | Recall | F1 |")
    lines.append("|------|-----------|--------|----|")
    for band in ("short", "medium", "long"):
        b = dino_metrics.get("per_band", {}).get(band, {})
        lines.append(
            f"| {band.capitalize()} (<150 / 150-400 / >400 px)"
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
    output_path: str | Path = "results/phase8_benchmark.json",
    run_pipeline: bool = False,
) -> dict:
    """Run the full benchmark and save results.

    Args:
        annotations_path: Path to COCO JSON annotation file (test split).
        dino_predictions: Pre-computed DINO predictions (skips pipeline run).
        output_path: Where to save the JSON results file.
        run_pipeline: If True, run pipeline live (requires weights).

    Returns:
        Results dict saved to output_path.
    """
    ground_truth = load_ground_truth(annotations_path)
    logger.info("Loaded %d ground-truth annotations", len(ground_truth))

    if run_pipeline and dino_predictions is None:
        logger.info("Running DINO pipeline on test set…")
        dino_predictions = run_pipeline_predictions(annotations_path)

    dino_metrics = evaluate(dino_predictions or [], ground_truth)

    # Confusion matrix — saved as PNG alongside the JSON output
    cm_path = Path(output_path).parent / "confusion_matrix.png"
    cm = confusion_matrix(
        dino_predictions or [], ground_truth, iou_threshold=0.5,
        save_path=cm_path,
    )
    logger.info(
        "Confusion matrix: TP=%d  FP=%d  FN=%d  TN=%d  → %s",
        cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1], cm_path,
    )

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
        "confusion_matrix": {
            "TP": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TN": int(cm[1, 1]),
            "confusion_matrix_png": str(cm_path),
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

    # Print markdown table to stdout
    print("\n" + format_markdown_table(dino_metrics) + "\n")

    return results


# ---------------------------------------------------------------------------
# Multi-method comparison
# ---------------------------------------------------------------------------

# Display order (unified always first, then ML methods, then classical).
_METHOD_ORDER = [
    "unified", "dinov3_heatmap_centerline", "dinov3_vitb", "dinov3_vitl", "tiny", "large",
    "streakmind_yolo", "opencv", "classical", "ml",
]

_METHOD_LABELS = {
    "unified":                 "Unified Confidence Score",
    "dinov3_heatmap_centerline": "DINOv3 Heatmap Centerline",
    "dinov3_vitb":             "DINOv3 ViT-B",
    "dinov3_vitl":             "DINOv3 ViT-L",
    "tiny":        "DINO Swin-T",
    "large":       "DINO Swin-L",
    "opencv":      "OpenCV",
    "classical":   "Classical",
    "ml":          "ML",
}


def format_comparison_table(method_metrics: dict[str, dict]) -> str:
    """Render a multi-method comparison as a Markdown table.

    Methods are ordered with ``"unified"`` first, then ML methods, then
    classical detectors.  Unknown method names appear last in alphabetical
    order.

    Args:
        method_metrics: Dict mapping method name → metrics dict from
            ``evaluate()``.

    Returns:
        Markdown-formatted comparison table string.
    """
    def _sort_key(m: str) -> tuple[int, str]:
        try:
            return (_METHOD_ORDER.index(m), m)
        except ValueError:
            return (len(_METHOD_ORDER), m)

    methods = sorted(method_metrics.keys(), key=_sort_key)
    labels  = [_METHOD_LABELS.get(m, m) for m in methods]

    metric_rows = [
        ("Precision",       "precision",            True),
        ("Recall",          "recall",               True),
        ("F1",              "f1",                   True),
        ("mAP@0.5",         "map_50",               True),
        ("mAP@0.75",        "map_75",               True),
        ("Angle error (°)", "mean_angle_error_deg", False),
    ]

    header = "| Metric | " + " | ".join(labels) + " |"
    sep    = "|--------|" + "|".join("-" * (len(l) + 2) for l in labels) + "|"
    lines  = [header, sep]

    for row_label, key, pct in metric_rows:
        vals = [_fmt(method_metrics[m].get(key, 0.0), pct) for m in methods]
        lines.append(f"| {row_label} | " + " | ".join(vals) + " |")

    for method in methods:
        m_label = _METHOD_LABELS.get(method, method)
        per_band = method_metrics[method].get("per_band", {})
        if not per_band:
            continue
        lines += [
            "",
            f"### Per-band — {m_label}",
            "| Band | Precision | Recall | F1 |",
            "|------|-----------|--------|----|",
        ]
        for band in ("short", "medium", "long"):
            b = per_band.get(band, {})
            lines.append(
                f"| {band.capitalize()} (<150 / 150–400 / >400 px)"
                f" | {_fmt(b.get('precision', 0))}"
                f" | {_fmt(b.get('recall', 0))}"
                f" | {_fmt(b.get('f1', 0))} |"
            )

    return "\n".join(lines)


def save_per_method_confusion_matrices(
    method_preds: dict[str, list[dict]],
    ground_truth: list[dict],
    output_dir: Path,
    iou_threshold: float = 0.5,
) -> dict[str, Path]:
    """Save a labelled confusion-matrix PNG for each method.

    Args:
        method_preds: Dict mapping method name → prediction list.
        ground_truth: Ground-truth annotation list.
        output_dir: Directory where PNGs are written.
        iou_threshold: IoU threshold forwarded to ``confusion_matrix()``.

    Returns:
        Dict mapping method name → saved PNG path.
    """
    from eval.metrics import confusion_matrix as _cm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for method, preds in method_preds.items():
        png_path = output_dir / f"confusion_matrix_{method}.png"
        label = _METHOD_LABELS.get(method, method)
        _cm(preds, ground_truth, iou_threshold=iou_threshold,
            save_path=png_path, title=label)
        paths[method] = png_path
        logger.info("Saved confusion matrix for %s → %s", method, png_path)

    return paths


def run_pipeline_predictions_all_methods(
    annotations_path: str | Path,
) -> dict[str, list[dict]]:
    """Run the multi-method pipeline on every image in a COCO annotation file.

    Loads the DINO model once, runs the full pipeline (DINO + classical)
    on each annotated image, then extracts per-method and
    unified prediction lists via ``extract_method_predictions()``.

    Args:
        annotations_path: Path to a COCO-format JSON annotation file.

    Returns:
        Dict mapping method name → list of prediction dicts for
        ``evaluate()``.  Always includes a ``"unified"`` key.

    Raises:
        FileNotFoundError: If model weights are not found.
        ImportError: If required ML packages are not installed.
    """
    from eval.metrics import extract_method_predictions

    with open(annotations_path) as f:
        coco = json.load(f)

    ann_path  = Path(annotations_path)
    image_dir = ann_path.parent.parent / "raw"

    from inference.pipeline import load_model, run as pipeline_run

    logger.info("Loading DINO model (once for all images)…")
    dino_model, dino_device = load_model()

    all_method_preds: dict[str, list[dict]] = {}

    for img_info in coco["images"]:
        fits_path = image_dir / img_info["file_name"]
        if not fits_path.exists():
            logger.warning("Image not found, skipping: %s", fits_path)
            continue
        logger.info("Running pipeline on %s", fits_path.name)
        grouped = pipeline_run(
            fits_path=fits_path,
            fast=True,
            model=dino_model,
            inference_device=dino_device,
        )
        per_method = extract_method_predictions(grouped, img_info["file_name"])
        for method, preds in per_method.items():
            all_method_preds.setdefault(method, []).extend(preds)

    return all_method_preds


def run_multi_method_benchmark(
    annotations_path: str | Path,
    method_predictions: dict[str, list[dict]] | None = None,
    output_path: str | Path = "results/multi_method_benchmark.json",
    run_pipeline: bool = False,
) -> dict:
    """Run a full multi-method benchmark and save results.

    Evaluates every method present in *method_predictions* (including
    ``"unified"``) and produces:
    - A Markdown comparison table printed to stdout.
    - One confusion-matrix PNG per method in ``<output_dir>/confusion_matrices/``.
    - A JSON results file at *output_path*.

    Args:
        annotations_path: Path to the COCO JSON annotation file (test split).
        method_predictions: Pre-computed per-method predictions.  When ``None``
            and *run_pipeline* is ``True``, the pipeline runs live.
        output_path: Destination for the JSON results file.
        run_pipeline: If ``True`` and *method_predictions* is ``None``, runs
            the multi-method inference pipeline on the test set (requires
            model weights).

    Returns:
        Results dict with a ``"methods"`` key mapping method name → metrics.

    Raises:
        ValueError: If neither *method_predictions* nor *run_pipeline* is
            provided.
        FileNotFoundError: If *run_pipeline* is True and weights are missing.
    """
    ground_truth = load_ground_truth(annotations_path)
    logger.info("Loaded %d ground-truth annotations", len(ground_truth))

    if method_predictions is None:
        if not run_pipeline:
            raise ValueError(
                "Provide method_predictions or set run_pipeline=True."
            )
        method_predictions = run_pipeline_predictions_all_methods(annotations_path)

    method_metrics: dict[str, dict] = {}
    for method, preds in method_predictions.items():
        logger.info("Evaluating %s  (%d predictions)…", method, len(preds))
        method_metrics[method] = evaluate(preds, ground_truth)

    output_path = Path(output_path)
    cm_dir = output_path.parent / "confusion_matrices"
    cm_paths = save_per_method_confusion_matrices(
        method_predictions, ground_truth, cm_dir
    )

    results = {
        "date_recorded": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methods": {
            method: {
                **metrics,
                "n_predictions": len(method_predictions[method]),
                "confusion_matrix_png": str(cm_paths.get(method, "")),
            }
            for method, metrics in method_metrics.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    print("\n" + format_comparison_table(method_metrics) + "\n")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(description="ARGUS Phase 8 benchmark")
    parser.add_argument("--annotations", required=True, help="COCO JSON annotation file")
    parser.add_argument("--dino-predictions", help="Pre-computed DINO predictions JSON")
    parser.add_argument("--method-predictions", help="Pre-computed multi-method predictions JSON "
                        "(dict mapping method → prediction list)")
    parser.add_argument("--multi-method", action="store_true",
                        help="Run multi-method benchmark (unified vs individual methods)")
    parser.add_argument("--run-pipeline", action="store_true", help="Run pipeline live (needs weights)")
    parser.add_argument("--output", default="results/phase8_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    if args.multi_method or args.method_predictions:
        method_preds = None
        if args.method_predictions:
            with open(args.method_predictions) as f:
                method_preds = json.load(f)
        multi_output = Path(args.output).parent / "multi_method_benchmark.json"
        run_multi_method_benchmark(
            annotations_path=args.annotations,
            method_predictions=method_preds,
            output_path=multi_output,
            run_pipeline=args.run_pipeline,
        )
    else:
        dino_preds = None
        if args.dino_predictions:
            with open(args.dino_predictions) as f:
                dino_preds = json.load(f)

        run_benchmark(
            annotations_path=args.annotations,
            dino_predictions=dino_preds,
            output_path=args.output,
            run_pipeline=args.run_pipeline,
        )
