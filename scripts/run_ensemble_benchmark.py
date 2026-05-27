"""scripts/run_ensemble_benchmark.py

Multi-model ensemble benchmark for ARGUS streak detection.

Runs the full pipeline (DINOv3 multisource + YOLO-OBB GTImages + ASTRiDE)
on one or more annotated test sets, then evaluates:
  - Each detector in isolation
  - The unified ensemble (with updated profiles + geometry fix + band weights)

Results are saved to results/ensemble_benchmark_<date>/ as JSON and Markdown.

Usage:
    # Standard test set only (default):
    python scripts/run_ensemble_benchmark.py

    # All three test sets:
    python scripts/run_ensemble_benchmark.py --all-sets

    # Specific annotation file:
    python scripts/run_ensemble_benchmark.py \\
        --annotations data/annotations/test.json --label satstreaks

Environment variables:
    MODEL_SIZE   (default: dinov3_vitb_multisource)
    MODEL_WEIGHTS
    ARGUS_NORM   (default: autostretch)
    STREAKMIND_YOLO_WEIGHTS  (default: weights/streakmind_yolo_combined/run/weights/best.pt)
    PYTORCH_ENABLE_MPS_FALLBACK=1   (set before running on Mac)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MODEL_SIZE", "dinov3_vitb_multisource")
os.environ.setdefault("ARGUS_NORM", "autostretch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault(
    "STREAKMIND_YOLO_WEIGHTS",
    str(Path(__file__).resolve().parent.parent
        / "weights" / "streakmind_yolo_combined" / "run" / "weights" / "best.pt"),
)
# Run streakmind_yolo as a single full-image pass (no tiling) to avoid
# 100+ tile evaluations per 4096×4096 image on Mac CPU.  At 640px → 4096px
# scale, medium streaks (269–800 px native) resolve to 42–125 px for YOLO,
# which is within its detection range.  Set this env var before import so the
# detector picks it up from the first call.
os.environ.setdefault("STREAKMIND_YOLO_TILE_SIZE", "8192")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ensemble_benchmark")


# ---------------------------------------------------------------------------
# Test-set registry
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_ANNOTATIONS_DIR = _ROOT / "data" / "annotations"
_RAW_DIR = _ROOT / "data"   # annotation file_names are relative to data/

TEST_SETS = {
    "satstreaks": {
        "annotations": _ANNOTATIONS_DIR / "test.json",
        "label": "SatStreaks test",
    },
    "brentimages": {
        "annotations": _ANNOTATIONS_DIR / "brentimages_20260515_eval.json",
        "label": "BrentImages Night 2 (zero-shot)",
    },
}

# Detectors whose predictions we report individually in the comparison table.
_SOLO_METHODS = [
    "dinov3_vitb_multisource",
    "streakmind_yolo",
    "astride",
    "unified",
]

# Human-readable labels
_METHOD_LABELS = {
    "dinov3_vitb_multisource": "DINOv3 Multisource",
    "streakmind_yolo":         "YOLO-OBB GTImages",
    "astride":                 "ASTRiDE",
    "unified":                 "Unified Ensemble (v2)",
}

# Band thresholds matching eval/metrics.py
_SHORT_MAX = 150.0
_LONG_MIN = 400.0


# ---------------------------------------------------------------------------
# Flat pipeline output → grouped format
# ---------------------------------------------------------------------------

def aggregate_flat_detections(flat_dets: list[dict]) -> list[dict]:
    """Convert flat pipeline output to the grouped format for extract_method_predictions.

    The pipeline returns one dict per (streak_id, method) pair.  This function
    collapses same-streak_id detections into one record per streak with a
    ``sources`` list, preserving the fused OBB and streak_length_px from the
    highest-confidence detection.

    Args:
        flat_dets: List of detection dicts from inference.pipeline.run().

    Returns:
        List of streak-group dicts with ``obb``, ``streak_length_px``, and
        ``sources`` keys, compatible with eval.metrics.extract_method_predictions.
    """
    groups: dict[object, list[dict]] = {}
    for det in flat_dets:
        sid = det.get("streak_id", id(det))
        groups.setdefault(sid, []).append(det)

    result = []
    for dets in groups.values():
        primary = max(dets, key=lambda d: d.get("confidence", 0.0))
        sources = [
            {"method": d["method"], "confidence": d["confidence"]}
            for d in dets
            if d.get("method") and d["method"] != "unified"
        ]
        result.append({
            "streak_id": primary.get("streak_id"),
            "obb": primary.get("obb"),
            "streak_length_px": primary.get("streak_length_px", 0.0),
            "sources": sources,
        })
    return result


# ---------------------------------------------------------------------------
# Prediction extraction helpers
# ---------------------------------------------------------------------------

def extract_per_method(flat_dets: list[dict], image_id: str) -> dict[str, list[dict]]:
    """Extract per-method and unified prediction lists from flat pipeline output.

    Args:
        flat_dets: List of detection dicts from the pipeline.
        image_id: FITS filename to stamp on each prediction.

    Returns:
        Dict mapping method name → list of prediction dicts for evaluate().
        Always includes a "unified" key.
    """
    from eval.metrics import extract_method_predictions
    grouped = aggregate_flat_detections(flat_dets)
    return extract_method_predictions(grouped, image_id)


# ---------------------------------------------------------------------------
# Ground-truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(annotations_path: Path) -> list[dict]:
    """Load COCO annotations into the flat format expected by evaluate().

    Reads the ``obb`` field directly from each annotation (preferred), which
    stores [cx, cy, w, h, angle_deg] or a dict with those keys — the same
    tight oriented bounding box used during labelling.  Falls back to the
    axis-aligned ``bbox`` field only when ``obb`` is absent.

    Args:
        annotations_path: Path to a COCO-format JSON annotation file.

    Returns:
        List of dicts with image_id, obb, streak_length_px.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    gt = []
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue

        obb_raw = ann.get("obb")
        if obb_raw:
            if isinstance(obb_raw, dict):
                try:
                    cx = float(obb_raw["cx"])
                    cy = float(obb_raw["cy"])
                    w  = float(obb_raw["w"])
                    h  = float(obb_raw["h"])
                    angle_deg = float(obb_raw["angle_deg"])
                except (KeyError, TypeError, ValueError):
                    continue
            else:
                if len(obb_raw) < 5:
                    continue
                cx, cy, w, h, angle_deg = [float(v) for v in obb_raw[:5]]
        else:
            # Fallback: reconstruct from axis-aligned bbox
            bbox = ann.get("bbox")
            if not bbox:
                continue
            import math
            bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            cx, cy = bx + bw / 2, by + bh / 2
            w, h = max(bw, bh), min(bw, bh)
            angle_deg = math.degrees(math.atan2(bh, bw)) % 180.0

        gt.append({
            "image_id": id_to_filename[ann["image_id"]],
            "obb": {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle_deg},
            "streak_length_px": max(w, h),
        })
    return gt


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark_on_set(
    annotations_path: Path,
    label: str,
    models: list,
    device: object,
) -> dict:
    """Run all detectors on one annotated test set and evaluate results.

    Args:
        annotations_path: COCO annotation JSON path.
        label: Human-readable name for this test set.
        models: List of (model, inference_device, spec) from load_models().
        device: Inference device.

    Returns:
        Dict with keys per method → metrics dict, plus metadata.
    """
    from eval.metrics import evaluate

    logger.info("=== %s ===", label)
    logger.info("Annotations: %s", annotations_path)

    with open(annotations_path) as f:
        coco = json.load(f)

    all_method_preds: dict[str, list[dict]] = defaultdict(list)

    n_images = len(coco["images"])
    n_found = 0
    n_skipped = 0

    for idx, img_info in enumerate(coco["images"], 1):
        fits_path = _RAW_DIR / img_info["file_name"]
        if not fits_path.exists():
            logger.debug("Missing %s — skipping", fits_path.name)
            n_skipped += 1
            continue

        n_found += 1
        if idx % 20 == 0 or idx == n_images:
            logger.info(
                "  [%d/%d] processing %s  (skipped=%d)",
                idx, n_images, fits_path.name, n_skipped,
            )

        try:
            from inference.pipeline import run_with_array
            flat_dets, _ = run_with_array(
                fits_path,
                fast=True,
                models=models,
                # yolo_full excluded: ~34 s/image on Mac CPU (144 tiles per 4096×4096 image).
                # streakmind_yolo runs as single-pass (STREAKMIND_YOLO_TILE_SIZE=8192).
                enabled_detectors={"dinov3_vitb_multisource", "streakmind_yolo", "astride"},
            )
        except Exception as exc:
            logger.warning("Pipeline failed on %s: %s", fits_path.name, exc)
            continue

        per_method = extract_per_method(flat_dets, img_info["file_name"])
        for method, preds in per_method.items():
            all_method_preds[method].extend(preds)

    logger.info(
        "Processed %d/%d images (%d skipped — FITS not on disk)",
        n_found, n_images, n_skipped,
    )

    if n_found == 0:
        logger.warning("No images found for %s — check data/raw/", label)
        return {"label": label, "n_images": 0, "n_skipped": n_skipped, "methods": {}}

    gt = load_ground_truth(annotations_path)

    method_metrics: dict[str, dict] = {}
    for method in _SOLO_METHODS:
        preds = all_method_preds.get(method, [])
        if not preds:
            logger.warning("  No predictions from method '%s'", method)
            method_metrics[method] = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "map_50": 0.0, "map_75": 0.0,
                "mean_angle_error_deg": 0.0,
                "n_predictions": 0,
                "per_band": {"short": {}, "medium": {}, "long": {}},
            }
            continue
        # mAP requires the full unfiltered P-R curve; pass all predictions.
        # P/R/F1 at deployment threshold (conf≥0.30) are reported separately.
        _CONF_THRESH = 0.30
        metrics = evaluate(preds, gt, iou_threshold=0.5)
        preds_thresh = [p for p in preds if p.get("confidence", 0.0) >= _CONF_THRESH]
        metrics_thresh = evaluate(preds_thresh, gt, iou_threshold=0.5)
        method_metrics[method] = {
            **metrics,
            "n_predictions": len(preds),
            # Deployment-threshold P/R/F1 stored separately
            "precision_30": metrics_thresh["precision"],
            "recall_30": metrics_thresh["recall"],
            "f1_30": metrics_thresh["f1"],
            "n_predictions_30": len(preds_thresh),
        }
        logger.info(
            "  %-28s  mAP@50=%.3f  P@30=%.3f  R@30=%.3f  F1@30=%.3f  n=%d (raw=%d)",
            _METHOD_LABELS.get(method, method),
            metrics["map_50"],
            metrics_thresh["precision"],
            metrics_thresh["recall"],
            metrics_thresh["f1"],
            len(preds_thresh),
            len(preds),
        )

    return {
        "label": label,
        "n_images": n_found,
        "n_skipped": n_skipped,
        "methods": method_metrics,
        "all_predictions": {m: v for m, v in all_method_preds.items()},
    }


# ---------------------------------------------------------------------------
# Markdown table formatter
# ---------------------------------------------------------------------------

def _fmt(v: float, pct: bool = True) -> str:
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def format_comparison_table(set_label: str, method_metrics: dict[str, dict]) -> str:
    """Render a comparison table for one test set as Markdown.

    Args:
        set_label: Name of the test set.
        method_metrics: Dict of method → metrics dict.

    Returns:
        Markdown string.
    """
    ordered = [m for m in _SOLO_METHODS if m in method_metrics]
    labels = [_METHOD_LABELS.get(m, m) for m in ordered]

    header = f"\n### {set_label}\n"
    col_header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|--------|" + "|".join("-" * (len(l) + 2) for l in labels) + "|"
    lines = [header, col_header, sep]

    rows = [
        ("mAP@0.50",        "map_50",           False),
        ("mAP@0.75",        "map_75",           False),
        ("P @conf≥0.30",    "precision_30",     True),
        ("R @conf≥0.30",    "recall_30",        True),
        ("F1 @conf≥0.30",   "f1_30",            True),
        ("N preds (raw)",   "n_predictions",    False),
        ("N preds @0.30",   "n_predictions_30", False),
    ]
    for row_label, key, pct in rows:
        vals = []
        for m in ordered:
            v = method_metrics[m].get(key, 0.0)
            vals.append(_fmt(v, pct) if pct else (f"{int(v)}" if key == "n_predictions" else f"{v:.3f}"))
        lines.append(f"| {row_label} | " + " | ".join(vals) + " |")

    # Per-band recall rows
    for band in ("short", "medium", "long"):
        vals = []
        for m in ordered:
            pb = method_metrics[m].get("per_band", {}).get(band, {})
            r = pb.get("recall", 0.0)
            vals.append(f"{r*100:.1f}%")
        lines.append(f"| Recall {band} | " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-model ensemble benchmark on ARGUS test sets."
    )
    parser.add_argument(
        "--all-sets", action="store_true",
        help="Run on all test sets (satstreaks, brentimages)",
    )
    parser.add_argument(
        "--sets", nargs="+",
        choices=list(TEST_SETS.keys()),
        default=["satstreaks"],
        help="Which test sets to run (default: satstreaks)",
    )
    parser.add_argument(
        "--annotations", type=Path,
        help="Custom annotation file path (overrides --sets)",
    )
    parser.add_argument(
        "--label", default="custom",
        help="Label for custom annotation set",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=_ROOT / "results" / f"ensemble_benchmark_{datetime.now().strftime('%Y%m%d')}",
        help="Directory to save results",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which sets to run
    if args.annotations:
        sets_to_run = [{"annotations": args.annotations, "label": args.label, "key": "custom"}]
    elif args.all_sets:
        sets_to_run = [
            {"annotations": v["annotations"], "label": v["label"], "key": k}
            for k, v in TEST_SETS.items()
        ]
    else:
        sets_to_run = [
            {"annotations": TEST_SETS[k]["annotations"], "label": TEST_SETS[k]["label"], "key": k}
            for k in args.sets
        ]

    # Load models once
    logger.info("Loading models…")
    t_load = time.perf_counter()
    from inference.pipeline import load_models
    models = load_models()
    logger.info("Models loaded in %.1fs", time.perf_counter() - t_load)

    from inference.device import get_device
    device = get_device()

    all_results: dict[str, dict] = {}
    md_sections: list[str] = []

    for spec in sets_to_run:
        t0 = time.perf_counter()
        result = run_benchmark_on_set(
            annotations_path=spec["annotations"],
            label=spec["label"],
            models=models,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Finished '%s' in %.1fs", spec["label"], elapsed)

        all_results[spec["key"]] = result

        if result["n_images"] > 0:
            md_sections.append(
                format_comparison_table(spec["label"], result["methods"])
            )

    # Save raw JSON results (without predictions to keep file size sane)
    summary = {
        "date_recorded": datetime.now(tz=timezone.utc).isoformat(),
        "model_size": os.environ.get("MODEL_SIZE"),
        "streakmind_yolo_weights": os.environ.get("STREAKMIND_YOLO_WEIGHTS"),
        "results": {
            k: {
                "label": v["label"],
                "n_images": v.get("n_images", 0),
                "n_skipped": v.get("n_skipped", 0),
                "methods": v.get("methods", {}),
            }
            for k, v in all_results.items()
        },
    }
    out_json = args.output_dir / "ensemble_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info("Saved JSON → %s", out_json)

    # Save Markdown report
    md_header = f"""# ARGUS Multi-Model Ensemble Benchmark

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**DINO model:** `{os.environ.get("MODEL_SIZE")}`
**YOLO-OBB GTImages weights:** `{Path(os.environ.get("STREAKMIND_YOLO_WEIGHTS", "")).name}`

## Summary

This benchmark evaluates four detectors independently and as a unified ensemble:

| Detector | Role |
|---|---|
| **DINOv3 Multisource** | Primary ML detector — high recall on long streaks, loose axis-aligned boxes |
| **YOLO-OBB GTImages** | Segment detector — tight OBBs, dominant recall on medium-length streaks (single-pass, TILE_SIZE=8192) |
| **ASTRiDE** | Classical detector — many false positives, corroboration signal only |
| **Unified Ensemble v2** | Updated profiles + YOLO geometry preference + per-band weights |

Confidence threshold for P/R evaluation: **0.30**. IoU threshold: **0.50**.
Band thresholds: short < 150 px, 150 ≤ medium < 400 px, long ≥ 400 px.
"""

    md_body = md_header + "\n".join(md_sections)
    out_md = args.output_dir / "ensemble_benchmark.md"
    with open(out_md, "w") as f:
        f.write(md_body)
    logger.info("Saved Markdown → %s", out_md)

    # Print to stdout
    print("\n" + "=" * 72)
    print(md_body)
    print("=" * 72)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
