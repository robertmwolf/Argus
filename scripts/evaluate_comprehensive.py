"""Comprehensive evaluation of the best DINOv3 checkpoint across all held-out sets.

Runs three evaluations:
  1. test_standard      — standard COCO test split (308 satstreaks)
  2. frigate            — zero-shot, Frigate frames (350 images, absolute paths)
  3. brentimages_night2 — zero-shot, BrentImages Night 2 (231 images)

For each set reports:
  - COCO mAP / mAP@50 / mAP@75 (via MMDetection CocoMetric)
  - Precision / Recall / F1 at conf>=0.3, IoU>=0.5 (greedy matching)
  - Per-band recall: short (<269 px diagonal), medium (269–800 px), long (>800 px)

Band threshold definition
-------------------------
SHORT_MAX and LONG_MIN are in **pixels, measured in the original image coordinate
space** — i.e. the diagonal of the COCO [x,y,w,h] bounding box before any
model-input downscaling.  Both GT annotations and model predictions are rescaled
back to original-image coordinates by MMDetection before band classification.

These are NOT arcsecond thresholds.  The pixel values have different angular
meanings for each source:
  - SatStreaks (4096×4096, HST ~0.05 arcsec/px):  269px ≈ 13 arcsec
  - Atwood     (6248×4176, ZWO @ 1.27 arcsec/px): 269px ≈ 342 arcsec (5.7 arcmin)
  - Frigate    (2325×1555, pixel scale unknown)

The thresholds are therefore a detection-difficulty proxy tied to the original
image resolution, not a physically invariant size classification.  They were
calibrated on the SatStreaks test set (308 images, all 4096×4096).  When
comparing results across sources with different resolutions, interpret band
numbers with this in mind.

Outputs:
  results/comprehensive_eval_YYYYMMDD_HHMMSS/
    {set}/metrics.json
    all_results.json
    report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = _REPO_ROOT / "weights/run_clean_vitb_nodm/best_coco_bbox_mAP_epoch_15.pth"
CONFIG = _REPO_ROOT / "models/dino/streak_dinov3_vitb.py"

# Band thresholds — pixels in ORIGINAL IMAGE coordinate space (not model-input space,
# not arcseconds).  See module docstring for angular equivalents per source.
SHORT_MAX = 269.0   # px diagonal; below this → "short"  (at Atwood: < 342 arcsec)
LONG_MIN  = 800.0   # px diagonal; above this → "long"   (at Atwood: > 1016 arcsec)

BRENT_RAW_DIR = "/Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood"

EVAL_SETS = [
    {
        "key": "test_standard",
        "split": "test",
        "label": "Standard test (satstreaks)",
    },
    {
        "key": "frigate",
        "split": "frigate_streaks",
        "label": "Frigate (zero-shot)",
    },
    {
        "key": "brentimages_night2",
        "split": "brentimages_20260515_eval",
        "label": "BrentImages Night 2 (zero-shot)",
    },
]


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def build_brentimages_eval_json() -> Path:
    """Build a BrentImages Night 2 eval file from reviewed labels.

    Bare filenames in the source annotations are resolved to absolute paths.
    Archived negative-only candidate files are intentionally not used here.
    """
    ann_dir = _REPO_ROOT / "data/annotations"
    out_path = ann_dir / "brentimages_20260515_eval.json"

    streaks_path = ann_dir / "brentimages_20260515.json"

    with open(streaks_path) as f:
        streaks = json.load(f)

    old_to_new_streaks: dict[int, int] = {}
    all_images = []
    next_id = 1

    for img in streaks["images"]:
        fname = img["file_name"]
        if not fname.startswith("/"):
            fname = f"{BRENT_RAW_DIR}/{fname}"
        new_img = dict(img)
        new_img["file_name"] = fname
        new_img["id"] = next_id
        old_to_new_streaks[img["id"]] = next_id
        all_images.append(new_img)
        next_id += 1

    all_annotations = []
    next_ann_id = 1
    for ann in streaks.get("annotations", []):
        new_ann = dict(ann)
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = old_to_new_streaks[ann["image_id"]]
        all_annotations.append(new_ann)
        next_ann_id += 1

    # Use the canonical category definition matching the training data (test.json)
    canonical_categories = [{"id": 1, "name": "streak", "supercategory": "satellite"}]
    merged = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": canonical_categories,
    }

    with open(out_path, "w") as f:
        json.dump(merged, f)

    logger.info("Built %s: %d images, %d annotations", out_path.name, len(all_images), len(all_annotations))
    return out_path


def normalize_categories(ann_path: Path) -> Path:
    """Return a path to a version of the annotation with canonical category def.

    If the categories already match, returns the original path unchanged.
    Otherwise writes a fixed copy with suffix _eval.json and returns that.
    """
    canonical = [{"id": 1, "name": "streak", "supercategory": "satellite"}]
    with open(ann_path) as f:
        d = json.load(f)
    if d.get("categories") == canonical:
        return ann_path
    out_path = ann_path.with_stem(ann_path.stem + "_eval")
    if out_path.exists():
        return out_path
    d["categories"] = canonical
    with open(out_path, "w") as f:
        json.dump(d, f)
    logger.info("Wrote normalized categories to %s", out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# MMDetection evaluation
# ---------------------------------------------------------------------------

def run_mmdet_eval(split: str, checkpoint: Path, work_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.evaluate_dino_checkpoint import evaluate_checkpoint

    metrics = evaluate_checkpoint(
        config_path=CONFIG,
        checkpoint_path=checkpoint,
        split=split,
        work_dir=work_dir,
        output_path=work_dir / "coco_metrics.json",
    )
    return metrics


def find_predictions_file(work_dir: Path, checkpoint: Path, split: str) -> Path:
    """Locate the COCO-format predictions bbox JSON written by MMDetection."""
    # MMDetection writes: {outfile_prefix}.bbox.json
    # outfile_prefix = work_dir / f"{checkpoint.stem}_{split}"
    candidate = work_dir / f"{checkpoint.stem}_{split}.bbox.json"
    if candidate.exists():
        return candidate
    # Fallback: glob
    matches = list(work_dir.glob("*.bbox.json"))
    if matches:
        return sorted(matches)[-1]
    raise FileNotFoundError(f"No bbox predictions found in {work_dir}")


# ---------------------------------------------------------------------------
# Custom metric helpers
# ---------------------------------------------------------------------------

def _bbox_diag(bbox: list[float]) -> float:
    """Diagonal of a COCO [x, y, w, h] bbox."""
    _, _, w, h = bbox
    return math.sqrt(w * w + h * h)


def _band(bbox: list[float]) -> str:
    d = _bbox_diag(bbox)
    if d < SHORT_MAX:
        return "short"
    if d < LONG_MIN:
        return "medium"
    return "long"


def _bbox_iou(b1: list[float], b2: list[float]) -> float:
    """IoU for two COCO [x, y, w, h] boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def compute_pr(
    gt_coco: dict,
    pred_coco: list[dict],
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedy precision/recall at a fixed confidence and IoU threshold."""
    # Build GT lookup: image_id → list of bbox
    gt_by_image: dict[int, list[list[float]]] = {}
    for ann in gt_coco.get("annotations", []):
        gt_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

    # Filter and sort predictions by score descending
    preds = [p for p in pred_coco if p["score"] >= conf_threshold]
    preds.sort(key=lambda p: p["score"], reverse=True)

    matched: dict[int, list[bool]] = {iid: [False] * len(bboxes) for iid, bboxes in gt_by_image.items()}
    tp = fp = 0
    n_gt = sum(len(v) for v in gt_by_image.values())

    for pred in preds:
        iid = pred["image_id"]
        gt_boxes = gt_by_image.get(iid, [])
        matched_flags = matched.get(iid, [])
        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt_boxes):
            if matched_flags[j]:
                continue
            iou = _bbox_iou(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            matched_flags[best_j] = True
            tp += 1
        else:
            fp += 1

    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt,
        "n_pred_above_conf": len(preds),
    }


def compute_per_band_recall(
    gt_coco: dict,
    pred_coco: list[dict],
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Per-band recall using project standard length bands."""
    # Group GT by band
    gt_by_band: dict[str, dict[int, list[tuple[int, list[float]]]]] = {
        "short": {}, "medium": {}, "long": {}
    }
    ann_to_band: dict[int, str] = {}
    for ann in gt_coco.get("annotations", []):
        b = _band(ann["bbox"])
        ann_to_band[ann["id"]] = b
        gt_by_band[b].setdefault(ann["image_id"], []).append((ann["id"], ann["bbox"]))

    preds = [p for p in pred_coco if p["score"] >= conf_threshold]
    preds.sort(key=lambda p: p["score"], reverse=True)

    # matched_ann: ann_id → bool
    matched_ann: dict[int, bool] = {ann["id"]: False for ann in gt_coco.get("annotations", [])}

    for pred in preds:
        iid = pred["image_id"]
        best_iou = 0.0
        best_ann_id = -1
        for b in ("short", "medium", "long"):
            for ann_id, gt_box in gt_by_band[b].get(iid, []):
                if matched_ann.get(ann_id, False):
                    continue
                iou = _bbox_iou(pred["bbox"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_ann_id = ann_id
        if best_iou >= iou_threshold and best_ann_id >= 0:
            matched_ann[best_ann_id] = True

    results: dict[str, Any] = {}
    for b in ("short", "medium", "long"):
        all_ids = [ann_id for anns in gt_by_band[b].values() for ann_id, _ in anns]
        n_gt = len(all_ids)
        tp = sum(1 for ann_id in all_ids if matched_ann.get(ann_id, False))
        fn = n_gt - tp
        recall = tp / n_gt if n_gt > 0 else None
        results[b] = {
            "recall": round(recall, 4) if recall is not None else None,
            "tp": tp,
            "fn": fn,
            "n_gt": n_gt,
        }
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:.3f}"


def write_report(all_results: dict, out_dir: Path) -> None:
    lines = [
        "# Comprehensive Evaluation Report",
        f"",
        f"**Model:** DINOv3 ViT-B Multi-source (clean cold-start)  ",
        f"**Checkpoint:** `{CHECKPOINT}`  ",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Confidence threshold (P/R/band):** 0.30 | IoU threshold: 0.50  ",
        "",
        "## Summary",
        "",
        "| Set | mAP | mAP@50 | mAP@75 | Precision | Recall | F1 |",
        "|-----|-----|--------|--------|-----------|--------|----|",
    ]

    for key, res in all_results.items():
        label = res.get("label", key)
        coco = res.get("coco_metrics", {})
        pr = res.get("pr", {})
        mAP = _fmt(coco.get("coco/bbox_mAP"))
        mAP50 = _fmt(coco.get("coco/bbox_mAP_50"))
        mAP75 = _fmt(coco.get("coco/bbox_mAP_75"))
        prec = _fmt(pr.get("precision"), pct=True)
        rec = _fmt(pr.get("recall"), pct=True)
        f1 = _fmt(pr.get("f1"), pct=True)
        lines.append(f"| {label} | {mAP} | {mAP50} | {mAP75} | {prec} | {rec} | {f1} |")

    lines += [
        "",
        "## Per-Band Recall (conf ≥ 0.30, IoU ≥ 0.50)",
        "",
        "Bands: short < 269 px diagonal, medium 269–800 px, long > 800 px",
        "",
        "| Set | Short recall | Short n | Medium recall | Medium n | Long recall | Long n |",
        "|-----|-------------|---------|--------------|----------|-------------|--------|",
    ]

    for key, res in all_results.items():
        label = res.get("label", key)
        bands = res.get("per_band_recall", {})
        short = bands.get("short", {})
        med = bands.get("medium", {})
        lng = bands.get("long", {})

        def fmt_band(b: dict) -> tuple[str, str]:
            r = b.get("recall")
            n = b.get("n_gt", 0)
            return _fmt(r, pct=True), str(n)

        sr, sn = fmt_band(short)
        mr, mn = fmt_band(med)
        lr, ln = fmt_band(lng)
        lines.append(f"| {label} | {sr} | {sn} | {mr} | {mn} | {lr} | {ln} |")

    lines += [
        "",
        "## Detailed Results",
        "",
    ]

    for key, res in all_results.items():
        label = res.get("label", key)
        lines += [f"### {label}", ""]
        coco = res.get("coco_metrics", {})
        lines += [
            f"**COCO metrics:**",
            f"- mAP: {_fmt(coco.get('coco/bbox_mAP'))}",
            f"- mAP@50: {_fmt(coco.get('coco/bbox_mAP_50'))}",
            f"- mAP@75: {_fmt(coco.get('coco/bbox_mAP_75'))}",
            f"- mAP_s: {_fmt(coco.get('coco/bbox_mAP_s'))}",
            f"- mAP_m: {_fmt(coco.get('coco/bbox_mAP_m'))}",
            f"- mAP_l: {_fmt(coco.get('coco/bbox_mAP_l'))}",
            "",
        ]
        pr = res.get("pr", {})
        lines += [
            f"**P/R @ conf≥0.30:**",
            f"- Precision: {_fmt(pr.get('precision'), pct=True)}",
            f"- Recall: {_fmt(pr.get('recall'), pct=True)}",
            f"- F1: {_fmt(pr.get('f1'), pct=True)}",
            f"- TP: {pr.get('tp', '—')}  FP: {pr.get('fp', '—')}  FN: {pr.get('fn', '—')}",
            f"- GT annotations: {pr.get('n_gt', '—')}  Predictions above conf: {pr.get('n_pred_above_conf', '—')}",
            "",
        ]
        bands = res.get("per_band_recall", {})
        lines += ["**Per-band recall:**"]
        for b in ("short", "medium", "long"):
            bd = bands.get(b, {})
            r = _fmt(bd.get("recall"), pct=True)
            n = bd.get("n_gt", 0)
            tp = bd.get("tp", 0)
            fn = bd.get("fn", 0)
            lines.append(f"- {b.capitalize()}: {r}  (TP={tp}, FN={fn}, n={n})")
        lines += ["", "---", ""]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for P/R")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=None,
                        help="Override MMDetection config path (default: hardcoded CONFIG)")
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=[s["key"] for s in EVAL_SETS] + ["all"],
        default=["all"],
    )
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)
    sys.path.insert(0, str(_REPO_ROOT))

    # Allow config override (e.g. for run3 400px vs default 256px)
    if args.config is not None:
        global CONFIG
        CONFIG = args.config

    run_sets = EVAL_SETS if "all" in args.sets else [s for s in EVAL_SETS if s["key"] in args.sets]

    # Build brentimages eval JSON if needed
    if any(s["key"] == "brentimages_night2" for s in run_sets):
        brent_eval = build_brentimages_eval_json()
        logger.info("BrentImages eval JSON ready: %s", brent_eval)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = _REPO_ROOT / f"results/comprehensive_eval_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}

    for eval_set in run_sets:
        key = eval_set["key"]
        split = eval_set["split"]
        label = eval_set["label"]
        logger.info("=" * 60)
        logger.info("Evaluating: %s (%s)", label, split)
        logger.info("=" * 60)

        work_dir = out_root / key
        work_dir.mkdir(parents=True, exist_ok=True)

        # Check annotation file exists
        ann_file = _REPO_ROOT / "data/annotations" / f"{split}.json"
        if not ann_file.exists():
            logger.warning("Annotation file missing: %s — skipping", ann_file)
            all_results[key] = {"label": label, "error": f"missing {ann_file}"}
            continue

        # Normalize categories to match training data definition
        normalized_ann = normalize_categories(ann_file)
        eval_split = normalized_ann.stem  # may differ if _eval suffix was added

        try:
            coco_metrics = run_mmdet_eval(eval_split, args.checkpoint, work_dir)
        except Exception as exc:
            logger.error("MMDetection eval failed for %s: %s\n%s", key, exc, traceback.format_exc())
            all_results[key] = {"label": label, "error": str(exc)}
            continue

        # Load GT and predictions for custom metrics
        with open(ann_file) as f:
            gt_coco = json.load(f)

        try:
            pred_file = find_predictions_file(work_dir, args.checkpoint, eval_split)
            with open(pred_file) as f:
                pred_coco = json.load(f)
            logger.info("Loaded %d predictions from %s", len(pred_coco), pred_file.name)
        except FileNotFoundError as exc:
            logger.warning("Predictions file not found: %s — custom metrics skipped", exc)
            pred_coco = []

        pr = compute_pr(gt_coco, pred_coco, conf_threshold=args.conf)
        per_band = compute_per_band_recall(gt_coco, pred_coco, conf_threshold=args.conf)

        set_results = {
            "label": label,
            "split": split,
            "coco_metrics": coco_metrics,
            "pr": pr,
            "per_band_recall": per_band,
        }
        all_results[key] = set_results

        # Write per-set metrics
        metrics_path = work_dir / "metrics.json"
        metrics_path.write_text(json.dumps(set_results, indent=2))
        logger.info(
            "%s — mAP@50=%.3f  P=%.3f  R=%.3f  F1=%.3f",
            key,
            coco_metrics.get("coco/bbox_mAP_50", 0),
            pr["precision"],
            pr["recall"],
            pr["f1"],
        )

    # Write combined output
    combined_path = out_root / "all_results.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    write_report(all_results, out_root)

    logger.info("=" * 60)
    logger.info("All results saved to %s", out_root)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
