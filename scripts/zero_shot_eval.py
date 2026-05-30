"""Zero-shot evaluation of the current model on a new scope's first night.

Run this BEFORE annotating Night 2 from a new telescope to decide whether
fine-tuning is needed.  Evaluates the current production checkpoint against
a held-out annotation file and reports per-band recall so you can see
exactly where the model is succeeding or failing on the new domain.

Decision thresholds (recommendations — adjust based on your requirements):
  Long-band recall  ≥ 80%   →  No fine-tuning needed; fold into next retrain
  Long-band recall  60–80%  →  Fine-tune advised (scripts/train_dino.py with
                                streak_dinov3_vitb_400px_ft.py config)
  Long-band recall  < 60%   →  Significant domain shift; investigate pixel scale,
                                normalisation, anchor coverage before fine-tuning

Outputs
-------
  results/zero_shot_{scope}_{YYYYMMDD_HHMMSS}/
    metrics.json    — structured results (machine-readable)
    report.md       — human-readable summary with decision recommendation

Usage
-----
  # Evaluate current model against a new scope's Night 1 annotation:
  python scripts/zero_shot_eval.py \\
      --annotation data/annotations/newscope_20260601.json \\
      --raw-dir /Volumes/External/newscope/Img_20260601 \\
      --scope newscope \\
      --label "New Scope Night 1 (zero-shot)"

  # With an explicit checkpoint and config override:
  python scripts/zero_shot_eval.py \\
      --annotation data/annotations/newscope_20260601.json \\
      --raw-dir /Volumes/External/newscope/Img_20260601 \\
      --scope newscope \\
      --checkpoint weights/run3_cold_nodm/best.pth \\
      --config models/dino/streak_dinov3_vitb_400px_run3.py

  # Multiple annotation files (streaks + negatives — negatives need no annotation):
  python scripts/zero_shot_eval.py \\
      --annotation data/annotations/newscope_20260601.json \\
      --negatives data/annotations/newscope_20260601_negatives.json \\
      --raw-dir /Volumes/External/newscope/Img_20260601 \\
      --scope newscope
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

# Defaults — override with --checkpoint / --config as needed
_DEFAULT_CHECKPOINT = _REPO_ROOT / "weights/run3_cold_nodm/best.pth"
_DEFAULT_CONFIG = _REPO_ROOT / "models/dino/streak_dinov3_vitb_400px_run3.py"

# Band thresholds — pixels in ORIGINAL IMAGE coordinate space (not model-input
# space, not arcseconds).  Both GT and predictions are compared in original-image
# coords after MMDetection's keep_ratio rescale.
#
# These values have different angular meanings per source:
#   Atwood (1.27 arcsec/px): 269px ≈ 342 arcsec (5.7'), 800px ≈ 1016 arcsec (16.9')
#   SatStreaks (HST, ~0.05 arcsec/px): 269px ≈ 13 arcsec
#   Other sources: depends on sensor pixel scale and optics
#
# The thresholds are a detection-difficulty proxy, NOT a physical size classification.
SHORT_MAX = 269.0   # px diagonal; below → "short"
LONG_MIN  = 800.0   # px diagonal; above → "long"

# Run 3 baseline metrics on the standard test set — used for the decision report
_RUN3_BASELINE = {
    "precision": 0.9485,
    "recall":    0.8377,
    "f1":        0.8897,
    "mAP":       0.782,
    "mAP_50":    0.878,
    "band_recall": {
        "short":  1.000,
        "medium": 0.909,
        "long":   0.834,
    },
    "source": "results/comprehensive_eval_20260528_154914/test_standard/metrics.json",
}

# Decision thresholds for the fine-tune recommendation
_THRESH_OK       = 0.80   # long recall ≥ this → fine-tune optional
_THRESH_FINETUNE = 0.60   # long recall ≥ this → fine-tune advised
                          # long recall < this  → investigate domain shift


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fix_bare_paths(images: list[dict], raw_dir: str) -> list[dict]:
    fixed = []
    for img in images:
        new = dict(img)
        if not img["file_name"].startswith("/"):
            new["file_name"] = f"{raw_dir}/{img['file_name']}"
        fixed.append(new)
    return fixed


def _normalize_categories(coco: dict) -> dict:
    """Replace categories with the canonical single-class streak definition."""
    canonical = [{"id": 1, "name": "streak", "supercategory": "satellite"}]
    return dict(coco, categories=canonical)


def build_eval_coco(
    annotation_path: Path,
    negatives_path: Path | None,
    raw_dir: str | None,
) -> dict:
    """Load and merge annotation + optional negatives into a single COCO dict."""
    data = _load_json(annotation_path)
    if raw_dir:
        data["images"] = _fix_bare_paths(data["images"], raw_dir)
    data = _normalize_categories(data)

    if negatives_path and negatives_path.exists():
        neg = _load_json(negatives_path)
        if raw_dir:
            neg["images"] = _fix_bare_paths(neg["images"], raw_dir)
        # Negatives have no annotations; just append images
        max_id = max((img["id"] for img in data["images"]), default=0)
        neg_images = []
        for img in neg["images"]:
            new_img = dict(img)
            new_img["id"] = max_id + img["id"]  # offset to avoid collisions
            neg_images.append(new_img)
        data = dict(data, images=list(data["images"]) + neg_images)

    return data


def write_tmp_coco(coco: dict, path: Path) -> None:
    """Write a COCO dict to disk with sequential IDs (required by MMDetection)."""
    all_images = []
    all_annotations = []
    old_to_new: dict[int, int] = {}
    next_img_id = 1
    next_ann_id = 1

    for img in coco["images"]:
        new_img = dict(img)
        old_to_new[img["id"]] = next_img_id
        new_img["id"] = next_img_id
        all_images.append(new_img)
        next_img_id += 1

    for ann in coco.get("annotations", []):
        if ann["image_id"] not in old_to_new:
            continue
        new_ann = dict(ann)
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = old_to_new[ann["image_id"]]
        all_annotations.append(new_ann)
        next_ann_id += 1

    out = dict(coco, images=all_images, annotations=all_annotations)
    with open(path, "w") as f:
        json.dump(out, f)


# ---------------------------------------------------------------------------
# Metric helpers (shared with evaluate_comprehensive.py)
# ---------------------------------------------------------------------------

def _bbox_diag(bbox: list[float]) -> float:
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
    gt_by_image: dict[int, list[list[float]]] = {}
    for ann in gt_coco.get("annotations", []):
        gt_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

    preds = [p for p in pred_coco if p["score"] >= conf_threshold]
    preds.sort(key=lambda p: p["score"], reverse=True)

    matched: dict[int, list[bool]] = {
        iid: [False] * len(bboxes) for iid, bboxes in gt_by_image.items()
    }
    tp = fp = 0
    n_gt = sum(len(v) for v in gt_by_image.values())

    for pred in preds:
        iid = pred["image_id"]
        gt_boxes = gt_by_image.get(iid, [])
        flags = matched.get(iid, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if flags[j]:
                continue
            iou = _bbox_iou(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            flags[best_j] = True
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
    gt_by_band: dict[str, dict[int, list[tuple[int, list[float]]]]] = {
        "short": {}, "medium": {}, "long": {}
    }
    for ann in gt_coco.get("annotations", []):
        b = _band(ann["bbox"])
        gt_by_band[b].setdefault(ann["image_id"], []).append((ann["id"], ann["bbox"]))

    preds = [p for p in pred_coco if p["score"] >= conf_threshold]
    preds.sort(key=lambda p: p["score"], reverse=True)

    matched_ann: dict[int, bool] = {ann["id"]: False for ann in gt_coco.get("annotations", [])}

    for pred in preds:
        iid = pred["image_id"]
        best_iou, best_ann_id = 0.0, -1
        for b in ("short", "medium", "long"):
            for ann_id, gt_box in gt_by_band[b].get(iid, []):
                if matched_ann.get(ann_id, False):
                    continue
                iou = _bbox_iou(pred["bbox"], gt_box)
                if iou > best_iou:
                    best_iou, best_ann_id = iou, ann_id
        if best_iou >= iou_threshold and best_ann_id >= 0:
            matched_ann[best_ann_id] = True

    results: dict[str, Any] = {}
    for b in ("short", "medium", "long"):
        all_ids = [ann_id for anns in gt_by_band[b].values() for ann_id, _ in anns]
        n_gt = len(all_ids)
        tp = sum(1 for ann_id in all_ids if matched_ann.get(ann_id, False))
        fn = n_gt - tp
        recall = round(tp / n_gt, 4) if n_gt > 0 else None
        results[b] = {"recall": recall, "tp": tp, "fn": fn, "n_gt": n_gt}
    return results


# ---------------------------------------------------------------------------
# MMDetection evaluation
# ---------------------------------------------------------------------------

def run_mmdet_eval(
    annotation_path: Path,
    checkpoint: Path,
    config: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Run MMDetection inference on annotation_path and return COCO metrics."""
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.evaluate_dino_checkpoint import evaluate_checkpoint

    # Temporarily register the annotation file under the name expected by
    # evaluate_checkpoint (which takes a "split" that resolves to data/annotations/{split}.json)
    split_name = annotation_path.stem
    metrics = evaluate_checkpoint(
        config_path=config,
        checkpoint_path=checkpoint,
        split=split_name,
        work_dir=work_dir,
        output_path=work_dir / "coco_metrics.json",
    )
    return metrics


def find_predictions_file(work_dir: Path, checkpoint: Path, split: str) -> Path:
    candidate = work_dir / f"{checkpoint.stem}_{split}.bbox.json"
    if candidate.exists():
        return candidate
    matches = list(work_dir.glob("*.bbox.json"))
    if matches:
        return sorted(matches)[-1]
    raise FileNotFoundError(f"No bbox predictions found in {work_dir}")


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _recommend(long_recall: float | None) -> str:
    if long_recall is None:
        return "UNKNOWN — no long-band GT annotations; check annotation file"
    if long_recall >= _THRESH_OK:
        return (
            f"OK — long recall {long_recall*100:.1f}% ≥ {_THRESH_OK*100:.0f}%.  "
            "Fine-tuning is optional.  Consider folding into the next scheduled "
            "full retrain rather than fine-tuning now."
        )
    if long_recall >= _THRESH_FINETUNE:
        return (
            f"FINE-TUNE ADVISED — long recall {long_recall*100:.1f}% is "
            f"between {_THRESH_FINETUNE*100:.0f}% and {_THRESH_OK*100:.0f}%.  "
            "Run fine-tune with streak_dinov3_vitb_400px_ft.py; include existing-"
            "domain images at ≥1:1 ratio in the training JSON to prevent regression."
        )
    return (
        f"INVESTIGATE — long recall {long_recall*100:.1f}% < {_THRESH_FINETUNE*100:.0f}%.  "
        "Significant domain shift detected.  Check: (1) pixel scale vs Atwood "
        "(1.27 arcsec/px); (2) FITS normalisation (apply_norm output range); "
        "(3) anchor box coverage for the new streak length distribution.  "
        "Fine-tuning alone may not be sufficient."
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    scope: str,
    label: str,
    checkpoint: Path,
    metrics: dict[str, Any],
    out_dir: Path,
    conf_threshold: float = 0.2,
) -> None:
    pr = metrics.get("pr", {})
    coco = metrics.get("coco_metrics", {})
    bands = metrics.get("per_band_recall", {})
    baseline = _RUN3_BASELINE

    long_recall = bands.get("long", {}).get("recall")
    recommendation = _recommend(long_recall)

    def pct(v: Any) -> str:
        return f"{v * 100:.1f}%" if v is not None else "—"

    def fmt(v: Any, d: int = 3) -> str:
        return f"{v:.{d}f}" if v is not None else "—"

    lines = [
        "# Zero-Shot Evaluation Report",
        "",
        f"**Scope:** {scope}  ",
        f"**Label:** {label}  ",
        f"**Checkpoint:** `{checkpoint}`  ",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Confidence threshold:** {conf_threshold:.2f} | IoU threshold: 0.50  ",
        "",
        "## Decision",
        "",
        f"> {recommendation}",
        "",
        "## Results vs Run 3 Baseline (standard test set)",
        "",
        "| Metric | This scope (zero-shot) | Run 3 baseline | Delta |",
        "|--------|------------------------|----------------|-------|",
    ]

    def delta(new: float | None, base: float | None, pct_fmt: bool = True) -> str:
        if new is None or base is None:
            return "—"
        d = new - base
        sign = "+" if d >= 0 else ""
        if pct_fmt:
            return f"{sign}{d * 100:.1f}pp"
        return f"{sign}{d:.3f}"

    p_new  = pr.get("precision")
    r_new  = pr.get("recall")
    f1_new = pr.get("f1")
    map_new = coco.get("coco/bbox_mAP")
    map50_new = coco.get("coco/bbox_mAP_50")

    rows = [
        ("Precision",     pct(p_new),     pct(baseline["precision"]),   delta(p_new,  baseline["precision"])),
        ("Recall",        pct(r_new),     pct(baseline["recall"]),      delta(r_new,  baseline["recall"])),
        ("F1",            pct(f1_new),    pct(baseline["f1"]),          delta(f1_new, baseline["f1"])),
        ("COCO mAP",      fmt(map_new),   fmt(baseline["mAP"]),         delta(map_new, baseline["mAP"], pct_fmt=False)),
        ("COCO mAP@50",   fmt(map50_new), fmt(baseline["mAP_50"]),      delta(map50_new, baseline["mAP_50"], pct_fmt=False)),
    ]
    for name, new_v, base_v, d in rows:
        lines.append(f"| {name} | {new_v} | {base_v} | {d} |")

    lines += [
        "",
        "## Per-Band Recall",
        "",
        "Bands: short < 269 px diagonal, medium 269–800 px, long > 800 px  ",
        "*(Recall is None when there are no GT annotations in that band.)*",
        "",
        "| Band | This scope | Run 3 baseline | n (this scope) | Delta |",
        "|------|------------|----------------|----------------|-------|",
    ]

    for b in ("short", "medium", "long"):
        bd = bands.get(b, {})
        r_new_b = bd.get("recall")
        r_base_b = baseline["band_recall"].get(b)
        n = bd.get("n_gt", 0)
        lines.append(
            f"| {b.capitalize()} | {pct(r_new_b)} | {pct(r_base_b)} | {n} | {delta(r_new_b, r_base_b)} |"
        )

    lines += [
        "",
        "## Detailed COCO Metrics",
        "",
        f"- mAP:     {fmt(map_new)}",
        f"- mAP@50:  {fmt(map50_new)}",
        f"- mAP@75:  {fmt(coco.get('coco/bbox_mAP_75'))}",
        f"- mAP_s:   {fmt(coco.get('coco/bbox_mAP_s'))}",
        f"- mAP_m:   {fmt(coco.get('coco/bbox_mAP_m'))}",
        f"- mAP_l:   {fmt(coco.get('coco/bbox_mAP_l'))}",
        "",
        "## Detailed P/R",
        "",
        f"- Precision: {pct(p_new)}",
        f"- Recall:    {pct(r_new)}",
        f"- F1:        {pct(f1_new)}",
        f"- TP: {pr.get('tp', '—')}  FP: {pr.get('fp', '—')}  FN: {pr.get('fn', '—')}",
        f"- GT annotations: {pr.get('n_gt', '—')}  "
        f"Predictions above conf: {pr.get('n_pred_above_conf', '—')}",
        "",
        "## Next Steps",
        "",
    ]

    if long_recall is not None and long_recall >= _THRESH_OK:
        lines += [
            "1. Mark the session `split: train` in `data/sessions/manifest.yaml`",
            "   (once Night 2+ annotated).",
            "2. Rebuild training JSON: `python scripts/build_training_json.py`",
            "3. Rebuild when accumulating enough new nights to warrant a full retrain.",
        ]
    elif long_recall is not None and long_recall >= _THRESH_FINETUNE:
        lines += [
            "1. Annotate Night 2 from this scope (target ≥200 images).",
            "2. Add to manifest with `split: train` and appropriate `mix_weight`.",
            "3. Build fine-tune JSON:",
            "   ```",
            "   python scripts/build_training_json.py \\",
            "       --mix-ratio {scope}:<weight> \\",
            f"      --output data/annotations/all_train_ft_{scope}.json",
            "   ```",
            "4. Run fine-tune:",
            "   ```",
            "   TRAIN_ANN_FILE=annotations/all_train_ft_{scope}.json \\",
            "   python -m training.train_dino \\",
            "       --config models/dino/streak_dinov3_vitb_400px_ft.py \\",
            f"      --work-dir weights/run_ft_{scope}",
            "   ```",
            "5. Re-evaluate on BOTH this scope (Night 1) AND the standard test set.",
            "   Accept only if standard-test recall does not drop > 2pp.",
        ]
    else:
        lines += [
            "1. Investigate domain shift before training:",
            f"   - Pixel scale: this scope vs Atwood (1.27 arcsec/px baseline)",
            "   - FITS normalisation: check `apply_norm()` output range on a sample",
            "   - Streak length distribution: compare band histogram to training data",
            "2. If pixel scale differs significantly, consider a new resolution tier",
            "   in the training config (current: 400px tiles).",
            "3. Only proceed to fine-tuning after understanding the cause of the gap.",
        ]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    logger.info("Report: %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="COCO annotation JSON for the new scope's first night "
             "(streak images only; will be copied to data/annotations/ if not already there)",
    )
    p.add_argument(
        "--negatives",
        type=Path,
        default=None,
        help="Optional COCO JSON of no-streak images from the same night "
             "(increases FP context for precision measurement)",
    )
    p.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Base directory for resolving bare filenames in the annotation file.  "
             "Not needed if annotation_file already contains absolute paths.",
    )
    p.add_argument(
        "--scope",
        type=str,
        required=True,
        help="Short identifier for the new scope (e.g. 'ridge_park', 'obs_b').  "
             "Used in output directory names.",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Human-readable label for the scope in the report "
             "(defaults to --scope if not provided)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help=f"Model checkpoint to evaluate (default: {_DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"MMDetection config (default: {_DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for P/R calculation (default: 0.20)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    label = args.label or args.scope

    os.chdir(_REPO_ROOT)
    sys.path.insert(0, str(_REPO_ROOT))

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")
    if not args.annotation.exists():
        raise SystemExit(f"Annotation file not found: {args.annotation}")

    # Build a merged COCO dict (fix bare paths, normalise categories)
    logger.info("Loading annotation: %s", args.annotation)
    gt_coco = build_eval_coco(args.annotation, args.negatives, args.raw_dir)
    n_img = len(gt_coco["images"])
    n_ann = len(gt_coco.get("annotations", []))
    logger.info("  %d images, %d annotations", n_img, n_ann)

    if n_ann == 0:
        raise SystemExit(
            "No annotations found in the provided file.  "
            "Check that the annotation file contains annotated streak images "
            "(Reject=0 in .strk files, processed by convert_gtimages.py)."
        )

    # Write a temporary annotation file in data/annotations/ where MMDetection
    # expects to find split JSON files
    tmp_split_name = f"_zs_eval_{args.scope}"
    tmp_ann_path = _REPO_ROOT / "data/annotations" / f"{tmp_split_name}.json"
    logger.info("Writing temporary annotation file: %s", tmp_ann_path)
    write_tmp_coco(gt_coco, tmp_ann_path)

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _REPO_ROOT / f"results/zero_shot_{args.scope}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results → %s", out_dir)

    # Copy the config for reproducibility
    import shutil
    shutil.copy(args.config, out_dir / args.config.name)

    # Run MMDetection evaluation
    logger.info("Running MMDetection inference…")
    try:
        coco_metrics = run_mmdet_eval(
            annotation_path=tmp_ann_path,
            checkpoint=args.checkpoint,
            config=args.config,
            work_dir=out_dir,
        )
    except Exception as exc:
        logger.error("MMDetection eval failed: %s\n%s", exc, traceback.format_exc())
        raise SystemExit(1) from exc
    finally:
        # Clean up the temporary annotation file
        if tmp_ann_path.exists():
            tmp_ann_path.unlink()
            logger.info("Removed temporary annotation file: %s", tmp_ann_path.name)

    # Load predictions for custom metrics
    try:
        pred_path = find_predictions_file(out_dir, args.checkpoint, tmp_split_name)
    except FileNotFoundError:
        logger.error("Could not find predictions file in %s", out_dir)
        raise SystemExit(1)

    with open(pred_path) as f:
        pred_coco = json.load(f)

    # Reload GT from the tmp file if still available; otherwise use in-memory dict
    # (we deleted it above, so use the in-memory version)
    pr_metrics = compute_pr(gt_coco, pred_coco, conf_threshold=args.conf)
    band_metrics = compute_per_band_recall(gt_coco, pred_coco, conf_threshold=args.conf)

    results = {
        "scope": args.scope,
        "label": label,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "annotation_file": str(args.annotation),
        "n_images": n_img,
        "n_annotations": n_ann,
        "coco_metrics": coco_metrics,
        "pr": pr_metrics,
        "per_band_recall": band_metrics,
        "baseline": _RUN3_BASELINE,
    }

    # Write structured results
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics: %s", metrics_path)

    # Write human-readable report
    write_report(
        scope=args.scope,
        label=label,
        checkpoint=args.checkpoint,
        metrics=results,
        out_dir=out_dir,
        conf_threshold=args.conf,
    )

    # Print decision to stdout for easy reading
    long_recall = band_metrics.get("long", {}).get("recall")
    print()
    print("=" * 60)
    print(f"  ZERO-SHOT EVAL: {label}")
    print("=" * 60)
    print(f"  Recall:       {pr_metrics['recall']*100:.1f}%")
    print(f"  Precision:    {pr_metrics['precision']*100:.1f}%")
    print(f"  Long recall:  {long_recall*100:.1f}% (n={band_metrics['long']['n_gt']})" if long_recall else "  Long recall:  — (no long GT)")
    print(f"  Med  recall:  {band_metrics['medium'].get('recall', 0)*100:.1f}% (n={band_metrics['medium']['n_gt']})")
    print(f"  Short recall: {band_metrics['short'].get('recall', 0)*100:.1f}% (n={band_metrics['short']['n_gt']})")
    print()
    print(f"  RECOMMENDATION: {_recommend(long_recall)}")
    print("=" * 60)
    print(f"  Full report: {out_dir / 'report.md'}")
    print()


def find_predictions_file(work_dir: Path, checkpoint: Path, split: str) -> Path:
    candidate = work_dir / f"{checkpoint.stem}_{split}.bbox.json"
    if candidate.exists():
        return candidate
    matches = list(work_dir.glob("*.bbox.json"))
    if matches:
        return sorted(matches)[-1]
    raise FileNotFoundError(f"No bbox predictions found in {work_dir}")


if __name__ == "__main__":
    main()
