"""Line-segment-based streak evaluation metrics.

Replaces OBB IoU matching with a three-criterion match that is geometrically
meaningful for thin elongated streaks:

  1. Angle error < angle_tol_deg   (180°-symmetric)
  2. Perpendicular offset < perp_tol_px
  3. 1-D length IoU along GT axis > length_iou_min

This avoids the pathological IoU collapse that OBB matching produces when a
streak is detected at a small lateral offset (a 400×3 px streak at 2 px offset
gets OBB IoU ≈ 0.20 and is counted as FP even though it is geometrically
correct).

Prediction / GT dict formats accepted by :func:`evaluate_segments`:

  prediction  — {image_id, confidence, x1, y1, x2, y2, streak_length_px}
  ground_truth — {image_id, x1, y1, x2, y2, streak_length_px}

Legacy ``obb``-only dicts are also accepted for backward compatibility.

# Source: StreakMind — segment-match evaluation methodology
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Import canonical types from inference module — single source of truth.
from inference.streak_segment import StreakSegment, obb_to_segment, detection_dict_to_segment  # noqa: F401

logger = logging.getLogger(__name__)

# Band thresholds (pixels) — kept in sync with eval/metrics.py
_SHORT_MAX = 150.0
_LONG_MIN = 400.0


# ---------------------------------------------------------------------------
# Three-criterion match geometry
# ---------------------------------------------------------------------------

def _angle_error_deg(pred_angle: float, gt_angle: float) -> float:
    """Angular error in degrees accounting for 180° streak symmetry.

    Args:
        pred_angle: Predicted angle in degrees.
        gt_angle: Ground-truth angle in degrees.

    Returns:
        Absolute angular error in [0, 90].
    """
    diff = abs(pred_angle - gt_angle) % 180.0
    return min(diff, 180.0 - diff)


def _perpendicular_offset(pred: StreakSegment, gt: StreakSegment) -> float:
    """Perpendicular distance from the GT line to the predicted centre.

    Projects ``pred.cx, pred.cy`` onto the GT line direction and measures
    the signed lateral residual.

    Args:
        pred: Predicted segment.
        gt: Ground-truth segment.

    Returns:
        Non-negative perpendicular distance in pixels.
    """
    rad = math.radians(gt.angle_deg)
    cos_gt = math.cos(rad)
    sin_gt = math.sin(rad)
    dx = pred.cx - gt.cx
    dy = pred.cy - gt.cy
    return abs(dx * sin_gt - dy * cos_gt)


def _length_1d_iou(pred: StreakSegment, gt: StreakSegment) -> float:
    """1-D IoU of the two segments projected onto the GT axis direction.

    Args:
        pred: Predicted segment.
        gt: Ground-truth segment.

    Returns:
        1-D IoU in [0, 1].
    """
    rad = math.radians(gt.angle_deg)
    cos_gt = math.cos(rad)
    sin_gt = math.sin(rad)

    # GT interval is [-half, +half] in the GT axis frame
    gt_half = gt.length_px / 2.0
    gt_lo = -gt_half
    gt_hi = gt_half

    # Project pred endpoints onto GT axis
    t1 = (pred.x1 - gt.cx) * cos_gt + (pred.y1 - gt.cy) * sin_gt
    t2 = (pred.x2 - gt.cx) * cos_gt + (pred.y2 - gt.cy) * sin_gt
    pred_lo = min(t1, t2)
    pred_hi = max(t1, t2)

    inter = max(0.0, min(pred_hi, gt_hi) - max(pred_lo, gt_lo))
    union = max(pred_hi, gt_hi) - min(pred_lo, gt_lo)
    return float(inter / union) if union > 0.0 else 0.0


def segment_match(
    pred: StreakSegment,
    gt: StreakSegment,
    angle_tol_deg: float = 15.0,
    perp_tol_px: float = 5.0,
    length_iou_min: float = 0.5,
) -> bool:
    """Return True if ``pred`` matches ``gt`` under all three criteria.

    A match requires:

    1. ``angle_error_deg(pred, gt) < angle_tol_deg``
    2. ``perpendicular_offset(pred, gt) < perp_tol_px``
    3. ``length_1d_iou(pred, gt) > length_iou_min``

    Args:
        pred: Predicted segment.
        gt: Ground-truth segment.
        angle_tol_deg: Maximum allowed angle error in degrees (default 5).
        perp_tol_px: Maximum allowed perpendicular offset in pixels (default 5).
        length_iou_min: Minimum 1-D IoU along GT axis (default 0.5).

    Returns:
        True if all three criteria are met.
    """
    if _angle_error_deg(pred.angle_deg, gt.angle_deg) >= angle_tol_deg:
        return False
    if _perpendicular_offset(pred, gt) >= perp_tol_px:
        return False
    if _length_1d_iou(pred, gt) <= length_iou_min:
        return False
    return True


# ---------------------------------------------------------------------------
# Greedy matching (segment-based)
# ---------------------------------------------------------------------------

def _greedy_match_segments(
    preds_for_image: list[StreakSegment],
    gts_for_image: list[StreakSegment],
    angle_tol_deg: float,
    perp_tol_px: float,
    length_iou_min: float,
) -> tuple[list[tuple[int, int]], list[bool]]:
    """Greedy segment matching between predictions and GTs for one image.

    Predictions must be pre-sorted by confidence descending.

    Args:
        preds_for_image: Predicted segments for a single image.
        gts_for_image: Ground-truth segments for the same image.
        angle_tol_deg: Angle tolerance in degrees.
        perp_tol_px: Perpendicular offset tolerance in pixels.
        length_iou_min: Minimum 1-D IoU for a match.

    Returns:
        matched_pairs: List of (pred_index, gt_index) tuples.
        is_tp: Boolean mask aligned with preds_for_image.
    """
    is_tp = [False] * len(preds_for_image)
    matched_gts: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []

    for pi, pred in enumerate(preds_for_image):
        # Find best GT by 1-D IoU (among those passing angle + perp criteria)
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gts_for_image):
            if gi in matched_gts:
                continue
            if not segment_match(pred, gt, angle_tol_deg, perp_tol_px, length_iou_min):
                continue
            iou = _length_1d_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_gi >= 0:
            is_tp[pi] = True
            matched_gts.add(best_gi)
            matched_pairs.append((pi, best_gi))

    return matched_pairs, is_tp


def _match_all_segments(
    predictions: list[StreakSegment],
    ground_truth: list[StreakSegment],
    angle_tol_deg: float,
    perp_tol_px: float,
    length_iou_min: float,
) -> tuple[list[bool], int, list[tuple[StreakSegment, StreakSegment]]]:
    """Match all predicted segments to GT segments across all images.

    Args:
        predictions: All predicted segments (any order).
        ground_truth: All ground-truth segments.
        angle_tol_deg: Angle tolerance in degrees.
        perp_tol_px: Perpendicular offset tolerance in pixels.
        length_iou_min: Minimum 1-D IoU for a match.

    Returns:
        is_tp: Boolean list aligned with globally confidence-sorted predictions.
        n_gt: Total number of ground-truth segments.
        matched_pairs: List of (pred_segment, gt_segment) for matched pairs.
    """
    preds_by_img: dict[Any, list[StreakSegment]] = defaultdict(list)
    gts_by_img: dict[Any, list[StreakSegment]] = defaultdict(list)
    for p in predictions:
        preds_by_img[p.image_id].append(p)
    for g in ground_truth:
        gts_by_img[g.image_id].append(g)

    sorted_preds: list[StreakSegment] = []
    is_tp_all: list[bool] = []
    matched_pairs: list[tuple[StreakSegment, StreakSegment]] = []

    all_image_ids = set(preds_by_img) | set(gts_by_img)
    for img_id in all_image_ids:
        img_preds = sorted(preds_by_img[img_id], key=lambda x: x.confidence, reverse=True)
        img_gts = gts_by_img[img_id]
        pairs, is_tp = _greedy_match_segments(
            img_preds, img_gts, angle_tol_deg, perp_tol_px, length_iou_min
        )
        sorted_preds.extend(img_preds)
        is_tp_all.extend(is_tp)
        for pi, gi in pairs:
            matched_pairs.append((img_preds[pi], img_gts[gi]))

    # Re-sort globally by confidence
    order = sorted(
        range(len(sorted_preds)),
        key=lambda i: sorted_preds[i].confidence,
        reverse=True,
    )
    is_tp_sorted = [is_tp_all[i] for i in order]
    return is_tp_sorted, len(ground_truth), matched_pairs


# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------

def evaluate_segments(
    predictions: list[dict],
    ground_truth: list[dict],
    angle_tol_deg: float = 15.0,
    perp_tol_px: float = 5.0,
    length_iou_min: float = 0.5,
    short_max_px: float = _SHORT_MAX,
    long_min_px: float = _LONG_MIN,
) -> dict:
    """Compute the full evaluation metric suite using line-segment matching.

    Accepts the same ``{image_id, confidence, obb, streak_length_px}`` dict
    format as :func:`eval.metrics.evaluate` and returns the same top-level
    keys plus two new diagnostics for matched pairs.

    Args:
        predictions: List of detection dicts (image_id, confidence, obb,
            streak_length_px).
        ground_truth: List of annotation dicts (image_id, obb,
            streak_length_px).
        angle_tol_deg: Maximum angle error for a match (degrees, default 5).
        perp_tol_px: Maximum perpendicular offset for a match (pixels, default 5).
        length_iou_min: Minimum 1-D IoU along GT axis for a match (default 0.5).
        short_max_px: Upper bound on "short" streak length band (pixels).
        long_min_px: Lower bound on "long" streak length band (pixels).

    Returns:
        Dict with keys:

        - ``precision``, ``recall``, ``f1``
        - ``mean_angle_error_deg``
        - ``mean_perp_offset_px``   (new — mean over matched pairs)
        - ``mean_length_1d_iou``    (new — mean over matched pairs)
        - ``per_band``: dict with ``short``, ``medium``, ``long`` sub-dicts,
          each containing ``precision``, ``recall``, ``f1``.
    """
    if not ground_truth or not predictions:
        return _empty_result()

    pred_segs = [detection_dict_to_segment(p) for p in predictions]
    for seg, p in zip(pred_segs, predictions):
        seg.streak_length_px = float(p.get("streak_length_px") or seg.length_px)

    gt_segs = [detection_dict_to_segment(g) for g in ground_truth]
    for seg, g in zip(gt_segs, ground_truth):
        seg.streak_length_px = float(g.get("streak_length_px") or seg.length_px)

    is_tp, n_gt, matched_pairs = _match_all_segments(
        pred_segs, gt_segs, angle_tol_deg, perp_tol_px, length_iou_min
    )

    tp = sum(is_tp)
    fp = len(is_tp) - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Diagnostic averages over matched pairs
    if matched_pairs:
        mean_angle_err = float(
            np.mean([_angle_error_deg(p.angle_deg, g.angle_deg) for p, g in matched_pairs])
        )
        mean_perp = float(
            np.mean([_perpendicular_offset(p, g) for p, g in matched_pairs])
        )
        mean_len_iou = float(
            np.mean([_length_1d_iou(p, g) for p, g in matched_pairs])
        )
    else:
        mean_angle_err = mean_perp = mean_len_iou = 0.0

    # Per-band breakdown
    bands = {
        "short":  (0.0, short_max_px),
        "medium": (short_max_px, long_min_px),
        "long":   (long_min_px, float("inf")),
    }
    per_band: dict[str, dict[str, float]] = {}
    for band_name, (lo, hi) in bands.items():
        band_preds = [
            p for p in predictions
            if lo <= float(p.get("streak_length_px") or 0.0) < hi
        ]
        band_gts = [
            g for g in ground_truth
            if lo <= float(g.get("streak_length_px") or 0.0) < hi
        ]
        if band_gts and band_preds:
            bp, br, bf = _band_prf(
                band_preds, band_gts,
                angle_tol_deg, perp_tol_px, length_iou_min,
            )
        else:
            bp = br = bf = 0.0
        per_band[band_name] = {
            "precision": round(bp, 4),
            "recall": round(br, 4),
            "f1": round(bf, 4),
        }

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_angle_error_deg": round(mean_angle_err, 3),
        "mean_perp_offset_px": round(mean_perp, 3),
        "mean_length_1d_iou": round(mean_len_iou, 4),
        "per_band": per_band,
    }


def _band_prf(
    predictions: list[dict],
    ground_truth: list[dict],
    angle_tol_deg: float,
    perp_tol_px: float,
    length_iou_min: float,
) -> tuple[float, float, float]:
    """Precision, recall, F1 for a subset of predictions/GT.

    Args:
        predictions: Filtered prediction dicts.
        ground_truth: Filtered GT dicts.
        angle_tol_deg: Angle tolerance in degrees.
        perp_tol_px: Perpendicular offset tolerance in pixels.
        length_iou_min: Minimum 1-D IoU.

    Returns:
        Tuple of (precision, recall, f1).
    """
    pred_segs = [detection_dict_to_segment(p) for p in predictions]
    gt_segs = [detection_dict_to_segment(g) for g in ground_truth]

    is_tp, n_gt, _ = _match_all_segments(
        pred_segs, gt_segs, angle_tol_deg, perp_tol_px, length_iou_min
    )
    tp = sum(is_tp)
    fp = len(is_tp) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _empty_result() -> dict:
    """Return zero-valued result dict.

    Returns:
        Dict with all metric keys set to 0.
    """
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mean_angle_error_deg": 0.0,
        "mean_perp_offset_px": 0.0,
        "mean_length_1d_iou": 0.0,
        "per_band": {
            "short":  {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "medium": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "long":   {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        },
    }


# ---------------------------------------------------------------------------
# JSON file-based evaluation
# ---------------------------------------------------------------------------

def _load_predictions_json(path: Path) -> list[dict]:
    """Load predictions list from a JSON file.

    Handles both a bare list and a dict with a ``predictions`` key.

    Args:
        path: Path to the JSON file.

    Returns:
        List of prediction dicts.
    """
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "predictions" in data:
            return data["predictions"]
        # Threshold sweep row format — not a flat prediction list
        raise ValueError(
            f"{path} appears to be a summary JSON, not a flat predictions list. "
            "Pass the raw predictions.json file."
        )
    raise ValueError(f"Unexpected JSON structure in {path}")


def _load_gt_from_coco(path: Path) -> list[dict]:
    """Load ground-truth annotations from a COCO-style JSON file.

    Converts each annotation's ``bbox`` (XYWH) and ``obb`` into the flat
    dict format expected by :func:`evaluate_segments`.

    Args:
        path: Path to the COCO annotations JSON file.

    Returns:
        List of GT dicts with keys ``image_id``, ``obb``, ``streak_length_px``.
    """
    with path.open() as f:
        coco = json.load(f)

    gts: list[dict] = []
    for ann in coco.get("annotations", []):
        obb = ann.get("obb")
        if not obb:
            # Fall back: derive from COCO bbox (x, y, w, h)
            bbox = ann["bbox"]
            x, y, bw, bh = bbox
            obb = {
                "cx": x + bw / 2,
                "cy": y + bh / 2,
                "w": bw,
                "h": bh,
                "angle_deg": 0.0,
            }
        length_px = float(
            ann.get("attributes", {}).get("length_px")
            or obb.get("w")
            or 0.0
        )
        gts.append({
            "image_id": ann["image_id"],
            "obb": obb,
            "streak_length_px": length_px,
        })
    return gts


def evaluate_segments_at_thresholds(
    predictions_json_path: str | Path,
    annotations_json_path: str | Path,
    thresholds: list[float],
    angle_tol_deg: float = 5.0,
    perp_tol_px: float = 5.0,
    length_iou_min: float = 0.5,
    short_max_px: float = _SHORT_MAX,
    long_min_px: float = _LONG_MIN,
) -> list[dict]:
    """Evaluate predictions at multiple confidence thresholds using segment match.

    Loads predictions and GT from JSON files, then calls
    :func:`evaluate_segments` at each threshold.

    Args:
        predictions_json_path: Path to flat predictions JSON list.
        annotations_json_path: Path to COCO-style annotations JSON.
        thresholds: Confidence thresholds to sweep.
        angle_tol_deg: Angle tolerance for segment match (degrees).
        perp_tol_px: Perpendicular offset tolerance (pixels).
        length_iou_min: Minimum 1-D IoU for a match.
        short_max_px: Upper bound for "short" band (pixels).
        long_min_px: Lower bound for "long" band (pixels).

    Returns:
        List of result dicts, one per threshold, each containing a
        ``threshold`` key plus all keys from :func:`evaluate_segments`.
    """
    predictions_json_path = Path(predictions_json_path)
    annotations_json_path = Path(annotations_json_path)

    logger.info("Loading predictions from %s", predictions_json_path)
    all_preds = _load_predictions_json(predictions_json_path)

    logger.info("Loading ground truth from %s", annotations_json_path)
    gts = _load_gt_from_coco(annotations_json_path)

    rows: list[dict] = []
    for t in sorted(thresholds):
        preds = [p for p in all_preds if float(p.get("confidence", 0.0)) >= t]
        logger.info("threshold=%.3f  n_preds=%d  n_gt=%d", t, len(preds), len(gts))
        result = evaluate_segments(
            preds, gts,
            angle_tol_deg=angle_tol_deg,
            perp_tol_px=perp_tol_px,
            length_iou_min=length_iou_min,
            short_max_px=short_max_px,
            long_min_px=long_min_px,
        )
        rows.append({"threshold": t, **result})
    return rows


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json

    logging.basicConfig(level=logging.INFO)
    print("=== streak_metrics.py smoke test ===")

    # A perfect match: identical OBBs
    pred_obb = {"cx": 100.0, "cy": 100.0, "w": 300.0, "h": 4.0, "angle_deg": 15.0}
    gt_obb   = {"cx": 100.0, "cy": 100.0, "w": 300.0, "h": 4.0, "angle_deg": 15.0}

    preds = [{"image_id": "img1", "confidence": 0.9, "obb": pred_obb, "streak_length_px": 300.0}]
    gts   = [{"image_id": "img1", "obb": gt_obb, "streak_length_px": 300.0}]

    result = evaluate_segments(preds, gts)
    print("Perfect match:", _json.dumps(result, indent=2))
    assert result["recall"] == 1.0,   f"Expected recall=1.0, got {result['recall']}"
    assert result["precision"] == 1.0, f"Expected precision=1.0, got {result['precision']}"
    print("PASS: perfect match")

    # OBB IoU failure case: 2 px lateral offset on a 400×3 px streak
    # OBB IoU would give ~0.20, but segment match should still match
    pred_obb2 = {"cx": 200.0, "cy": 102.0, "w": 400.0, "h": 3.0, "angle_deg": 0.0}
    gt_obb2   = {"cx": 200.0, "cy": 100.0, "w": 400.0, "h": 3.0, "angle_deg": 0.0}
    preds2 = [{"image_id": "img2", "confidence": 0.8, "obb": pred_obb2, "streak_length_px": 400.0}]
    gts2   = [{"image_id": "img2", "obb": gt_obb2, "streak_length_px": 400.0}]

    result2 = evaluate_segments(preds2, gts2, perp_tol_px=5.0)
    print("2px lateral offset:", _json.dumps(result2, indent=2))
    assert result2["recall"] == 1.0,   f"Expected recall=1.0 for 2px offset, got {result2['recall']}"
    print("PASS: 2px lateral offset matched (OBB IoU would fail this)")

    # Ensure a large lateral offset is correctly rejected
    pred_obb3 = {"cx": 200.0, "cy": 120.0, "w": 400.0, "h": 3.0, "angle_deg": 0.0}
    preds3 = [{"image_id": "img3", "confidence": 0.8, "obb": pred_obb3, "streak_length_px": 400.0}]
    gts3   = [{"image_id": "img3", "obb": gt_obb2, "streak_length_px": 400.0}]
    result3 = evaluate_segments(preds3, gts3, perp_tol_px=5.0)
    assert result3["recall"] == 0.0,   f"Expected recall=0.0 for 20px offset, got {result3['recall']}"
    print("PASS: 20px lateral offset correctly rejected")

    print("\nAll smoke tests passed.")
