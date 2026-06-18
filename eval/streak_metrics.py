"""Endpoint-based streak evaluation metrics.

Uses a three-criterion match that is geometrically meaningful for thin streaks:

  1. Angle error < angle_tol_deg   (180°-symmetric)
  2. Perpendicular offset < perp_tol_px
  3. 1-D length IoU along GT axis > length_iou_min

Prediction / GT dict formats accepted by :func:`evaluate_segments`:

  prediction  — {image_id, confidence, x1, y1, x2, y2, streak_length_px}
  ground_truth — {image_id, x1, y1, x2, y2, streak_length_px}

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
from inference.streak_segment import StreakSegment, detection_dict_to_segment
from training.annotation_endpoints import annotation_to_endpoints

logger = logging.getLogger(__name__)

# Band thresholds (pixels) — architecture-aligned (see agent_docs/heatmap_training_lessons.md).
#   < _MIN_LENGTH      sub-resolvable (≈ <4 ViT patches at 400px tiling) — out of scope
#   [_MIN_LENGTH, _SHORT_MAX)  short  — fits in one 400px tile, no stitch (pure model capability)
#   [_SHORT_MAX, _LONG_MIN)    medium — multi-tile, stitch-dependent
#   [_LONG_MIN, inf)           long   — many-tile, stitch stress-test (~spans a large frame fraction)
# _SHORT_MAX = native_tile_size (single-tile/multi-tile boundary). Frigate (<50px) is excluded.
_MIN_LENGTH = 50.0
_SHORT_MAX = 400.0
_LONG_MIN = 1000.0


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
    min_length_px: float = _MIN_LENGTH,
) -> dict:
    """Compute the full evaluation metric suite using line-segment matching.

    Args:
        predictions: Endpoint detection dictionaries.
        ground_truth: Endpoint annotation dictionaries.
        angle_tol_deg: Maximum angle error for a match (degrees, default 5).
        perp_tol_px: Maximum perpendicular offset for a match (pixels, default 5).
        length_iou_min: Minimum 1-D IoU along GT axis for a match (default 0.5).
        short_max_px: Upper bound on "short" streak length band (pixels).
        long_min_px: Lower bound on "long" streak length band (pixels).
        min_length_px: Sub-resolution floor (pixels). Streaks shorter than this
            are dropped from BOTH ground truth and predictions before scoring:
            sub-floor GT is not counted as missed recall, and sub-floor
            predictions are dropped (not counted as false positives). Excludes
            Frigate-scale (<50px) micro-streaks the 16px-patch grid cannot resolve.

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

    # Drop sub-resolution streaks (< min_length_px) from both sides. Below the
    # ~4-patch resolvability floor the 16px-grid architecture cannot form a line,
    # so sub-floor GT is not scored as missed recall and sub-floor predictions
    # are dropped (not counted as FPs). Keeps preds/GT and their segments in sync.
    if min_length_px > 0:
        keep_p = [i for i, s in enumerate(pred_segs) if s.streak_length_px >= min_length_px]
        keep_g = [i for i, s in enumerate(gt_segs) if s.streak_length_px >= min_length_px]
        predictions = [predictions[i] for i in keep_p]
        pred_segs   = [pred_segs[i] for i in keep_p]
        ground_truth = [ground_truth[i] for i in keep_g]
        gt_segs     = [gt_segs[i] for i in keep_g]
        if not ground_truth or not predictions:
            return _empty_result()

    is_tp, n_gt, matched_pairs = _match_all_segments(
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
        "short":  (min_length_px, short_max_px),
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

    Converts historical source annotations at the dataset boundary.

    Args:
        path: Path to the COCO annotations JSON file.

    Returns:
        Endpoint ground-truth dictionaries.
    """
    with path.open() as f:
        coco = json.load(f)

    gts: list[dict] = []
    for ann in coco.get("annotations", []):
        x1, y1, x2, y2 = annotation_to_endpoints(ann)
        length_px = math.hypot(x2 - x1, y2 - y1)
        gts.append({
            "image_id": ann["image_id"],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
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
    logging.basicConfig(level=logging.INFO)
    segment = {"image_id": "img1", "confidence": 0.9,
               "x1": 0.0, "y1": 10.0, "x2": 300.0, "y2": 10.0}
    result = evaluate_segments([segment], [segment])
    assert result["recall"] == 1.0
    print("Endpoint metric smoke test passed.")
