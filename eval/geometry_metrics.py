"""Canonical endpoint geometry evaluation for ARGUS models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval.streak_metrics import (
    _angle_error_deg,
    _load_gt_from_coco,
    _match_all_segments,
)
from inference.streak_segment import StreakSegment, detection_dict_to_segment

DEFAULT_PERP_THRESHOLD_PX = 20.0


def _endpoint_error_px(pred: StreakSegment, truth: StreakSegment) -> float:
    """Return mean endpoint error, invariant to endpoint order."""
    direct = (
        math.hypot(pred.x1 - truth.x1, pred.y1 - truth.y1)
        + math.hypot(pred.x2 - truth.x2, pred.y2 - truth.y2)
    ) / 2.0
    swapped = (
        math.hypot(pred.x1 - truth.x2, pred.y1 - truth.y2)
        + math.hypot(pred.x2 - truth.x1, pred.y2 - truth.y1)
    ) / 2.0
    return min(direct, swapped)


def _summary(values: list[float]) -> dict[str, float]:
    """Summarize a geometry error vector."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 3),
        "median": round(float(np.median(array)), 3),
        "p90": round(float(np.percentile(array, 90)), 3),
    }


def _band(length: float) -> str:
    if length < 400.0:
        return "short"
    if length < 1000.0:
        return "medium"
    return "long"


def _geometry_stats(pairs: list[tuple[StreakSegment, StreakSegment]]) -> dict[str, Any]:
    angles = [_angle_error_deg(pred.angle_deg, truth.angle_deg) for pred, truth in pairs]
    endpoints = [_endpoint_error_px(pred, truth) for pred, truth in pairs]
    return {
        "n_pairs": len(pairs),
        "angle_err_deg": _summary(angles),
        "endpoint_err_px": _summary(endpoints),
    }


def evaluate_geometry(
    predictions: list[dict],
    ground_truth: list[dict],
    perp_threshold_px: float = DEFAULT_PERP_THRESHOLD_PX,
) -> dict[str, Any]:
    """Evaluate detection, angle, and endpoint accuracy.

    Args:
        predictions: Endpoint prediction dictionaries.
        ground_truth: Endpoint annotation dictionaries.
        perp_threshold_px: Maximum centerline offset for matching.
    Returns:
        Tiered metrics compatible with existing result comparison tooling.
    """
    pred_segments = [detection_dict_to_segment(item) for item in predictions]
    truth_segments = [detection_dict_to_segment(item) for item in ground_truth]
    is_tp, n_gt, pairs = _match_all_segments(
        pred_segments,
        truth_segments,
        angle_tol_deg=15.0,
        perp_tol_px=perp_threshold_px,
        length_iou_min=0.1,
    )
    found = len(pairs)
    false_positives = len(is_tp) - sum(is_tp)
    band_gt = {name: 0 for name in ("short", "medium", "long")}
    band_found = {name: 0 for name in band_gt}
    for truth in truth_segments:
        band_gt[_band(truth.length_px)] += 1
    for _, truth in pairs:
        band_found[_band(truth.length_px)] += 1

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    tier1 = {
        "n_gt": n_gt,
        "n_found": found,
        "n_false_positives": false_positives,
        "detection_recall": ratio(found, n_gt),
        "detection_precision": ratio(found, found + false_positives),
        "per_band": {
            name: {
                "n_gt": band_gt[name],
                "n_found": band_found[name],
                "recall": ratio(band_found[name], band_gt[name]),
            }
            for name in band_gt
        },
    }
    return {
        "perp_threshold_px": perp_threshold_px,
        "tier1_detection": tier1,
        "tier2_raw_geometry": _geometry_stats(pairs),
        "tier3_refined_geometry": None,
    }


def _load_predictions(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["predictions"] if isinstance(data, dict) else data


def main() -> int:
    """Run endpoint geometry evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--perp-threshold-px", type=float, default=DEFAULT_PERP_THRESHOLD_PX)
    args = parser.parse_args()
    metrics = evaluate_geometry(
        _load_predictions(args.predictions),
        _load_gt_from_coco(args.annotations),
        args.perp_threshold_px,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
