"""Line-segment metrics for no-OBB streak detection.

These metrics compare predicted centerline segments directly against annotated
streak centerlines.  They are intended for heatmap detectors where converting
to boxes would hide the actual quality of the output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _obb_to_line_segment(obb: dict[str, Any]) -> dict[str, float]:
    """Convert an OBB annotation to its centerline segment."""
    cx = float(obb["cx"])
    cy = float(obb["cy"])
    half = float(obb["w"]) * 0.5
    angle = math.radians(float(obb.get("angle_deg", 0.0)))
    dx = half * math.cos(angle)
    dy = half * math.sin(angle)
    return {
        "x1": cx - dx,
        "y1": cy - dy,
        "x2": cx + dx,
        "y2": cy + dy,
        "angle_deg": float(obb.get("angle_deg", 0.0)) % 180.0,
        "length_px": float(obb["w"]),
    }


def _prediction_line(prediction: dict[str, Any]) -> dict[str, float] | None:
    """Return a native line segment from a prediction, falling back to OBB."""
    line = prediction.get("line_segment")
    if isinstance(line, dict):
        try:
            x1 = float(line["x1"])
            y1 = float(line["y1"])
            x2 = float(line["x2"])
            y2 = float(line["y2"])
        except (KeyError, TypeError, ValueError):
            return None
        dx = x2 - x1
        dy = y2 - y1
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "angle_deg": float(line.get("angle_deg", math.degrees(math.atan2(dy, dx)))) % 180.0,
            "length_px": float(line.get("length_px", math.hypot(dx, dy))),
        }
    obb = prediction.get("obb")
    if isinstance(obb, dict):
        return _obb_to_line_segment(obb)
    return None


def load_line_ground_truth(annotations_path: str | Path) -> list[dict[str, Any]]:
    """Load COCO OBB annotations as centerline ground truth."""
    coco = json.loads(Path(annotations_path).read_text())
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    records: list[dict[str, Any]] = []
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        raw = ann.get("obb")
        if not raw:
            continue
        if isinstance(raw, dict):
            obb = {
                "cx": float(raw["cx"]),
                "cy": float(raw["cy"]),
                "w": float(raw["w"]),
                "h": float(raw["h"]),
                "angle_deg": float(raw["angle_deg"]),
            }
        else:
            if len(raw) < 5:
                continue
            cx, cy, w, h, angle_deg = [float(value) for value in raw[:5]]
            obb = {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle_deg}
        line = _obb_to_line_segment(obb)
        records.append(
            {
                "image_id": id_to_filename.get(ann["image_id"], str(ann["image_id"])),
                "line_segment": line,
                "streak_length_px": float(line["length_px"]),
            }
        )
    return records


def _sample_segment(line: dict[str, float], step_px: float = 1.0) -> np.ndarray:
    """Sample points along a line segment."""
    x1 = float(line["x1"])
    y1 = float(line["y1"])
    x2 = float(line["x2"])
    y2 = float(line["y2"])
    length = max(math.hypot(x2 - x1, y2 - y1), 1.0)
    count = max(2, int(math.ceil(length / max(step_px, 1e-6))) + 1)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return np.column_stack((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))


def _point_to_segment_distances(points: np.ndarray, line: dict[str, float]) -> np.ndarray:
    """Return distance from each point to a segment."""
    a = np.array([float(line["x1"]), float(line["y1"])], dtype=np.float32)
    b = np.array([float(line["x2"]), float(line["y2"])], dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-6:
        return np.linalg.norm(points - a[None, :], axis=1)
    t = np.clip(((points - a[None, :]) @ ab) / denom, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - closest, axis=1)


def _angle_error_deg(a: float, b: float) -> float:
    """Return 180-degree-symmetric angle error."""
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def _line_match_score(
    prediction: dict[str, float],
    ground_truth: dict[str, float],
    tolerance_px: float,
) -> dict[str, float]:
    """Return distance-tolerant mutual coverage between two line segments."""
    gt_points = _sample_segment(ground_truth)
    pred_points = _sample_segment(prediction)
    gt_coverage = float((_point_to_segment_distances(gt_points, prediction) <= tolerance_px).mean())
    pred_coverage = float((_point_to_segment_distances(pred_points, ground_truth) <= tolerance_px).mean())
    return {
        "score": min(gt_coverage, pred_coverage),
        "gt_coverage": gt_coverage,
        "pred_coverage": pred_coverage,
        "angle_error_deg": _angle_error_deg(prediction.get("angle_deg", 0.0), ground_truth.get("angle_deg", 0.0)),
    }


def _greedy_line_match(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    tolerance_px: float,
    coverage_threshold: float,
) -> tuple[list[bool], list[tuple[dict[str, Any], dict[str, Any], dict[str, float]]]]:
    """Greedily match line predictions to ground truth per image."""
    from collections import defaultdict

    preds_by_img: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    gts_by_img: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        line = _prediction_line(pred)
        if line is None:
            continue
        preds_by_img[pred["image_id"]].append({**pred, "line_segment": line})
    for gt in ground_truth:
        gts_by_img[gt["image_id"]].append(gt)

    sorted_predictions: list[dict[str, Any]] = []
    is_tp_all: list[bool] = []
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, float]]] = []
    for image_id in set(preds_by_img) | set(gts_by_img):
        image_preds = sorted(preds_by_img[image_id], key=lambda item: item.get("confidence", 0.0), reverse=True)
        image_gts = gts_by_img[image_id]
        matched_gts: set[int] = set()
        image_tp = [False] * len(image_preds)
        for pred_idx, pred in enumerate(image_preds):
            best_idx = -1
            best_metrics: dict[str, float] | None = None
            best_score = 0.0
            for gt_idx, gt in enumerate(image_gts):
                if gt_idx in matched_gts:
                    continue
                metrics = _line_match_score(pred["line_segment"], gt["line_segment"], tolerance_px)
                if metrics["score"] > best_score:
                    best_score = metrics["score"]
                    best_idx = gt_idx
                    best_metrics = metrics
            if best_idx >= 0 and best_metrics and best_score >= coverage_threshold:
                matched_gts.add(best_idx)
                image_tp[pred_idx] = True
                matched_pairs.append((pred, image_gts[best_idx], best_metrics))
        sorted_predictions.extend(image_preds)
        is_tp_all.extend(image_tp)

    order = sorted(range(len(sorted_predictions)), key=lambda idx: sorted_predictions[idx].get("confidence", 0.0), reverse=True)
    return [is_tp_all[idx] for idx in order], matched_pairs


def evaluate_line_segments(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    tolerance_px: float = 6.0,
    coverage_threshold: float = 0.10,
) -> dict[str, Any]:
    """Evaluate no-OBB line-segment predictions.

    Args:
        predictions: Prediction dicts with ``image_id`` and ``line_segment``.
            Predictions with only ``obb`` are converted to centerlines.
        ground_truth: Records from ``load_line_ground_truth``.
        tolerance_px: Maximum point-to-segment distance for coverage.
        coverage_threshold: Minimum mutual coverage score to count a match.

    Returns:
        Precision, recall, F1, image-level false-positive rate, and matched
        geometry diagnostics.
    """
    is_tp, matched_pairs = _greedy_line_match(
        predictions,
        ground_truth,
        tolerance_px=tolerance_px,
        coverage_threshold=coverage_threshold,
    )
    n_predictions = len([p for p in predictions if _prediction_line(p) is not None])
    tp = int(sum(is_tp))
    fp = int(n_predictions - tp)
    fn = int(len(ground_truth) - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    gt_positive_images = {gt["image_id"] for gt in ground_truth}
    pred_positive_images = {
        pred["image_id"]
        for pred in predictions
        if _prediction_line(pred) is not None
    }
    negative_pred_images = pred_positive_images - gt_positive_images
    matched_metrics = [metrics for _, _, metrics in matched_pairs]
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_predictions": n_predictions,
        "n_ground_truth": len(ground_truth),
        "positive_images_with_prediction": len(pred_positive_images & gt_positive_images),
        "negative_images_with_prediction": len(negative_pred_images),
        "tolerance_px": tolerance_px,
        "coverage_threshold": coverage_threshold,
        "mean_gt_coverage": round(float(np.mean([m["gt_coverage"] for m in matched_metrics])), 4)
        if matched_metrics
        else 0.0,
        "mean_pred_coverage": round(float(np.mean([m["pred_coverage"] for m in matched_metrics])), 4)
        if matched_metrics
        else 0.0,
        "mean_angle_error_deg": round(float(np.mean([m["angle_error_deg"] for m in matched_metrics])), 3)
        if matched_metrics
        else 0.0,
    }
