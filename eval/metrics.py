"""Evaluation metrics for ARGUS streak detection.

All functions operate on two flat lists:

  predictions — one dict per predicted detection:
    {
      "image_id":        str | int,
      "confidence":      float,          # in [0, 1]
      "obb":             dict,           # {cx, cy, w, h, angle_deg}
      "streak_length_px": float,         # used for per-band breakdown
    }

  ground_truth — one dict per annotated detection:
    {
      "image_id":        str | int,
      "obb":             dict,           # {cx, cy, w, h, angle_deg}
      "streak_length_px": float,
    }

Streak length bands (pixels):
  short  : < 150
  medium : 150 – 400
  long   : > 400

# Source: StreakMind — mAP and angle-error evaluation methodology
# Ref: agent_docs/argus_phases.md
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from inference.confidence import compute_unified_confidence

# Band thresholds (pixels)
_SHORT_MAX = 150.0
_LONG_MIN = 400.0


# ---------------------------------------------------------------------------
# OBB geometry helpers
# ---------------------------------------------------------------------------

def _obb_to_corners(obb: dict) -> np.ndarray:
    """Return (4, 2) array of OBB corner coordinates.

    Args:
        obb: Dict with keys cx, cy, w, h, angle_deg.

    Returns:
        Array of shape (4, 2) in image pixel space.
    """
    cx, cy = float(obb["cx"]), float(obb["cy"])
    w, h = float(obb["w"]), float(obb["h"])
    rad = math.radians(float(obb["angle_deg"]))
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    half_w, half_h = w / 2, h / 2
    local = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return (local @ rot.T) + np.array([cx, cy])


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula for polygon area.

    Args:
        pts: (N, 2) array of vertices.

    Returns:
        Signed area (take abs for unsigned).
    """
    x, y = pts[:, 0], pts[:, 1]
    return float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _obb_iou(a: dict, b: dict) -> float:
    """Compute IoU between two oriented bounding boxes using Shapely.

    Falls back to axis-aligned bbox IoU when Shapely is unavailable or
    when either OBB is very thin (h < 5px), which causes degenerate Shapely
    geometry when the OBBs are not precisely aligned.

    Args:
        a: OBB dict for prediction.
        b: OBB dict for ground truth.

    Returns:
        IoU in [0, 1].
    """
    # Use axis-aligned bbox IoU when ground truth is a thin streak (h < 5px).
    # DINO outputs axis-aligned bboxes; OBB angle from bbox aspect-ratio is too
    # coarse to produce valid rotated-polygon IoU against 3px-wide GT streaks.
    if float(b.get("h", 99)) < 5.0 or float(a.get("h", 99)) < 5.0:
        return _bbox_iou_from_obb(a, b)

    try:
        from shapely.geometry import Polygon

        poly_a = Polygon(_obb_to_corners(a))
        poly_b = Polygon(_obb_to_corners(b))
        if not poly_a.is_valid or not poly_b.is_valid:
            return _bbox_iou_from_obb(a, b)
        inter = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return _bbox_iou_from_obb(a, b)


def _bbox_iou_from_obb(a: dict, b: dict) -> float:
    """Axis-aligned bbox IoU derived from OBB extents (ignores rotation).

    Args:
        a: OBB dict (cx, cy, w, h, angle_deg).
        b: OBB dict (cx, cy, w, h, angle_deg).

    Returns:
        IoU in [0, 1].
    """
    def _to_xyxy(obb: dict) -> tuple[float, float, float, float]:
        cx, cy = float(obb["cx"]), float(obb["cy"])
        hw, hh = float(obb["w"]) / 2, float(obb["h"]) / 2
        # Use the bbox enclosing the OBB (axis-aligned)
        rad = math.radians(float(obb.get("angle_deg", 0)))
        cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
        bw = hw * cos_a + hh * sin_a
        bh = hw * sin_a + hh * cos_a
        return cx - bw, cy - bh, cx + bw, cy + bh

    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _angle_error_deg(pred_angle: float, gt_angle: float) -> float:
    """Angular error in degrees accounting for 180° streak symmetry.

    A streak at θ° is identical to one at θ+180°, so errors are clamped
    to [0°, 90°].

    Args:
        pred_angle: Predicted angle in degrees.
        gt_angle: Ground-truth angle in degrees.

    Returns:
        Absolute angular error in [0, 90].
    """
    diff = abs(pred_angle - gt_angle) % 180.0
    return min(diff, 180.0 - diff)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _greedy_match(
    preds_for_image: list[dict],
    gts_for_image: list[dict],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[bool]]:
    """Greedy IoU matching between predictions and GTs for one image.

    Predictions must be pre-sorted by confidence descending.

    Args:
        preds_for_image: Predictions for a single image.
        gts_for_image: Ground-truth annotations for the same image.
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        matched_pairs: List of (pred_index, gt_index) tuples.
        is_tp: Boolean mask aligned with preds_for_image.
    """
    is_tp = [False] * len(preds_for_image)
    matched_gts: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []

    for pi, pred in enumerate(preds_for_image):
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gts_for_image):
            if gi in matched_gts:
                continue
            iou = _obb_iou(pred["obb"], gt["obb"])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_threshold and best_gi >= 0:
            is_tp[pi] = True
            matched_gts.add(best_gi)
            matched_pairs.append((pi, best_gi))

    return matched_pairs, is_tp


def _match_all(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float,
) -> tuple[list[bool], int, list[tuple[dict, dict]]]:
    """Match all predictions to ground truth across all images.

    Args:
        predictions: All predicted detections (any order).
        ground_truth: All ground-truth annotations.
        iou_threshold: IoU threshold for a match.

    Returns:
        is_tp:         Boolean list aligned with predictions sorted by confidence.
        n_gt:          Total number of ground-truth annotations.
        matched_pairs: List of (pred_dict, gt_dict) for matched pairs.
    """
    # Group by image_id
    from collections import defaultdict
    preds_by_img: dict = defaultdict(list)
    gts_by_img: dict = defaultdict(list)
    for p in predictions:
        preds_by_img[p["image_id"]].append(p)
    for g in ground_truth:
        gts_by_img[g["image_id"]].append(g)

    # Sort each image's predictions by confidence descending
    sorted_preds: list[dict] = []
    is_tp_all: list[bool] = []
    matched_pairs: list[tuple[dict, dict]] = []

    all_image_ids = set(preds_by_img) | set(gts_by_img)
    for img_id in all_image_ids:
        img_preds = sorted(preds_by_img[img_id], key=lambda x: x.get("confidence", 0), reverse=True)
        img_gts = gts_by_img[img_id]
        pairs, is_tp = _greedy_match(img_preds, img_gts, iou_threshold)
        sorted_preds.extend(img_preds)
        is_tp_all.extend(is_tp)
        for pi, gi in pairs:
            matched_pairs.append((img_preds[pi], img_gts[gi]))

    # Re-sort globally by confidence (for mAP PR curve)
    order = sorted(range(len(sorted_preds)), key=lambda i: sorted_preds[i].get("confidence", 0), reverse=True)
    is_tp_sorted = [is_tp_all[i] for i in order]

    return is_tp_sorted, len(ground_truth), matched_pairs


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def _compute_ap(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float,
) -> float:
    """Compute Average Precision (AP) at a given IoU threshold.

    Uses the area-under-curve approach over all confidence thresholds.

    Args:
        predictions: All predicted detections.
        ground_truth: All ground-truth annotations.
        iou_threshold: IoU threshold for a match.

    Returns:
        AP in [0, 1].
    """
    if not predictions or not ground_truth:
        return 0.0

    is_tp, n_gt, _ = _match_all(predictions, ground_truth, iou_threshold)

    tp_cum = np.cumsum(is_tp).astype(float)
    fp_cum = np.cumsum([not t for t in is_tp]).astype(float)

    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / n_gt

    # Prepend sentinel values for AUC calculation
    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])

    # Monotonically decreasing precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Area under curve (trapezoid rule over recall axis)
    recall_changes = np.diff(recalls)
    return float(np.sum(recall_changes * precisions[1:]))


def _compute_prf(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 at a given IoU threshold.

    Uses confidence-agnostic matching (all predictions are considered).

    Args:
        predictions: All predicted detections.
        ground_truth: All ground-truth annotations.
        iou_threshold: IoU threshold for a match.

    Returns:
        Tuple of (precision, recall, f1), each in [0, 1].
    """
    if not ground_truth:
        return (0.0, 0.0, 0.0)
    if not predictions:
        return (0.0, 0.0, 0.0)

    is_tp, n_gt, _ = _match_all(predictions, ground_truth, iou_threshold)
    tp = sum(is_tp)
    fp = len(is_tp) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _compute_mean_angle_error(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Mean angle error (degrees) over IoU-matched prediction / GT pairs.

    Args:
        predictions: All predicted detections.
        ground_truth: All ground-truth annotations.
        iou_threshold: IoU threshold for a match.

    Returns:
        Mean angular error in degrees, or 0.0 if no matches.
    """
    if not predictions or not ground_truth:
        return 0.0

    _, _, matched_pairs = _match_all(predictions, ground_truth, iou_threshold)
    if not matched_pairs:
        return 0.0

    errors = [
        _angle_error_deg(
            pred["obb"]["angle_deg"],
            gt["obb"]["angle_deg"],
        )
        for pred, gt in matched_pairs
    ]
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
    short_threshold: float = _SHORT_MAX,
    long_threshold: float = _LONG_MIN,
) -> dict:
    """Compute the full evaluation metric suite for streak detection.

    Args:
        predictions: List of detection dicts (image_id, confidence, obb,
                     streak_length_px).
        ground_truth: List of annotation dicts (image_id, obb,
                      streak_length_px).
        iou_threshold: Primary IoU threshold for precision/recall/F1.
        short_threshold: Streak length (px) upper bound for "short" band.
        long_threshold: Streak length (px) lower bound for "long" band.

    Returns:
        Dict with keys: precision, recall, f1, map_50, map_75,
        mean_angle_error_deg, per_band (short/medium/long each with
        precision/recall/f1).
    """
    precision, recall, f1 = _compute_prf(predictions, ground_truth, iou_threshold)
    map_50 = _compute_ap(predictions, ground_truth, 0.5)
    map_75 = _compute_ap(predictions, ground_truth, 0.75)
    mean_angle_err = _compute_mean_angle_error(predictions, ground_truth, iou_threshold)

    bands = {
        "short":  (0.0, short_threshold),
        "medium": (short_threshold, long_threshold),
        "long":   (long_threshold, float("inf")),
    }
    per_band: dict[str, dict[str, float]] = {}
    for band_name, (lo, hi) in bands.items():
        def in_band(det: dict, lo: float = lo, hi: float = hi) -> bool:
            length = det.get("streak_length_px") or 0.0
            return lo <= length < hi

        band_preds = [p for p in predictions if in_band(p)]
        band_gts = [g for g in ground_truth if in_band(g)]
        bp, br, bf = _compute_prf(band_preds, band_gts, iou_threshold)
        per_band[band_name] = {"precision": bp, "recall": br, "f1": bf}

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map_50": round(map_50, 4),
        "map_75": round(map_75, 4),
        "mean_angle_error_deg": round(mean_angle_err, 3),
        "per_band": {
            k: {m: round(v, 4) for m, v in band.items()}
            for k, band in per_band.items()
        },
    }


# ---------------------------------------------------------------------------
# Multi-method prediction extraction
# ---------------------------------------------------------------------------

def extract_method_predictions(
    grouped_pipeline_output: list[dict],
    image_id: str | int,
) -> dict[str, list[dict]]:
    """Extract per-method and unified prediction lists from grouped pipeline output.

    ``inference.pipeline.run()`` returns one dict per streak group, each with
    a ``sources`` list recording which detectors fired and at what confidence.
    This function fans those entries back out into flat per-method prediction
    lists that are compatible with ``evaluate()``.

    The primary OBB (from the highest-confidence single-method detection in
    the group) is reused for every source in that group.  This is a necessary
    approximation: the pipeline retains only one OBB per streak group.

    A synthetic ``"unified"`` method is included whose confidence is the
    noisy-OR combination of all individual-method confidences::

        unified_conf = 1 – Π(1 – conf_i)

    For a single-source group this equals that source's confidence exactly.

    Args:
        grouped_pipeline_output: Streak-group dicts returned by
            ``inference.pipeline.run()``.  Each must have ``obb``,
            ``streak_length_px``, and ``sources`` keys.
        image_id: Image identifier (filename) stamped on every prediction.

    Returns:
        Dict mapping method name → list of prediction dicts for ``evaluate()``.
        Always includes the ``"unified"`` key.
    """
    method_preds: dict[str, list[dict]] = {}
    unified_preds: list[dict] = []

    for group in grouped_pipeline_output:
        obb = group.get("obb")
        if not obb:
            continue
        length = group.get("streak_length_px") or 0.0
        # Filter out the synthetic "unified" entry if the pipeline already added it.
        sources = [s for s in (group.get("sources") or []) if s.get("method") != "unified"]

        for src in sources:
            method = src.get("method")
            if not method:
                continue
            method_preds.setdefault(method, []).append({
                "image_id": image_id,
                "confidence": float(src["confidence"]),
                "obb": obb,
                "streak_length_px": float(length),
            })

        if sources:
            unified_conf = compute_unified_confidence(sources)["score"]
            unified_preds.append({
                "image_id": image_id,
                "confidence": unified_conf,
                "obb": obb,
                "streak_length_px": float(length),
            })

    method_preds["unified"] = unified_preds
    return method_preds


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
    save_path: str | Path | None = None,
    title: str = "",
) -> np.ndarray:
    """Compute and optionally plot a 2×2 confusion matrix for streak detection.

    Rows = actual (Positive streak / Negative no-streak).
    Columns = predicted (Positive / Negative).

    An image is treated as positive if it has ≥ 1 ground-truth annotation.
    A detection is a True Positive if it IoU-matches a GT box above threshold.

    Args:
        predictions: All predicted detections (image_id, confidence, obb, …).
        ground_truth: All ground-truth annotations (image_id, obb, …).
        iou_threshold: IoU threshold for TP matching.
        save_path: If given, save a PNG confusion-matrix plot to this path.
            Parent directory is created if needed.
        title: Optional label shown in the PNG title (e.g. method name).

    Returns:
        2×2 numpy array [[TP, FP], [FN, TN]] as int32.

    # Source: StreakMind — evaluation methodology
    # Ref: agent_docs/argus_phases.md
    """
    is_tp, n_gt, _ = _match_all(predictions, ground_truth, iou_threshold)
    tp = int(sum(is_tp))
    fp = int(len(is_tp) - tp)
    fn = int(n_gt - tp)

    # TN: images with no GT and no prediction
    all_image_ids = {p["image_id"] for p in predictions} | {g["image_id"] for g in ground_truth}
    from collections import defaultdict
    pred_by_img: dict = defaultdict(list)
    gt_by_img:   dict = defaultdict(list)
    for p in predictions:
        pred_by_img[p["image_id"]].append(p)
    for g in ground_truth:
        gt_by_img[g["image_id"]].append(g)

    tn = sum(
        1 for img_id in all_image_ids
        if not gt_by_img[img_id] and not pred_by_img[img_id]
    )

    cm = np.array([[tp, fp], [fn, tn]], dtype=np.int32)

    if save_path is not None:
        _plot_confusion_matrix(cm, Path(save_path), title=title)

    return cm


def _plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str = "") -> None:
    """Render and save a confusion matrix PNG.

    Args:
        cm: 2×2 array [[TP, FP], [FN, TN]].
        save_path: Output path for the PNG file.
        title: Optional title prefix shown above the matrix.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib optional — skip silently

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Streak\n(Positive)", "No-streak\n(Negative)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nPositive", "Predicted\nNegative"])
    ax.set_yticklabels(["Actual\nPositive", "Actual\nNegative"])

    labels = [["TP", "FP"], ["FN", "TN"]]
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{labels[i][j]}\n{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    prefix = f"{title} — " if title else ""
    ax.set_title(f"{prefix}Confusion Matrix (IoU ≥ 0.5)\n"
                 f"Precision={cm[0,0]/(cm[0,0]+cm[0,1]+1e-9):.1%}  "
                 f"Recall={cm[0,0]/(cm[0,0]+cm[1,0]+1e-9):.1%}")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-identification quality metrics
# ---------------------------------------------------------------------------

def evaluate_crossid(detections: list[dict]) -> dict:
    """Compute cross-identification quality metrics from pipeline detections.

    Extracts along-track (Atrk) and cross-track (Xtrk) residuals from the
    top-1 candidate of each identified detection and computes RMS statistics.

    Atrk RMS reflects timing/epoch errors; Xtrk RMS reflects orbital plane
    errors.  A correct identification typically has Xtrk < Atrk.

    # Source: SkyTrack (colleague) — ComputeRMSResidual
    # Ref: examples/streak_live.inc, line 2150

    Args:
        detections: Detection dicts that have gone through cross_identify(),
            each with an 'identifications' list whose top entry may contain
            'atrk_arcsec' and 'xtrk_arcsec' keys.

    Returns:
        Dict with keys:
          n_detections        — total number of input detections
          n_identified        — detections with at least one candidate
          identification_rate — n_identified / n_detections
          n_with_residuals    — candidates that have Atrk/Xtrk values
          atrk_rms_arcsec     — RMS along-track residual (arcsec)
          xtrk_rms_arcsec     — RMS cross-track residual (arcsec)
          total_rms_arcsec    — RMS of sqrt(Atrk²+Xtrk²) (arcsec)
          top1_confidence_mean — mean confidence of top-1 candidates
    """
    n_det   = len(detections)
    n_id    = 0
    conf_sum = 0.0
    atrk_sq_sum = 0.0
    xtrk_sq_sum = 0.0
    total_sq_sum = 0.0
    n_resid = 0

    for det in detections:
        ids = det.get("identifications") or []
        if not ids:
            continue
        n_id += 1
        top = ids[0]
        conf_sum += float(top.get("confidence", 0.0))

        atrk = top.get("atrk_arcsec")
        xtrk = top.get("xtrk_arcsec")
        if atrk is not None and xtrk is not None:
            atrk = float(atrk)
            xtrk = float(xtrk)
            atrk_sq_sum  += atrk ** 2
            xtrk_sq_sum  += xtrk ** 2
            total_sq_sum += atrk ** 2 + xtrk ** 2
            n_resid += 1

    id_rate          = n_id / n_det if n_det > 0 else 0.0
    conf_mean        = conf_sum / n_id if n_id > 0 else 0.0
    atrk_rms         = math.sqrt(atrk_sq_sum  / n_resid) if n_resid > 0 else 0.0
    xtrk_rms         = math.sqrt(xtrk_sq_sum  / n_resid) if n_resid > 0 else 0.0
    total_rms        = math.sqrt(total_sq_sum / n_resid) if n_resid > 0 else 0.0

    return {
        "n_detections":         n_det,
        "n_identified":         n_id,
        "identification_rate":  round(id_rate, 4),
        "n_with_residuals":     n_resid,
        "atrk_rms_arcsec":      round(atrk_rms,  2),
        "xtrk_rms_arcsec":      round(xtrk_rms,  2),
        "total_rms_arcsec":     round(total_rms,  2),
        "top1_confidence_mean": round(conf_mean,  4),
    }


if __name__ == "__main__":
    import json

    # Smoke test with synthetic data
    preds = [
        {"image_id": "img1", "confidence": 0.95, "obb": {"cx": 100, "cy": 100, "w": 200, "h": 10, "angle_deg": 5.0}, "streak_length_px": 200},
        {"image_id": "img1", "confidence": 0.60, "obb": {"cx": 300, "cy": 300, "w": 100, "h": 8,  "angle_deg": -10.0}, "streak_length_px": 100},
    ]
    gts = [
        {"image_id": "img1", "obb": {"cx": 100, "cy": 100, "w": 200, "h": 10, "angle_deg": 4.0}, "streak_length_px": 200},
        {"image_id": "img1", "obb": {"cx": 300, "cy": 300, "w": 100, "h": 8,  "angle_deg": -9.0}, "streak_length_px": 100},
    ]
    result = evaluate(preds, gts)
    print(json.dumps(result, indent=2))
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    print("Smoke test passed.")
