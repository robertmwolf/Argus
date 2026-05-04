"""Radon-based angle refinement and oriented bounding-box NMS for ARGUS.

Post-processes raw axis-aligned detections from DINO into oriented bounding
boxes (OBBs) with sub-degree angle precision, then suppresses duplicate
detections via rotated-IoU NMS.

CPU-only: skimage.transform.radon and Shapely are not GPU-accelerated.
This is expected — do not attempt to move these operations to MPS or CUDA.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OBB construction
# ---------------------------------------------------------------------------

def bbox_to_obb(bbox: list[float], angle_deg: float) -> dict:
    """Convert an axis-aligned bounding box and streak angle to an OBB dict.

    The OBB is centred on the bbox midpoint.  The *w* (long) axis is the
    extent of the bbox projected onto the streak direction; *h* (short) axis
    is the perpendicular extent — an estimate of streak width.

    Args:
        bbox: [x1, y1, x2, y2] axis-aligned box in pixel coordinates.
        angle_deg: Streak orientation in degrees (0–180).

    Returns:
        Dict with keys: cx, cy, w, h, angle_deg.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = abs(x2 - x1)
    bh = abs(y2 - y1)

    theta = math.radians(angle_deg % 180.0)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))

    # Extent along the streak axis
    w = bw * cos_t + bh * sin_t
    # Extent perpendicular to the streak (width of the streak itself)
    h = bw * sin_t + bh * cos_t
    # Ensure w >= h (w is always the long axis)
    if h > w:
        w, h = h, w
        angle_deg = (angle_deg + 90.0) % 180.0

    return {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle_deg}


# ---------------------------------------------------------------------------
# Radon angle refinement
# ---------------------------------------------------------------------------

def refine_angle(
    image_crop: np.ndarray,
    obb: dict,
    angle_search_range: float = 15.0,
) -> float:
    """Refine OBB angle using the Radon transform on the streak crop.

    Searches a narrow angular window around DINO's initial estimate.
    The Radon sinogram column with maximum variance corresponds to the
    projection angle at which the streak integrates to a single bright peak —
    i.e. the true streak orientation.

    CPU-only — skimage.transform.radon is numpy-backed.

    # Source: StreakMind — Radon angle refinement for OBB streaks
    # Ref: agent_docs/streakmind_phases.md

    Args:
        image_crop: Greyscale or 3-channel uint8/float32 crop centred on the
            streak.  If 3-channel, the mean across channels is used.
        obb: Detection OBB dict {cx, cy, w, h, angle_deg}.
        angle_search_range: ±degrees around DINO's predicted angle to search.
            Must be ≥ 0.  If 0, returns obb['angle_deg'] unchanged.

    Returns:
        Refined angle in degrees in the range [0, 180).
    """
    # Import here so CPU-only module is not loaded until needed
    from skimage.transform import radon  # type: ignore[import]

    initial_angle = obb.get("angle_deg", 0.0)

    if angle_search_range <= 0:
        return float(initial_angle % 180.0)

    # Normalise crop to float32 greyscale
    crop = np.asarray(image_crop, dtype=np.float32)
    if crop.ndim == 3:
        crop = crop.mean(axis=2)

    if crop.size == 0 or min(crop.shape) < 4:
        logger.debug("refine_angle: crop too small (%s), returning initial angle", crop.shape)
        return float(initial_angle % 180.0)

    # Coordinate system: skimage.transform.radon at angle θ integrates along
    # lines perpendicular to θ — it rotates the image by −θ then sums columns.
    # The sinogram variance peaks when θ is perpendicular to the streak, so:
    #
    #   θ_radon_peak  =  90° − φ_streak  (mod 180°)
    #   φ_streak      =  90° − θ_radon   (mod 180°)
    #
    # We must search in Radon-angle space, then convert the winner back.
    # Do NOT wrap the search window to [0, 180°) before calling radon — that
    # breaks the contiguous range when the window straddles the boundary.
    # skimage accepts any real θ, treating θ and θ+180° as equivalent.
    step = 0.5
    radon_center = 90.0 - initial_angle
    radon_angles = np.arange(
        radon_center - angle_search_range,
        radon_center + angle_search_range + step,
        step,
    )

    try:
        sinogram = radon(crop, theta=radon_angles, circle=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("radon transform failed: %s — returning initial angle", exc)
        return float(initial_angle % 180.0)

    # Column with maximum variance → strongest perpendicular projection
    col_variances = sinogram.var(axis=0)
    best_idx = int(np.argmax(col_variances))
    best_radon_angle = float(radon_angles[best_idx])

    # Convert winning Radon angle back to image streak angle
    refined = float((90.0 - best_radon_angle) % 180.0)

    logger.debug(
        "refine_angle: initial=%.1f°  radon_center=%.1f°  best_radon=%.1f°  refined=%.1f°",
        initial_angle, radon_center, best_radon_angle, refined,
    )
    return refined


# ---------------------------------------------------------------------------
# Rotated-IoU NMS
# ---------------------------------------------------------------------------

def _obb_to_polygon(obb: dict):  # returns shapely.Polygon
    """Convert an OBB dict to a Shapely Polygon (rotated rectangle).

    Args:
        obb: Dict with keys cx, cy, w, h, angle_deg.

    Returns:
        shapely.geometry.Polygon representing the four corners of the OBB.
    """
    from shapely.geometry import Polygon  # type: ignore[import]

    cx = obb["cx"]
    cy = obb["cy"]
    w = obb["w"]
    h = obb["h"]
    theta = math.radians(obb["angle_deg"])
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    hw = w / 2.0
    hh = h / 2.0

    # Four corners in local OBB frame, rotated to image frame
    corners = [
        (cx + hw * cos_t - hh * sin_t, cy + hw * sin_t + hh * cos_t),
        (cx - hw * cos_t - hh * sin_t, cy - hw * sin_t + hh * cos_t),
        (cx - hw * cos_t + hh * sin_t, cy - hw * sin_t - hh * cos_t),
        (cx + hw * cos_t + hh * sin_t, cy + hw * sin_t - hh * cos_t),
    ]
    return Polygon(corners)


def _rotated_iou(obb_a: dict, obb_b: dict) -> float:
    """Compute intersection-over-union for two OBBs using Shapely.

    Args:
        obb_a: First OBB dict {cx, cy, w, h, angle_deg}.
        obb_b: Second OBB dict {cx, cy, w, h, angle_deg}.

    Returns:
        IoU in [0, 1].  Returns 0.0 if either polygon is degenerate.
    """
    try:
        poly_a = _obb_to_polygon(obb_a)
        poly_b = _obb_to_polygon(obb_b)
        if not poly_a.is_valid or not poly_b.is_valid:
            return 0.0
        intersection = poly_a.intersection(poly_b).area
        if intersection == 0.0:
            return 0.0
        union = poly_a.area + poly_b.area - intersection
        return float(intersection / union) if union > 0 else 0.0
    except Exception as exc:  # pragma: no cover
        logger.debug("rotated IoU failed: %s", exc)
        return 0.0


def nms_detections(
    detections: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Non-maximum suppression on OBB detections using Shapely polygon IoU.

    Suppresses lower-confidence detections whose rotated IoU with a
    kept detection exceeds *iou_threshold*.

    Args:
        detections: List of detection dicts, each containing an 'obb' key
            (from bbox_to_obb) and a 'confidence' key.
        iou_threshold: Suppress if rotated IoU exceeds this value.

    Returns:
        Filtered list of detection dicts, ordered by confidence descending.
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)

    kept: list[dict] = []
    suppressed = set()

    for i, det in enumerate(sorted_dets):
        if i in suppressed:
            continue
        kept.append(det)
        obb_i = det.get("obb")
        if obb_i is None:
            continue
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            obb_j = sorted_dets[j].get("obb")
            if obb_j is None:
                continue
            iou = _rotated_iou(obb_i, obb_j)
            if iou > iou_threshold:
                suppressed.add(j)
                logger.debug(
                    "NMS: suppressed detection %d (IoU=%.3f > %.3f)", j, iou, iou_threshold
                )

    logger.debug("NMS: kept %d / %d detections", len(kept), len(sorted_dets))
    return kept


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    # Quick smoke-test: build a tiny synthetic streak image and refine its angle
    rng = np.random.default_rng(0)
    h, w = 128, 128
    img = rng.integers(90, 110, size=(h, w), dtype=np.uint8).astype(np.float32)

    # Inject a streak at 45°
    true_angle = 45.0
    theta_rad = math.radians(true_angle)
    for t in np.linspace(-50, 50, 300):
        px = int(w // 2 + t * math.cos(theta_rad))
        py = int(h // 2 + t * math.sin(theta_rad))
        if 0 <= px < w and 0 <= py < h:
            img[py, px] += 500.0

    obb = {"cx": 64.0, "cy": 64.0, "w": 110.0, "h": 4.0, "angle_deg": 50.0}
    refined = refine_angle(img.astype(np.uint8), obb, angle_search_range=20.0)
    print(f"True angle: {true_angle}°  DINO initial: {obb['angle_deg']}°  Refined: {refined:.1f}°")
    assert abs(refined - true_angle) <= 5.0, f"Refinement too far off: {refined:.1f}°"
    print("refine_angle smoke-test passed.")

    # NMS smoke-test: two heavily overlapping detections
    dets = [
        {"confidence": 0.9, "obb": {"cx": 64, "cy": 64, "w": 80, "h": 5, "angle_deg": 45}},
        {"confidence": 0.6, "obb": {"cx": 65, "cy": 65, "w": 80, "h": 5, "angle_deg": 45}},
        {"confidence": 0.8, "obb": {"cx": 200, "cy": 200, "w": 80, "h": 5, "angle_deg": 10}},
    ]
    kept = nms_detections(dets, iou_threshold=0.5)
    print(f"NMS: {len(dets)} → {len(kept)} (expected 2)")
    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.9
    print("NMS smoke-test passed.")
    sys.exit(0)
