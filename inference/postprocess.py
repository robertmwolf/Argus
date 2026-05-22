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

    # Extent along the Radon-refined streak axis (angle_deg must not change here —
    # flipping it by 90° would cause extend_obb_to_streak_extent to scan
    # perpendicular to the streak and miss all bright pixels).
    w = bw * cos_t + bh * sin_t
    # Extent perpendicular to the streak axis (approximates streak width)
    h = bw * sin_t + bh * cos_t

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
    # Ref: agent_docs/argus_phases.md

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

    # Downsample large crops before Radon to keep runtime bounded.
    # DINO bboxes scaled back from a low-resolution inference pass (e.g. 256 px
    # on a 6k-pixel sensor) produce crops of 2000–3000 px; Radon on these with
    # 360 angles takes many minutes on CPU.  512 px preserves sub-degree
    # angular precision while keeping Radon runtime under ~1 s.
    _MAX_RADON_SIDE = 512
    h_crop, w_crop = crop.shape[:2]
    if max(h_crop, w_crop) > _MAX_RADON_SIDE:
        import cv2 as _cv2
        scale_r = _MAX_RADON_SIDE / max(h_crop, w_crop)
        crop = _cv2.resize(
            crop,
            (max(1, int(w_crop * scale_r)), max(1, int(h_crop * scale_r))),
            interpolation=_cv2.INTER_AREA,
        )
        logger.debug("refine_angle: downsampled crop %s→%s for Radon", (h_crop, w_crop), crop.shape)

    # Subtract background so Radon variance reflects the streak, not the sky level.
    # Without this, a non-zero background dominates the Radon at all angles and
    # the variance peak gets pulled toward whichever axis is most compressed by
    # the crop geometry (typically 90° for tall narrow crops).
    crop = np.clip(crop - np.median(crop), 0.0, None)

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
    step = 1.0
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
# Streak extent — extend OBB endpoints to the true streak tips
# ---------------------------------------------------------------------------

def extend_obb_to_streak_extent(
    array: np.ndarray,
    obb: dict,
    sample_halfwidth: int = 8,
    threshold_sigma: float = 3.0,
    _gray: "np.ndarray | None" = None,
    _threshold: float | None = None,
) -> dict:
    """Extend OBB w/cx/cy so endpoints cover the full streak, not just the bbox.

    DINO bboxes often capture only a portion of a long streak.  This function
    traces the streak axis across the full image and finds where the signal
    drops to background level, then returns an updated OBB with corrected
    centre and long-axis length.

    Args:
        array: uint8 (H, W, 3) or (H, W) image array from FITSLoader.
        obb: OBB dict {cx, cy, w, h, angle_deg} — modified copy is returned.
        sample_halfwidth: Half-width of the perpendicular sampling strip.
            Must be large enough to cover the typical offset between the bbox
            centre and the actual streak axis — default 8 px.
        threshold_sigma: Multiplier on std above the image median.  Set higher
            than the mean-based default (1.5) because we use the strip maximum
            rather than the mean; default 3.0 keeps the background false-positive
            rate below ~2 % for a 17-pixel strip.

    Returns:
        Updated OBB dict with corrected cx, cy, w.  h and angle_deg unchanged.
    """
    cx = obb["cx"]
    cy = obb["cy"]
    angle_rad = math.radians(obb["angle_deg"])
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    if _gray is not None:
        gray = _gray
    else:
        gray = np.asarray(array, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)
    h_img, w_img = gray.shape

    if _threshold is not None:
        threshold = _threshold
    else:
        bg = float(np.median(gray))
        threshold = bg + threshold_sigma * float(gray.std())

    # Perpendicular unit vector (for the cross-section sample)
    perp_cos = -sin_a
    perp_sin = cos_a

    # Compute the t-range that keeps (x,y) inside the image
    eps = 1e-9
    t_candidates: list[float] = []
    if abs(cos_a) > eps:
        t_candidates += [(-cx) / cos_a, (w_img - 1 - cx) / cos_a]
    if abs(sin_a) > eps:
        t_candidates += [(-cy) / sin_a, (h_img - 1 - cy) / sin_a]
    diag = math.sqrt(w_img ** 2 + h_img ** 2)
    t_lo = max(min(t_candidates) if t_candidates else -diag, -diag)
    t_hi = min(max(t_candidates) if t_candidates else diag,  diag)

    # Vectorised strip sampling: build a (n_t × n_s) index grid covering all
    # positions along the streak axis at once, then take the per-row max.
    # Using max-per-strip (rather than mean) lets a 1–2 px wide streak clear
    # the threshold even when the bbox centre is offset from the streak axis.
    _t = np.arange(t_lo, t_hi, 1.0)
    if _t.size == 0:
        logger.debug("extend_obb: t-range empty — OBB unchanged")
        return dict(obb)

    _s = np.arange(-sample_halfwidth, sample_halfwidth + 1)  # (n_s,)
    _xc = cx + _t * cos_a  # (n_t,)
    _yc = cy + _t * sin_a  # (n_t,)

    _xi = np.clip(
        np.round(_xc[:, None] + _s[None, :] * perp_cos).astype(np.intp),
        0, w_img - 1,
    )  # (n_t, n_s)
    _yi = np.clip(
        np.round(_yc[:, None] + _s[None, :] * perp_sin).astype(np.intp),
        0, h_img - 1,
    )  # (n_t, n_s)

    _strip_max = gray[_yi, _xi].max(axis=1)  # (n_t,)
    bright_idx = np.where(_strip_max > threshold)[0]

    if bright_idx.size == 0:
        logger.debug("extend_obb: no bright pixels found along axis — OBB unchanged")
        return dict(obb)

    # Group bright t values into contiguous runs (gap tolerance = 5 px).
    # bright_idx indexes into _t which steps by 1 px, so an index gap equals
    # a t gap.  Vectorised: find where consecutive index gaps exceed tolerance,
    # then build run (start, end) pairs without a Python loop.
    # Then select the run that contains t=0 (the OBB centre is guaranteed to
    # lie on the streak), falling back to the longest run if t=0 is not bright.
    # This prevents isolated noise spikes beyond the streak from inflating w.
    gap_tolerance = 5.0
    if bright_idx.size == 1:
        runs: list[tuple[float, float]] = [(float(_t[bright_idx[0]]), float(_t[bright_idx[0]]))]
    else:
        _gap_mask = np.diff(bright_idx) > gap_tolerance          # (n-1,)
        _rs = np.concatenate([[0], np.where(_gap_mask)[0] + 1])  # run-start positions
        _re = np.concatenate([np.where(_gap_mask)[0], [len(bright_idx) - 1]])  # run-end
        runs = [
            (float(_t[bright_idx[int(s)]]), float(_t[bright_idx[int(e)]]))
            for s, e in zip(_rs, _re)
        ]

    # Prefer the run that straddles t=0 (OBB centre on the streak)
    centre_runs = [(s, e) for s, e in runs if s <= 0.0 <= e]
    if centre_runs:
        t_start, t_end = centre_runs[0]
    else:
        t_start, t_end = max(runs, key=lambda r: r[1] - r[0])

    new_w   = t_end - t_start
    if new_w < float(obb["w"]):
        logger.debug(
            "extend_obb: candidate run would shrink OBB %.0f→%.0f px; keeping original",
            obb["w"], new_w,
        )
        return dict(obb)

    new_cx  = cx + (t_start + t_end) / 2.0 * cos_a
    new_cy  = cy + (t_start + t_end) / 2.0 * sin_a

    logger.debug(
        "extend_obb: t=[%.0f, %.0f]  old_w=%.0f→new_w=%.0f  "
        "cx=(%.1f→%.1f)  cy=(%.1f→%.1f)",
        t_start, t_end, obb["w"], new_w, cx, new_cx, cy, new_cy,
    )
    return {**obb, "cx": new_cx, "cy": new_cy, "w": new_w}


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


def _rotated_iom(obb_a: dict, obb_b: dict) -> float:
    """Compute intersection-over-minimum for two OBBs using Shapely.

    Unlike IoU, IoM is robust to partial detections and perpendicular offsets
    on thin elongated streaks.  A short detection that is fully contained within
    a longer detection of the same streak scores IoM ≈ 1.0 but IoU ≈ 0.2 — IoU
    would wrongly split them into different streak groups.

    Args:
        obb_a: First OBB dict {cx, cy, w, h, angle_deg}.
        obb_b: Second OBB dict {cx, cy, w, h, angle_deg}.

    Returns:
        intersection / min(area_a, area_b) in [0, 1].  Returns 0.0 if either
        polygon is degenerate or has zero area.
    """
    try:
        poly_a = _obb_to_polygon(obb_a)
        poly_b = _obb_to_polygon(obb_b)
        if not poly_a.is_valid or not poly_b.is_valid:
            return 0.0
        min_area = min(poly_a.area, poly_b.area)
        if min_area <= 0.0:
            return 0.0
        intersection = poly_a.intersection(poly_b).area
        return float(intersection / min_area)
    except Exception as exc:  # pragma: no cover
        logger.debug("rotated IoM failed: %s", exc)
        return 0.0


def _angle_delta_deg(a: float, b: float) -> float:
    """Return the smallest angle difference for 180°-symmetric streaks."""
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def _interval_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Return the gap between two 1-D intervals, or 0 when they overlap."""
    return max(0.0, max(min(a0, a1), min(b0, b1)) - min(max(a0, a1), max(b0, b1)))


def _collinear_streak_match(
    obb_a: dict,
    obb_b: dict,
    angle_threshold_deg: float = 15.0,
    perpendicular_threshold_px: float = 40.0,
    max_gap_px: float = 900.0,
) -> bool:
    """Return True when two non-overlapping OBBs look like same-line fragments."""
    if _angle_delta_deg(obb_a.get("angle_deg", 0.0), obb_b.get("angle_deg", 0.0)) > angle_threshold_deg:
        return False

    theta = math.radians(float(obb_a.get("angle_deg", 0.0)))
    cos_a = math.cos(theta)
    sin_a = math.sin(theta)
    perp_x = -sin_a
    perp_y = cos_a

    dx = float(obb_b["cx"]) - float(obb_a["cx"])
    dy = float(obb_b["cy"]) - float(obb_a["cy"])
    perpendicular_distance = abs(dx * perp_x + dy * perp_y)
    if perpendicular_distance > perpendicular_threshold_px:
        return False

    t_b = dx * cos_a + dy * sin_a
    half_a = float(obb_a.get("w", 0.0)) / 2.0
    half_b = float(obb_b.get("w", 0.0)) / 2.0
    gap = _interval_gap(-half_a, half_a, t_b - half_b, t_b + half_b)
    dynamic_gap = min(
        max_gap_px,
        max(120.0, 2.0 * max(float(obb_a.get("w", 0.0)), float(obb_b.get("w", 0.0)))),
    )
    return gap <= dynamic_gap


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


def group_detections(
    detections: list[dict],
    iou_threshold: float = 0.5,
    iom_threshold: float = 0.3,
) -> list[dict]:
    """Group overlapping detections by streak without suppressing any.

    Detections whose OBBs overlap above *iou_threshold* (IoU) **or**
    *iom_threshold* (IoMin = intersection / min-area) are assigned the same
    ``streak_id`` (1-based int).  All detections are returned — nothing is
    suppressed — so callers can display one row per (streak, method) pair.

    IoMin is the primary signal here.  For thin elongated streaks, IoU is a
    poor match metric: a 3 px perpendicular offset on a 5×500 px streak drops
    IoU to ~0.25, far below any useful threshold.  IoMin instead asks "does
    30 % of the smaller detection lie inside the larger?" — robust to partial
    detections and small lateral offsets between methods.

    Within each group, detections are ordered by confidence descending.
    Groups themselves are ordered by the confidence of their best detection.

    Args:
        detections: Detection dicts, each with 'obb' and 'confidence' keys.
        iou_threshold: OBB IoU threshold above which two detections are
            considered the same streak (kept for same-size detection pairs).
        iom_threshold: OBB IoMin threshold above which two detections are
            considered the same streak (handles partial/offset detections).

    Returns:
        All input detections with a 'streak_id' int field added.
    """
    if not detections:
        return []

    # Work in confidence-descending order so the highest-confidence detection
    # is the "seed" of each group.
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    n = len(sorted_dets)
    group_id = [-1] * n

    next_id = 0
    for i in range(n):
        if group_id[i] == -1:
            group_id[i] = next_id
            next_id += 1
        else:
            # Already assigned by a higher-confidence seed — don't extend the
            # group further from here, which would cause chain-linking across
            # unrelated detections.
            continue
        obb_i = sorted_dets[i].get("obb")
        if obb_i is None:
            continue
        for j in range(i + 1, n):
            if group_id[j] != -1:
                continue
            obb_j = sorted_dets[j].get("obb")
            if obb_j is None:
                continue
            if (
                _rotated_iou(obb_i, obb_j) > iou_threshold
                or _rotated_iom(obb_i, obb_j) > iom_threshold
                or _collinear_streak_match(obb_i, obb_j)
            ):
                group_id[j] = group_id[i]

    for det, gid in zip(sorted_dets, group_id):
        det["streak_id"] = gid + 1  # 1-based for display

    # Sort by group (ascending) then by confidence within the group (descending)
    sorted_dets.sort(key=lambda d: (d["streak_id"], -d.get("confidence", 0.0)))

    logger.debug(
        "group_detections: %d detection(s) → %d streak group(s)",
        n, next_id,
    )
    return sorted_dets


def fuse_group_geometries(detections: list[dict]) -> list[dict]:
    """Fuse each grouped streak's fragment OBBs into one endpoint-spanning OBB.

    ``group_detections`` assigns a shared ``streak_id`` to detections that
    likely belong to the same physical streak. This helper makes the group draw
    as one streak by projecting every member's endpoints onto the longest
    member's axis, then writing the fused OBB back to every member in the group.
    Single-member groups are left unchanged.

    Args:
        detections: Detection dicts with ``streak_id``, ``obb``, and confidence.

    Returns:
        The same list with grouped OBB geometry updated in place.
    """
    groups: dict[object, list[dict]] = {}
    for det in detections:
        if det.get("obb") is None:
            continue
        groups.setdefault(det.get("streak_id"), []).append(det)

    for group in groups.values():
        if len(group) < 2:
            continue

        primary = max(
            group,
            key=lambda d: (float((d.get("obb") or {}).get("w", 0.0)), d.get("confidence", 0.0)),
        )
        base = primary["obb"]
        theta = math.radians(float(base["angle_deg"]))
        cos_a = math.cos(theta)
        sin_a = math.sin(theta)
        perp_x = -sin_a
        perp_y = cos_a
        origin_x = float(base["cx"])
        origin_y = float(base["cy"])

        t_values: list[float] = []
        perp_offsets: list[float] = []
        widths: list[float] = []
        for det in group:
            obb = det["obb"]
            cx = float(obb["cx"])
            cy = float(obb["cy"])
            half = float(obb["w"]) / 2.0
            dx = cx - origin_x
            dy = cy - origin_y
            centre_t = dx * cos_a + dy * sin_a
            perp_offsets.append(dx * perp_x + dy * perp_y)
            t_values.extend([centre_t - half, centre_t + half])
            widths.append(float(obb.get("h", 0.0)))

        t_start = min(t_values)
        t_end = max(t_values)
        centre_t = (t_start + t_end) / 2.0
        centre_perp = float(np.median(perp_offsets)) if perp_offsets else 0.0
        fused = {
            **base,
            "cx": origin_x + centre_t * cos_a + centre_perp * perp_x,
            "cy": origin_y + centre_t * sin_a + centre_perp * perp_y,
            "w": max(float(base["w"]), t_end - t_start),
            "h": max(float(np.median(widths)) if widths else float(base["h"]), 3.0),
        }

        for det in group:
            det["obb"] = dict(fused)
            det["streak_length_px"] = float(fused["w"])

    return detections


# ---------------------------------------------------------------------------
# Detection quality classification
# ---------------------------------------------------------------------------

# Quality flag constants (mirrors SkyTrack convention: 0 = good)
QUALITY_GOOD        = 0  # passes all checks
QUALITY_EDGE        = 1  # streak centre or tip within edge_margin_px of image border
QUALITY_LOW_CONF    = 2  # DINO detection confidence below threshold
QUALITY_TOO_SHORT   = 3  # streak length below minimum
QUALITY_NO_WCS      = 4  # no WCS plate solution — sky coords unavailable


def classify_detection_quality(
    det: dict,
    image_shape: tuple[int, int],
    edge_margin_px: int = 20,
    min_confidence: float = 0.30,
    min_length_px: float = 50.0,
) -> int:
    """Assign a quality flag to a detection, mirroring SkyTrack's reject codes.

    Checks are applied in priority order; the first failing check wins.
    Flag 0 means the detection passes all checks.

    # Source: SkyTrack (colleague) — StreakProcess reject flags
    # Ref: examples/streak_live.inc, line 295–323

    Args:
        det: Detection dict with keys: obb, confidence, streak_length_px,
             ra_tip1_deg, dec_tip1_deg, ra_tip2_deg, dec_tip2_deg.
        image_shape: (height, width) of the source image in pixels.
        edge_margin_px: Pixels from any edge that counts as "on the edge".
        min_confidence: DINO confidence threshold below which flag 2 is set.
        min_length_px: Minimum streak length in pixels; shorter → flag 3.

    Returns:
        Integer quality flag (0–4).  See QUALITY_* constants.
    """
    h_img, w_img = image_shape
    obb = det.get("obb") or {}

    # --- Edge check (flag 1) -------------------------------------------------
    cx = float(obb.get("cx", 0))
    cy = float(obb.get("cy", 0))
    half = float(obb.get("w", 0)) / 2.0
    angle_rad = math.radians(float(obb.get("angle_deg", 0)))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    tip1_x = cx - half * cos_a
    tip1_y = cy - half * sin_a
    tip2_x = cx + half * cos_a
    tip2_y = cy + half * sin_a

    m = edge_margin_px
    edge_contacts: list[str] = []
    for px, py in ((tip1_x, tip1_y), (tip2_x, tip2_y)):
        if px < m:
            edge_contacts.append("left")
        if px > w_img - m:
            edge_contacts.append("right")
        if py < m:
            edge_contacts.append("top")
        if py > h_img - m:
            edge_contacts.append("bottom")

    if edge_contacts:
        det["edge_clipped"] = True
        det["edge_contacts"] = sorted(set(edge_contacts))
        return QUALITY_EDGE

    det["edge_clipped"] = False
    det["edge_contacts"] = []

    # --- Low confidence (flag 2) ---------------------------------------------
    if float(det.get("confidence", 1.0)) < min_confidence:
        return QUALITY_LOW_CONF

    # --- Too short (flag 3) --------------------------------------------------
    if float(det.get("streak_length_px") or 0.0) < min_length_px:
        return QUALITY_TOO_SHORT

    # --- No WCS (flag 4) -----------------------------------------------------
    has_sky = (
        det.get("ra_tip1_deg") is not None
        or det.get("ra_tip2_deg") is not None
    )
    if not has_sky:
        return QUALITY_NO_WCS

    return QUALITY_GOOD


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
