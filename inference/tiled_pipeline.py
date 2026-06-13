"""Tiled inference for ARGUS satellite streak detection.

Large frames can contain streaks that are either too large (BrentImages, 440–1450 px
native) or too small (Frigate, 20–80 px native) for the whole-image inference path.
This module tiles each frame with a configurable native_tile_size, optionally
upsamples each crop to ``model_input_size`` before inference, remaps tile
detections back to full-image coordinates (accounting for the magnification), and
merges cross-tile duplicates with NMS.

Key parameters:

* ``native_tile_size`` — crop footprint in **source-image pixels**.
* ``model_input_size`` — resize target passed to the model (always 400 for current
  checkpoints).  ``magnification = model_input_size / native_tile_size``.

For Frigate (streaks ~40 px native, target ~150 px at model input)::

    native_tile_size ≈ 400 × (40 / 150) ≈ 107 px  →  use 110 px
    magnification    ≈ 400 / 110 ≈ 3.6×

The ``tile_size`` and ``image_size`` parameters on ``run_tiled_inference`` are
deprecated aliases for ``native_tile_size`` and ``model_input_size`` respectively.
They are kept for backward-compatibility and will be removed in a future release.

Source: adaptive_tiling_plan.md §2 — decoupled native/model tile sizes
Ref: docs/adaptive_tiling_plan.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger(__name__)

Prediction = dict[str, Any]


def select_tile_params(
    image_shape: tuple[int, int],
    expected_streak_native_px: float,
    target_streak_model_px: float = 150.0,
    model_input_size: int = 400,
    min_native_tile: int = 80,
    max_downscale: float = 3.0,
) -> dict[str, Any]:
    """Choose ``native_tile_size`` and ``overlap`` for adaptive tiling.

    The native tile size is computed so that a streak of
    ``expected_streak_native_px`` pixels in the source image appears at
    ``target_streak_model_px`` pixels at model input after resizing the crop.

    Args:
        image_shape: ``(H, W)`` of the source image in native pixels.
        expected_streak_native_px: Estimated streak length in source pixels.
        target_streak_model_px: Desired streak size at model input (px).
            Default 150 px sits comfortably inside the model's training range.
        model_input_size: Resize target for model input (always 400 for current
            checkpoints).
        min_native_tile: Floor on native tile size to prevent absurdly small
            crops that degrade in quality when upsampled.
        max_downscale: Ceiling expressed as a downscale factor relative to
            ``model_input_size``.  A value of 3.0 means the largest allowed
            native tile is ``3 × model_input_size`` pixels.

    Returns:
        Dict with keys ``"native_tile_size"`` (``int``) and
        ``"overlap"`` (``float``).  The overlap is chosen so that the tile
        stride never exceeds one estimated max-streak length, preventing a
        streak from falling entirely between tile boundaries.

    Example:
        >>> select_tile_params((1555, 2325), 40.0)
        {'native_tile_size': 107, 'overlap': 0.5}

    Source: adaptive_tiling_plan.md §2.2 — automatic parameter selection
    Ref: docs/adaptive_tiling_plan.md
    """
    native_tile_size = int(
        model_input_size * expected_streak_native_px / target_streak_model_px
    )
    native_tile_size = max(native_tile_size, min_native_tile)
    native_tile_size = min(native_tile_size, int(model_input_size * max_downscale))

    # Overlap must cover at least one max-streak length so no streak is missed.
    max_streak_native = expected_streak_native_px * 3.0  # conservative upper bound
    min_overlap_frac = max_streak_native / native_tile_size
    overlap = float(min(0.5, max(0.25, min_overlap_frac)))

    return {"native_tile_size": native_tile_size, "overlap": overlap}


_INTERP_CODES: dict[str, int] = {
    # Maps user-facing names to OpenCV interpolation flag integers (stable constants).
    "nearest":  0,   # cv2.INTER_NEAREST
    "bilinear": 1,   # cv2.INTER_LINEAR  — default, good up to ~4× upscale
    "cubic":    2,   # cv2.INTER_CUBIC
    "lanczos":  4,   # cv2.INTER_LANCZOS4 — highest quality, ~2× slower than bilinear
}


def tile_image(
    img_array: "np.ndarray",
    tile_size: int,
    overlap: float,
    resize_to: int | None = None,
    interp: str = "bilinear",
) -> Iterator[tuple["np.ndarray", int, int]]:
    """Yield square, padded image tiles that cover the full image.

    When ``resize_to`` is set and differs from ``tile_size``, each crop is
    resized to ``resize_to × resize_to`` pixels before yielding.  The
    ``(x0, y0)`` offsets are **always in native source-image pixels**,
    regardless of resize; callers must scale bounding boxes by
    ``1 / (resize_to / tile_size)`` when remapping back to native coordinates.

    Args:
        img_array: Image array with shape ``(H, W)`` or ``(H, W, C)``.
        tile_size: Native crop edge length in source-image pixels.
        overlap: Fractional overlap between neighboring tiles in ``[0, 1)``.
        resize_to: If set, resize each crop to this size before yielding.
            Use ``model_input_size`` (e.g. 400) when tiling at a smaller
            native scale for upsampling magnification.
        interp: Interpolation method for resize, one of ``"bilinear"``
            (default), ``"lanczos"``, ``"cubic"``, ``"nearest"``.
            ``"lanczos"`` preserves more high-frequency edge detail for the
            ViT backbone at the cost of ~2× slower resize; see
            docs/adaptive_tiling_plan.md §4 Q3.

    Yields:
        Tuples of ``(tile_array, x0, y0)`` where ``x0``/``y0`` are the tile's
        top-left coordinate in the **original** image coordinate system.

    Raises:
        ValueError: If tile parameters or interpolation name are invalid.
    """
    import numpy as np

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in the range [0, 1)")
    if img_array.ndim not in {2, 3}:
        raise ValueError("img_array must have shape (H, W) or (H, W, C)")
    if interp not in _INTERP_CODES:
        raise ValueError(
            f"interp must be one of {list(_INTERP_CODES)}; got {interp!r}"
        )

    h_img, w_img = img_array.shape[:2]
    if h_img <= 0 or w_img <= 0:
        raise ValueError("img_array must be non-empty")

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    x_starts = _tile_starts(w_img, tile_size, stride)
    y_starts = _tile_starts(h_img, tile_size, stride)

    padded_h = y_starts[-1] + tile_size
    padded_w = x_starts[-1] + tile_size
    pad_h = max(0, padded_h - h_img)
    pad_w = max(0, padded_w - w_img)
    pad_spec = ((0, pad_h), (0, pad_w))
    if img_array.ndim == 3:
        pad_spec = (*pad_spec, (0, 0))
    padded = np.pad(img_array, pad_spec, mode="edge")

    do_resize = resize_to is not None and resize_to != tile_size
    interp_code = _INTERP_CODES[interp]

    for y0 in y_starts:
        for x0 in x_starts:
            crop = padded[y0:y0 + tile_size, x0:x0 + tile_size].copy()
            if do_resize:
                import cv2  # lazy import — only needed when magnification ≠ 1
                crop = cv2.resize(crop, (resize_to, resize_to),
                                  interpolation=interp_code)
            yield crop, x0, y0


def remap_predictions(
    preds: list[Prediction],
    x0: int,
    y0: int,
    magnification: float = 1.0,
) -> list[Prediction]:
    """Add tile offsets to ``[x, y, w, h]`` prediction boxes.

    When the tile was upsampled before inference (``magnification > 1``),
    bounding boxes returned by the model are in *resized-tile* coordinates and
    must be divided by the magnification before the tile offset is applied.

    Args:
        preds: Predictions with COCO-style ``bbox`` values ``[x, y, w, h]``
            in the tile's model-input coordinate system.
        x0: Tile X offset in the full image (native source pixels).
        y0: Tile Y offset in the full image (native source pixels).
        magnification: ``model_input_size / native_tile_size``.  Pass a value
            other than 1.0 when the tile was resized before inference.

    Returns:
        New prediction dictionaries in full-image native-pixel coordinates.

    Source: adaptive_tiling_plan.md §2.4 — prediction coordinate remapping
    Ref: docs/adaptive_tiling_plan.md
    """
    remapped: list[Prediction] = []
    for pred in preds:
        x, y, w, h = [float(v) for v in pred["bbox"]]
        if magnification != 1.0:
            # Step 1: tile model coords → native tile coords
            x /= magnification
            y /= magnification
            w /= magnification
            h /= magnification
        # Step 2: native tile coords → full-image coords
        updated = dict(pred)
        updated["bbox"] = [x + float(x0), y + float(y0), w, h]
        remapped.append(updated)
    return remapped


def nms_predictions(
    preds: list[Prediction],
    iou_threshold: float = 0.4,
) -> list[Prediction]:
    """Apply cross-tile IoU NMS to COCO-style predictions.

    Args:
        preds: Predictions with keys ``bbox`` (``[x, y, w, h]``), ``score``,
            and ``category_id``.
        iou_threshold: IoU threshold above which lower-scored boxes are
            suppressed, independently per category.

    Returns:
        Predictions kept after NMS, sorted by descending score.
    """
    if not preds:
        return []
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the range [0, 1]")

    try:
        keep_indices = _torchvision_nms(preds, iou_threshold)
    except Exception as exc:
        logger.debug("torchvision NMS unavailable; using NumPy fallback: %s", exc)
        keep_indices = _numpy_nms(preds, iou_threshold)

    return [preds[i] for i in keep_indices]


def stitch_collinear_fragments(
    preds: list[Prediction],
    max_gap_px: float = 200.0,
    max_perp_ratio: float = 0.8,
    angle_tol_deg: float = 10.0,
    min_aspect_ratio: float = 1.5,
    max_growth_ratio: float = 3.0,
    conf_floor: float = 0.5,
) -> list[Prediction]:
    """Merge non-overlapping collinear streak fragments from multi-tile inference.

    After NMS, long streaks that span multiple tiles appear as 2–3 separate
    detections with no bounding-box overlap.  IoU NMS cannot merge them.
    This function identifies such fragments by checking:

    1. Both boxes have a similar aspect-ratio-derived streak angle
       (within ``angle_tol_deg``).
    2. The center-to-center vector is approximately aligned with that shared
       angle (within ``angle_tol_deg``).
    3. The gap between the projected extents along the streak direction is
       ≤ ``max_gap_px``.
    4. The perpendicular center offset is ≤ ``max_perp_ratio`` × the average
       transverse box width.
    5. The merged span would be ≤ ``max_growth_ratio`` × the longer input
       fragment.  This is enforced TWICE: pairwise when building edges, and
       again as a GLOBAL cap when collapsing each connected component (see
       below) so transitive A–B–C chains cannot blow past the ratio.

    Only fragments with ``score ≥ conf_floor`` participate in stitching; lower-
    confidence fragments pass through unchanged (they do not seed or extend
    chains, which is what previously let background noise inflate a clean streak
    into a frame-spanning blob).  Boxes with an axis-aligned aspect ratio below
    ``min_aspect_ratio`` are likewise excluded (near-square boxes have ambiguous
    angle).

    Connected components are collapsed by a **guarded greedy merge**: fragments
    are ordered along the shared streak axis and merged one at a time only while
    the running merged span stays ≤ ``max_growth_ratio`` × the longest original
    fragment in the chain.  A fragment that would breach the cap starts a new
    sub-detection instead of being absorbed.  This is the fix for the union-find
    transitivity bug where pairwise-legal merges chained into 4–5× blow-ups.

    Args:
        preds: Post-NMS predictions in full-image coordinates with
            ``bbox`` in ``[x, y, w, h]`` format.
        max_gap_px: Maximum gap (native source pixels) between two boxes along
            the streak direction.  Set to ``native_tile_size`` for BrentImages-
            style long streaks; smaller for dense-field images.
        max_perp_ratio: Maximum perpendicular center offset expressed as a
            fraction of the average transverse box width.  0.5 allows up to
            half-a-box-width of lateral displacement.
        angle_tol_deg: Angular tolerance in degrees for both the box-angle
            agreement check and the center-direction alignment check.
        min_aspect_ratio: Minimum axis-aligned aspect ratio (longer / shorter
            side) to consider a box as a streak candidate.  Boxes below this
            threshold are returned unchanged without stitching.
        max_growth_ratio: Maximum ratio of merged span to the longest original
            fragment's span.  Default 3.0 blocks merges that would create a
            detection more than 3× longer than any input.  Enforced both
            pairwise and globally per connected component.
        conf_floor: Minimum ``score`` for a fragment to participate in
            stitching.  Lower-confidence fragments pass through unchanged so
            background-noise fragments cannot seed or extend a chain.

    Returns:
        Predictions after merging collinear fragments, sorted by descending
        score.  Singletons (no compatible pair found) pass through unchanged.

    Source: adaptive_tiling_plan.md §4 Q2 — long-streak fragmentation
    Ref: docs/adaptive_tiling_plan.md
    """
    import math

    if len(preds) < 2:
        return list(preds)

    angle_tol = math.radians(angle_tol_deg)

    def _center(pred: Prediction) -> tuple[float, float]:
        obb = pred.get("obb")
        if isinstance(obb, dict) and "cx" in obb and "cy" in obb:
            return float(obb["cx"]), float(obb["cy"])
        x, y, w, h = pred["bbox"]
        return x + w / 2.0, y + h / 2.0

    def _angle(pred: Prediction) -> float:
        """Streak angle in [0, π/2].

        Prefers the ``obb.angle_deg`` field when present so that diagonal
        streaks are correctly characterised.  The axis-aligned bbox fallback
        returns atan2(h, w) which is valid for axis-aligned detections only.
        """
        obb = pred.get("obb")
        if isinstance(obb, dict) and "angle_deg" in obb:
            a = float(obb["angle_deg"]) % 180.0
            return math.radians(min(a, 180.0 - a))  # fold to [0, π/2]
        _, _, w, h = pred["bbox"]
        return math.atan2(max(h, 1e-6), max(w, 1e-6))

    def _aspect(pred: Prediction) -> float:
        """Axis-ratio of the detection.

        Prefers the ``obb`` field (w/h) so that diagonal streaks with near-
        square axis-aligned bboxes are not incorrectly filtered out.
        """
        obb = pred.get("obb")
        if isinstance(obb, dict):
            w_obb = float(obb.get("w", 0))
            h_obb = float(obb.get("h", 0))
            if w_obb > 0 and h_obb > 0:
                return max(w_obb, h_obb) / max(min(w_obb, h_obb), 1e-6)
        _, _, w, h = pred["bbox"]
        lo = min(w, h)
        hi = max(w, h)
        return hi / max(lo, 1e-6)

    def _merge_obb(
        obb_a: dict[str, float],
        obb_b: dict[str, float],
    ) -> dict[str, float]:
        """Return a proper oriented OBB spanning both collinear fragment OBBs.

        Projects each fragment's extent onto the shared streak axis, takes
        the full range along that axis, and uses a length-weighted average
        for the perpendicular centre position.  Width (minor axis) is the
        max of the two fragments — conservative, never narrows the detection.
        """
        angle = math.radians(
            (obb_a["angle_deg"] + obb_b["angle_deg"]) / 2.0
        )
        cos_t = math.cos(angle)
        sin_t = math.sin(angle)

        proj_a = obb_a["cx"] * cos_t + obb_a["cy"] * sin_t
        proj_b = obb_b["cx"] * cos_t + obb_b["cy"] * sin_t
        half_a = obb_a["w"] / 2.0
        half_b = obb_b["w"] / 2.0

        start = min(proj_a - half_a, proj_b - half_b)
        end   = max(proj_a + half_a, proj_b + half_b)
        center_proj = (start + end) / 2.0

        perp_a = -obb_a["cx"] * sin_t + obb_a["cy"] * cos_t
        perp_b = -obb_b["cx"] * sin_t + obb_b["cy"] * cos_t
        len_a, len_b = max(obb_a["w"], 1e-6), max(obb_b["w"], 1e-6)
        center_perp = (perp_a * len_a + perp_b * len_b) / (len_a + len_b)

        cx = center_proj * cos_t - center_perp * sin_t
        cy = center_proj * sin_t + center_perp * cos_t

        return {
            "cx": cx,
            "cy": cy,
            "w": end - start,
            "h": max(obb_a["h"], obb_b["h"]),
            "angle_deg": (obb_a["angle_deg"] + obb_b["angle_deg"]) / 2.0,
        }

    def _merge(pA: Prediction, pB: Prediction) -> Prediction:
        """Return the union bounding box with the max score and merged OBB."""
        xA, yA, wA, hA = pA["bbox"]
        xB, yB, wB, hB = pB["bbox"]
        x1 = min(xA, xB)
        y1 = min(yA, yB)
        x2 = max(xA + wA, xB + wB)
        y2 = max(yA + hA, yB + hB)
        merged = dict(pA)
        merged["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        merged["score"] = max(
            float(pA.get("score", 0.0)),
            float(pB.get("score", 0.0)),
        )
        obb_a = pA.get("obb")
        obb_b = pB.get("obb")
        if isinstance(obb_a, dict) and isinstance(obb_b, dict):
            import math as _math
            merged["obb"] = _merge_obb(obb_a, obb_b)
            obb = merged["obb"]
            r = _math.radians(obb["angle_deg"])
            merged["x1"] = obb["cx"] - obb["w"] / 2 * _math.cos(r)
            merged["y1"] = obb["cy"] - obb["w"] / 2 * _math.sin(r)
            merged["x2"] = obb["cx"] + obb["w"] / 2 * _math.cos(r)
            merged["y2"] = obb["cy"] + obb["w"] / 2 * _math.sin(r)
            merged["streak_length_px"] = obb["w"]
        return merged

    angles = [_angle(p) for p in preds]
    aspects = [_aspect(p) for p in preds]
    centers = [_center(p) for p in preds]
    scores = [float(p.get("score", p.get("confidence", 0.0))) for p in preds]
    n = len(preds)

    # A fragment is eligible to stitch only if it is streak-shaped AND confident
    # enough.  Ineligible fragments form no edges and pass through as singletons.
    eligible = [aspects[i] >= min_aspect_ratio and scores[i] >= conf_floor for i in range(n)]

    edges: list[tuple[int, int]] = []
    for i in range(n):
        if not eligible[i]:
            continue
        for j in range(i + 1, n):
            if not eligible[j]:
                continue

            theta_i = angles[i]
            theta_j = angles[j]

            # 1. Both boxes must have similar streak angle.
            if abs(theta_i - theta_j) > angle_tol:
                continue

            theta_avg = (theta_i + theta_j) / 2.0
            cos_t = math.cos(theta_avg)
            sin_t = math.sin(theta_avg)

            cx_i, cy_i = centers[i]
            cx_j, cy_j = centers[j]
            dx = cx_j - cx_i
            dy = cy_j - cy_i
            dist = math.hypot(dx, dy)
            if dist < 1.0:
                continue  # identical centres; NMS would already merge these

            # 2. Center-to-center direction must be ≈ streak direction.
            #    Map to [0, π/2] so both left→right and right→left are accepted.
            phi = math.atan2(abs(dy), abs(dx))
            if abs(phi - theta_avg) > angle_tol:
                continue

            # 3. Perpendicular offset of centres (projected onto v ⊥ streak).
            #    v = (−sin θ, cos θ)
            perp_offset = abs(-sin_t * dx + cos_t * dy)
            _, _, wi, hi = preds[i]["bbox"]
            _, _, wj, hj = preds[j]["bbox"]
            # Transverse width of each box = projection of box dims onto v.
            transverse_i = wi * sin_t + hi * cos_t
            transverse_j = wj * sin_t + hj * cos_t
            avg_transverse = (transverse_i + transverse_j) / 2.0
            if perp_offset > max_perp_ratio * avg_transverse:
                continue

            # 4. Gap along streak direction must be ≤ max_gap_px.
            #    Project each box's interval onto u = (cos θ, sin θ).
            proj_i = cx_i * cos_t + cy_i * sin_t
            proj_j = cx_j * cos_t + cy_j * sin_t
            # Half-extents along streak direction.
            half_i = (wi * cos_t + hi * sin_t) / 2.0
            half_j = (wj * cos_t + hj * sin_t) / 2.0
            # Ensure left box comes first for gap formula.
            if proj_i > proj_j:
                proj_i, proj_j = proj_j, proj_i
                half_i, half_j = half_j, half_i
            gap = (proj_j - half_j) - (proj_i + half_i)
            if gap > max_gap_px:
                continue

            # 5. Merged span must not grow more than max_growth_ratio × the
            #    longer input fragment — prevents short detections from being
            #    absorbed into long false-positive chains via union-find.
            merged_span = (max(proj_i + half_i, proj_j + half_j)
                           - min(proj_i - half_i, proj_j - half_j))
            if merged_span > max_growth_ratio * max(2.0 * half_i, 2.0 * half_j):
                continue

            edges.append((i, j))

    def _along_len(pred: Prediction) -> float:
        """Streak-axis length of a fragment (OBB major axis, bbox-diag fallback)."""
        obb = pred.get("obb")
        if isinstance(obb, dict) and "w" in obb:
            return float(obb["w"])
        slp = pred.get("streak_length_px")
        if slp:
            return float(slp)
        _, _, w, h = pred["bbox"]
        return math.hypot(w, h)

    def _axis_interval(pred: Prediction, cos_t: float, sin_t: float) -> tuple[float, float]:
        """Projected [lo, hi] extent of a fragment along the (cos_t, sin_t) axis."""
        obb = pred.get("obb")
        if isinstance(obb, dict) and "cx" in obb:
            c = obb["cx"] * cos_t + obb["cy"] * sin_t
            half = float(obb.get("w", 0.0)) / 2.0
        else:
            x, y, w, h = pred["bbox"]
            c = (x + w / 2.0) * cos_t + (y + h / 2.0) * sin_t
            half = (abs(w * cos_t) + abs(h * sin_t)) / 2.0
        return c - half, c + half

    # Collapse each connected component with a GUARDED GREEDY MERGE.  Union-find
    # only tells us which fragments are pairwise-compatible; merging them all
    # unconditionally is what let A–B–C chains blow past max_growth_ratio.  We
    # instead order each component along its mean streak axis and merge one
    # fragment at a time, starting a fresh sub-detection whenever the next
    # fragment is separated by more than max_gap_px along the axis, OR the single
    # merge step would more than max_growth_ratio× the running detection.  Both
    # guards together stop transitive noise chains while still allowing a genuine
    # long streak to assemble from many small gap-respecting collinear fragments.
    groups = _union_find_components(n, edges)
    result: list[Prediction] = []
    for group in groups:
        if len(group) == 1:
            result.append(preds[group[0]])
            continue
        theta = sum(angles[i] for i in group) / len(group)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        ordered = sorted(group, key=lambda i: centers[i][0] * cos_t + centers[i][1] * sin_t)
        cur = preds[ordered[0]]
        for k in ordered[1:]:
            cand = preds[k]
            lo_c, hi_c = _axis_interval(cur, cos_t, sin_t)
            lo_k, hi_k = _axis_interval(cand, cos_t, sin_t)
            gap = max(lo_k - hi_c, lo_c - hi_k)        # >0 only when disjoint
            trial = _merge(cur, cand)
            trial_span = float(trial.get("streak_length_px") or _along_len(trial))
            if gap <= max_gap_px and trial_span <= max_growth_ratio * _along_len(cur):
                cur = trial
            else:
                result.append(cur)            # gap/growth breached — emit, restart
                cur = cand
        result.append(cur)

    result.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
    return result


def run_tiled_inference(
    image_path: str | Path,
    model_config: str | Path,
    checkpoint: str | Path,
    native_tile_size: int = 400,
    model_input_size: int = 400,
    overlap: float = 0.5,
    conf_threshold: float = 0.2,
    interp: str = "bilinear",
    stitch: bool = False,
    stitch_max_gap_px: float = 400.0,
    # Deprecated aliases — kept for backward compatibility.
    tile_size: int | None = None,
    image_size: int | None = None,
) -> list[Prediction]:
    """Run DINO inference over native-resolution overlapping tiles.

    The ``native_tile_size`` controls the crop footprint in source pixels.
    Each crop is resized to ``model_input_size`` before inference when
    ``native_tile_size != model_input_size``, yielding a magnification of
    ``model_input_size / native_tile_size``.

    For Frigate images (20–80 px streaks) use ``native_tile_size=110`` with
    the default ``model_input_size=400`` to achieve ~3.6× magnification,
    bringing streaks into the model's training sweet-spot of 70–290 px.

    For BrentImages long streaks (440–1450 px) that span 3+ tiles, set
    ``stitch=True`` to merge non-overlapping collinear fragments after NMS.

    Args:
        image_path: FITS, PNG, or JPEG image path.
        model_config: MMDetection config path, or an ARGUS model size alias
            accepted by ``inference.pipeline._select_config``.
        checkpoint: Model checkpoint path.
        native_tile_size: Crop footprint in source-image pixels.  Use the same
            value for later tiled training annotations.
        model_input_size: Resize target before model inference (always 400 for
            current checkpoints).
        overlap: Fractional tile overlap in ``[0, 1)``.  Use the same value for
            later tiled training annotations.
        conf_threshold: Minimum model score to keep before cross-tile NMS.
        interp: Interpolation method for tile resize — ``"bilinear"`` (default)
            or ``"lanczos"`` (higher quality, ~2× slower).  See Q3 in
            docs/adaptive_tiling_plan.md for the trade-off discussion.
        stitch: If ``True``, run ``stitch_collinear_fragments()`` after NMS to
            merge non-overlapping collinear long-streak fragments.  Primarily
            useful for BrentImages-style 1450 px streaks.  Off by default.
        stitch_max_gap_px: Passed to ``stitch_collinear_fragments`` when
            ``stitch=True``.  Defaults to ``native_tile_size``.
        tile_size: **Deprecated** alias for ``native_tile_size``.
        image_size: **Deprecated** alias for ``model_input_size``.

    Returns:
        Final predictions as dictionaries with ``bbox`` in ``[x, y, w, h]``
        full-image native-pixel coordinates plus ``score`` and ``category_id``.

    Source: adaptive_tiling_plan.md §2.3 — updated run_tiled_inference signature
    Ref: docs/adaptive_tiling_plan.md
    """
    # Handle deprecated aliases.
    if tile_size is not None:
        logger.warning(
            "run_tiled_inference: 'tile_size' is deprecated; use 'native_tile_size'."
        )
        native_tile_size = tile_size
    if image_size is not None:
        logger.warning(
            "run_tiled_inference: 'image_size' is deprecated; use 'model_input_size'."
        )
        model_input_size = image_size

    magnification = model_input_size / native_tile_size

    t0 = time.perf_counter()
    image_path = Path(image_path)
    checkpoint = Path(checkpoint)

    from inference.fits_loader import FITSLoader
    from inference.pipeline import _load_model, _run_inference, _select_config

    loaded = FITSLoader().load(image_path)
    img_array = loaded["array"]

    config_path = Path(model_config)
    if not config_path.exists():
        config_path = _select_config(str(model_config))

    from inference.device import get_device
    import torch

    device = get_device()
    inference_device = device
    if device.type == "mps" and not torch.cuda.is_available():
        inference_device = torch.device("cpu")
        logger.debug("MPS device detected; forcing tiled DINO inference to CPU")

    model = _load_model(config_path, checkpoint, inference_device)

    resize_to = model_input_size if magnification != 1.0 else None
    tiles = list(
        tile_image(img_array, tile_size=native_tile_size, overlap=overlap,
                   resize_to=resize_to, interp=interp)
    )
    logger.info(
        "Running tiled inference on %s: %d tiles, native_tile_size=%d, "
        "model_input_size=%d, magnification=%.2f×, overlap=%.2f, interp=%s",
        image_path.name,
        len(tiles),
        native_tile_size,
        model_input_size,
        magnification,
        overlap,
        interp,
    )

    predictions: list[Prediction] = []
    for idx, (tile, x0, y0) in enumerate(tiles, 1):
        tile_dets = _run_inference(
            model,
            tile,
            image_size=model_input_size,
            confidence_threshold=conf_threshold,
            model_name="tiled_dino",
        )
        tile_preds = [_pipeline_det_to_prediction(det) for det in tile_dets]
        predictions.extend(remap_predictions(tile_preds, x0, y0, magnification=magnification))
        logger.debug(
            "tile %d/%d at (%d,%d): %d detections",
            idx, len(tiles), x0, y0, len(tile_preds),
        )

    after_nms = nms_predictions(predictions, iou_threshold=0.4)
    if stitch:
        n_before = len(after_nms)
        after_nms = stitch_collinear_fragments(after_nms, max_gap_px=stitch_max_gap_px)
        logger.info(
            "Collinear stitching: %d detections → %d after merging fragments",
            n_before, len(after_nms),
        )
    final = _clip_predictions_to_image(
        after_nms,
        width=img_array.shape[1],
        height=img_array.shape[0],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Tiled inference complete: %d detections in %.0f ms", len(final), elapsed_ms)
    return final


def save_visualization(
    image_path: str | Path,
    predictions: list[Prediction],
    output_dir: str | Path,
    tile_size: int,
    overlap: float,
) -> Path:
    """Save a PNG visualization with tile grid and detections drawn.

    The tile grid is drawn using the **native** tile size (source-image pixels),
    which matches the actual crop footprint regardless of ``model_input_size``.

    Args:
        image_path: Source image path.
        predictions: Full-image predictions with ``[x, y, w, h]`` boxes in
            native-pixel coordinates.
        output_dir: Directory where the visualization should be written.
        tile_size: Native tile size (source-image pixels) used for inference.
        overlap: Tile overlap used for inference.

    Returns:
        Path to the saved visualization.
    """
    import cv2
    from inference.fits_loader import FITSLoader

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = FITSLoader().load(image_path)["array"]
    vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    for x0 in _tile_starts(image.shape[1], tile_size, stride):
        cv2.line(vis, (x0, 0), (x0, image.shape[0] - 1), (80, 80, 80), 1)
        cv2.line(
            vis,
            (min(x0 + tile_size, image.shape[1] - 1), 0),
            (min(x0 + tile_size, image.shape[1] - 1), image.shape[0] - 1),
            (80, 80, 80),
            1,
        )
    for y0 in _tile_starts(image.shape[0], tile_size, stride):
        cv2.line(vis, (0, y0), (image.shape[1] - 1, y0), (80, 80, 80), 1)
        cv2.line(
            vis,
            (0, min(y0 + tile_size, image.shape[0] - 1)),
            (image.shape[1] - 1, min(y0 + tile_size, image.shape[0] - 1)),
            (80, 80, 80),
            1,
        )

    for pred in predictions:
        x, y, w, h = pred["bbox"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            vis,
            f"{float(pred.get('score', 0.0)):.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path = output_dir / f"{image_path.stem}_tiled.png"
    cv2.imwrite(str(out_path), vis)
    return out_path


def _union_find_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Return connected components as lists of node indices (union-find).

    Args:
        n: Number of nodes (indices 0..n-1).
        edges: List of ``(i, j)`` pairs indicating that nodes ``i`` and ``j``
            are in the same component.

    Returns:
        List of components, each component being a list of node indices.
    """
    parent = list(range(n))
    rank = [0] * n

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1

    for i, j in edges:
        union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return fixed-stride starts whose final tile may extend into padding."""
    import math

    if length <= tile_size:
        return [0]
    n_tiles = int(math.ceil((length - tile_size) / stride)) + 1
    return [idx * stride for idx in range(n_tiles)]


def _pipeline_det_to_prediction(det: dict[str, Any]) -> Prediction:
    """Convert existing pipeline ``[x1, y1, x2, y2]`` output to ``[x, y, w, h]``."""
    x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
    return {
        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
        "score": float(det.get("confidence", det.get("score", 0.0))),
        "category_id": int(det.get("category_id", 1)),
    }


def _xywh_to_xyxy(pred: Prediction) -> list[float]:
    """Convert one prediction bbox to ``[x1, y1, x2, y2]``."""
    x, y, w, h = [float(v) for v in pred["bbox"]]
    return [x, y, x + w, y + h]


def _torchvision_nms(preds: list[Prediction], iou_threshold: float) -> list[int]:
    """Run torchvision NMS independently per category."""
    import torch
    from torchvision.ops import nms

    keep: list[int] = []
    categories = sorted({int(pred.get("category_id", 1)) for pred in preds})
    for category in categories:
        indices = [
            i for i, pred in enumerate(preds)
            if int(pred.get("category_id", 1)) == category
        ]
        boxes = torch.tensor([_xywh_to_xyxy(preds[i]) for i in indices], dtype=torch.float32)
        scores = torch.tensor(
            [float(preds[i].get("score", 0.0)) for i in indices],
            dtype=torch.float32,
        )
        kept = nms(boxes, scores, iou_threshold).cpu().tolist()
        keep.extend(indices[int(i)] for i in kept)
    keep.sort(key=lambda i: float(preds[i].get("score", 0.0)), reverse=True)
    return keep


def _numpy_nms(preds: list[Prediction], iou_threshold: float) -> list[int]:
    """Pure-NumPy NMS fallback, independently per category."""
    import numpy as np

    keep: list[int] = []
    categories = sorted({int(pred.get("category_id", 1)) for pred in preds})
    for category in categories:
        indices = np.array(
            [i for i, pred in enumerate(preds) if int(pred.get("category_id", 1)) == category],
            dtype=np.int64,
        )
        boxes = np.array([_xywh_to_xyxy(preds[int(i)]) for i in indices], dtype=np.float32)
        scores = np.array(
            [float(preds[int(i)].get("score", 0.0)) for i in indices],
            dtype=np.float32,
        )
        order = scores.argsort()[::-1]

        while order.size > 0:
            current = int(order[0])
            keep.append(int(indices[current]))
            if order.size == 1:
                break
            ious = _box_iou_xyxy(boxes[current], boxes[order[1:]])
            order = order[1:][ious <= iou_threshold]

    keep.sort(key=lambda i: float(preds[i].get("score", 0.0)), reverse=True)
    return keep


def _box_iou_xyxy(box: "np.ndarray", boxes: "np.ndarray") -> "np.ndarray":
    """Compute IoU between one ``xyxy`` box and many ``xyxy`` boxes."""
    import numpy as np

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    union = box_area + boxes_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def _clip_predictions_to_image(
    preds: list[Prediction],
    width: int,
    height: int,
) -> list[Prediction]:
    """Clip prediction boxes to image bounds and drop degenerate boxes."""
    clipped: list[Prediction] = []
    for pred in preds:
        x, y, w, h = [float(v) for v in pred["bbox"]]
        x1 = min(max(x, 0.0), float(width))
        y1 = min(max(y, 0.0), float(height))
        x2 = min(max(x + w, 0.0), float(width))
        y2 = min(max(y + h, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        updated = dict(pred)
        updated["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        clipped.append(updated)
    return clipped


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tiled inference."""
    parser = argparse.ArgumentParser(description="Run ARGUS tiled DINO inference.")
    parser.add_argument("--image", required=True, help="Path to FITS/PNG/JPEG image")
    parser.add_argument(
        "--model-config",
        default="dinov3_vitb",
        help="MMDetection config path or ARGUS model size alias",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--native-tile-size",
        type=int,
        default=400,
        help=(
            "Native crop footprint in source-image pixels (default 400). "
            "Set to ~110 for Frigate images to achieve ~3.6× upsampling magnification."
        ),
    )
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=400,
        help="Resize target before model inference (default 400; matches current checkpoints).",
    )
    parser.add_argument("--overlap", type=float, default=0.5, help="Fractional tile overlap")
    parser.add_argument(
        "--conf-threshold", type=float, default=0.2, help="Minimum detection score"
    )
    parser.add_argument(
        "--interp",
        choices=list(_INTERP_CODES),
        default="bilinear",
        help=(
            "Interpolation method for tile resize (default: bilinear). "
            "Use 'lanczos' for higher quality at ~2× the resize cost."
        ),
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help=(
            "Run collinear-fragment stitcher after NMS to merge non-overlapping "
            "long-streak fragments (useful for BrentImages 1450 px streaks)."
        ),
    )
    parser.add_argument(
        "--stitch-max-gap",
        type=float,
        default=400.0,
        help="Max gap in native pixels between collinear fragments to merge (default 400).",
    )
    parser.add_argument(
        "--output", required=True, help="Directory for visualization and JSON output"
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def main() -> None:
    """Run the tiled inference CLI."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    predictions = run_tiled_inference(
        image_path=args.image,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        native_tile_size=args.native_tile_size,
        model_input_size=args.model_input_size,
        overlap=args.overlap,
        conf_threshold=args.conf_threshold,
        interp=args.interp,
        stitch=args.stitch,
        stitch_max_gap_px=args.stitch_max_gap,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_path = save_visualization(
        args.image, predictions, output_dir,
        tile_size=args.native_tile_size, overlap=args.overlap,
    )
    json_path = output_dir / f"{Path(args.image).stem}_tiled_predictions.json"
    json_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print(f"Detected {len(predictions)} streak candidate(s)")
    print(f"Visualization: {vis_path}")
    print(f"Predictions:   {json_path}")


if __name__ == "__main__":
    main()
