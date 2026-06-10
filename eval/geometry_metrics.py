"""Three-tier streak geometry evaluation for ARGUS.

This module is the **canonical evaluation standard** for all ARGUS models.
Every run is compared against prior runs using these metrics.

Tier 1 — Detection
    Did the model find the streak at all?  A prediction "finds" a GT streak when
    its predicted centre projects onto the GT centerline **within the segment
    bounds** and the perpendicular distance to that centreline is below
    ``perp_threshold_px``.  The strict no-buffer rule is intentional: a
    prediction whose centre lands off the end of the streak cannot be refined by
    Radon + endpoint tracing (there is no streak signal to latch onto at that
    position).

Tier 2 — Raw geometry
    For each Tier-1 matched pair: angular error (mod 180° symmetry) and mean
    endpoint error between the model's raw OBB output and the GT streak.

Tier 3 — Refined geometry
    Same geometry metrics recomputed after running Radon angle refinement and
    OBB endpoint extension on the matched predictions.  Requires pixel data
    (pass ``image_arrays``).  The improvement over Tier 2 quantifies how much
    post-processing contributes.

Usage (offline, predictions already saved):
    python -m eval.geometry_metrics \\
        --predictions results/run15_vits/t0.50/predictions.json \\
        --annotations data/annotations/val_atwood.json \\
        --output results/run15_vits/geometry_eval.json

    # With Tier 3 (needs raw FITS images):
    python -m eval.geometry_metrics \\
        --predictions results/run15_vits/t0.50/predictions.json \\
        --annotations data/annotations/val_atwood.json \\
        --images-dir data/raw \\
        --output results/run15_vits/geometry_eval.json
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Streak length band boundaries (pixels) — same as eval/metrics.py
_SHORT_MAX = 150.0
_LONG_MIN = 400.0

# Default perpendicular-distance threshold for Tier-1 match
DEFAULT_PERP_THRESHOLD_PX = 10.0


# ---------------------------------------------------------------------------
# OBB geometry helpers
# ---------------------------------------------------------------------------

def _centerline_endpoints(obb: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (p1, p2) pixel endpoints of the streak centerline.

    The streak runs along the *w* axis of the OBB.  h is the cross-track width.

    Args:
        obb: OBB dict with keys cx, cy, w, h, angle_deg.

    Returns:
        Tuple (p1, p2) each an (x, y) float array.
    """
    cx = float(obb["cx"])
    cy = float(obb["cy"])
    half_w = float(obb["w"]) / 2.0
    angle_rad = math.radians(float(obb["angle_deg"]))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    p1 = np.array([cx - half_w * cos_a, cy - half_w * sin_a])
    p2 = np.array([cx + half_w * cos_a, cy + half_w * sin_a])
    return p1, p2


def _point_to_segment(
    point: np.ndarray,
    seg_p1: np.ndarray,
    seg_p2: np.ndarray,
) -> tuple[float, float]:
    """Perpendicular distance from a point to a finite line segment, and projection t.

    Args:
        point: Query point as (x, y).
        seg_p1: First endpoint of the segment.
        seg_p2: Second endpoint of the segment.

    Returns:
        Tuple (perp_dist, t) where t is the projection parameter along the
        segment (0 = p1, 1 = p2).  t outside [0, 1] means the closest point
        on the *infinite line* lies off the segment end.
    """
    seg = seg_p2 - seg_p1
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-9:
        return float(np.linalg.norm(point - seg_p1)), 0.0
    t = float(np.dot(point - seg_p1, seg) / seg_len_sq)
    proj = seg_p1 + t * seg
    return float(np.linalg.norm(point - proj)), t


def _angle_error_deg(pred_angle: float, gt_angle: float) -> float:
    """Angular error in degrees, accounting for 180° streak symmetry.

    Args:
        pred_angle: Predicted angle in degrees.
        gt_angle: Ground-truth angle in degrees.

    Returns:
        Absolute angular error in [0, 90].
    """
    diff = abs(pred_angle - gt_angle) % 180.0
    return min(diff, 180.0 - diff)


def _endpoint_error_px(pred_obb: dict, gt_obb: dict) -> float:
    """Mean endpoint error between a predicted and GT OBB.

    Tries both endpoint orientations (streak is undirected) and picks the
    minimum.  Reports the mean of the two per-endpoint distances.

    Args:
        pred_obb: Predicted OBB dict.
        gt_obb: Ground-truth OBB dict.

    Returns:
        Mean endpoint distance in pixels.
    """
    gt_p1, gt_p2 = _centerline_endpoints(gt_obb)
    pr_p1, pr_p2 = _centerline_endpoints(pred_obb)
    err_fwd = (np.linalg.norm(gt_p1 - pr_p1) + np.linalg.norm(gt_p2 - pr_p2)) / 2.0
    err_rev = (np.linalg.norm(gt_p1 - pr_p2) + np.linalg.norm(gt_p2 - pr_p1)) / 2.0
    return float(min(err_fwd, err_rev))


def _band_for(streak_length_px: float) -> str:
    """Return the length band label for a streak.

    Args:
        streak_length_px: Streak length in pixels.

    Returns:
        "short", "medium", or "long".
    """
    if streak_length_px < _SHORT_MAX:
        return "short"
    if streak_length_px < _LONG_MIN:
        return "medium"
    return "long"


# ---------------------------------------------------------------------------
# Tier-1 matching
# ---------------------------------------------------------------------------

def _centerline_match(
    pred: dict,
    gt: dict,
    perp_threshold_px: float,
) -> bool:
    """Return True if pred's centre projects within the GT segment at ≤ threshold.

    Strict criterion: the projection parameter t must be in [0, 1].  A
    prediction whose centre lies off the end of the GT streak is a miss,
    because Radon + endpoint refinement has no signal to latch onto there.

    Args:
        pred: Prediction dict with an "obb" sub-dict.
        gt: Ground-truth dict with an "obb" sub-dict.
        perp_threshold_px: Maximum allowed perpendicular distance.

    Returns:
        True if the strict match criterion is satisfied.
    """
    pred_center = np.array([float(pred["obb"]["cx"]), float(pred["obb"]["cy"])])
    gt_p1, gt_p2 = _centerline_endpoints(gt["obb"])
    perp_dist, t = _point_to_segment(pred_center, gt_p1, gt_p2)
    return (0.0 <= t <= 1.0) and (perp_dist <= perp_threshold_px)


def _greedy_centerline_match(
    preds: list[dict],
    gts: list[dict],
    perp_threshold_px: float,
) -> tuple[list[tuple[int, int]], list[bool]]:
    """Greedy matching by confidence descending, using centerline criterion.

    Args:
        preds: Predictions for a single image, pre-sorted by confidence.
        gts: Ground-truth annotations for the same image.
        perp_threshold_px: Perpendicular threshold for a valid match.

    Returns:
        matched_pairs: List of (pred_index, gt_index).
        is_tp: Boolean mask aligned with preds.
    """
    is_tp = [False] * len(preds)
    matched_gts: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []

    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if gi in matched_gts:
                continue
            if _centerline_match(pred, gt, perp_threshold_px):
                is_tp[pi] = True
                matched_gts.add(gi)
                matched_pairs.append((pi, gi))
                break  # greedy: first match wins for this pred

    return matched_pairs, is_tp


def _match_all_images(
    predictions: list[dict],
    ground_truth: list[dict],
    perp_threshold_px: float,
) -> tuple[list[tuple[dict, dict]], list[bool], int]:
    """Match predictions to GT across all images.

    Args:
        predictions: All predicted detections.
        ground_truth: All ground-truth annotations.
        perp_threshold_px: Perpendicular threshold for a valid match.

    Returns:
        matched_pairs: List of (pred_dict, gt_dict) for Tier-2/3 geometry.
        is_tp: Boolean list aligned with confidence-sorted predictions.
        n_gt: Total GT count.
    """
    preds_by_img: dict[str, list[dict]] = defaultdict(list)
    gts_by_img: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        preds_by_img[str(p["image_id"])].append(p)
    for g in ground_truth:
        gts_by_img[str(g["image_id"])].append(g)

    all_image_ids = set(preds_by_img) | set(gts_by_img)
    matched_pairs: list[tuple[dict, dict]] = []
    is_tp_all: list[bool] = []

    for img_id in all_image_ids:
        img_preds = sorted(
            preds_by_img[img_id],
            key=lambda x: x.get("confidence", 0.0),
            reverse=True,
        )
        img_gts = gts_by_img[img_id]
        pairs, is_tp = _greedy_centerline_match(img_preds, img_gts, perp_threshold_px)
        is_tp_all.extend(is_tp)
        for pi, gi in pairs:
            matched_pairs.append((img_preds[pi], img_gts[gi]))

    return matched_pairs, is_tp_all, len(ground_truth)


# ---------------------------------------------------------------------------
# Geometry summary helpers
# ---------------------------------------------------------------------------

def _geometry_stats(
    pairs: list[tuple[dict, dict]],
    use_refined_pred: bool = False,
) -> dict:
    """Compute angle and endpoint statistics over matched pairs.

    Args:
        pairs: List of (pred_dict, gt_dict).  Each pred_dict may have an
            "obb_refined" key (set during Tier-3 computation).
        use_refined_pred: If True, use "obb_refined" instead of "obb" for
            the prediction side.

    Returns:
        Dict with mean/median/p90 for angle error and endpoint error, plus
        per-band breakdowns.
    """
    if not pairs:
        empty = {"mean": 0.0, "median": 0.0, "p90": 0.0}
        return {
            "n_pairs": 0,
            "angle_err_deg": empty,
            "endpoint_err_px": empty,
            "per_band": {
                b: {"n_pairs": 0, "angle_err_deg": dict(empty), "endpoint_err_px": dict(empty)}
                for b in ("short", "medium", "long")
            },
        }

    angle_errs: list[float] = []
    ep_errs: list[float] = []
    band_angle: dict[str, list[float]] = {"short": [], "medium": [], "long": []}
    band_ep:    dict[str, list[float]] = {"short": [], "medium": [], "long": []}

    for pred, gt in pairs:
        pred_obb = pred.get("obb_refined") if use_refined_pred else None
        if pred_obb is None:
            pred_obb = pred["obb"]

        ae = _angle_error_deg(
            float(pred_obb["angle_deg"]),
            float(gt["obb"]["angle_deg"]),
        )
        ee = _endpoint_error_px(pred_obb, gt["obb"])
        angle_errs.append(ae)
        ep_errs.append(ee)

        band = _band_for(float(gt.get("streak_length_px") or gt["obb"].get("w", 0)))
        band_angle[band].append(ae)
        band_ep[band].append(ee)

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        arr = np.array(vals)
        return {
            "mean":   round(float(arr.mean()), 3),
            "median": round(float(np.median(arr)), 3),
            "p90":    round(float(np.percentile(arr, 90)), 3),
        }

    per_band: dict = {}
    for band in ("short", "medium", "long"):
        per_band[band] = {
            "n_pairs": len(band_angle[band]),
            "angle_err_deg": _stats(band_angle[band]),
            "endpoint_err_px": _stats(band_ep[band]),
        }

    return {
        "n_pairs": len(pairs),
        "angle_err_deg": _stats(angle_errs),
        "endpoint_err_px": _stats(ep_errs),
        "per_band": per_band,
    }


# ---------------------------------------------------------------------------
# Tier-3 refinement
# ---------------------------------------------------------------------------

def _extract_crop(image: np.ndarray, obb: dict, pad_factor: float = 1.5) -> np.ndarray:
    """Extract a padded rectangular crop around an OBB for Radon input.

    Args:
        image: Full image array (H, W) or (H, W, C).
        obb: OBB dict {cx, cy, w, h, angle_deg}.
        pad_factor: Multiplier on the OBB half-extents for padding.

    Returns:
        Cropped sub-array (may be smaller than requested at image edges).
    """
    cx, cy = int(round(float(obb["cx"]))), int(round(float(obb["cy"])))
    half_side = int(math.ceil(max(float(obb["w"]), float(obb["h"])) / 2.0 * pad_factor)) + 8
    h_img, w_img = image.shape[:2]
    x1 = max(0, cx - half_side)
    y1 = max(0, cy - half_side)
    x2 = min(w_img, cx + half_side)
    y2 = min(h_img, cy + half_side)
    return image[y1:y2, x1:x2]


def _refine_prediction(
    pred: dict,
    image: np.ndarray,
) -> dict:
    """Return a copy of pred with "obb_refined" populated via Radon + endpoint extension.

    Args:
        pred: Prediction dict with an "obb" key.
        image: Full image array for the corresponding image_id.

    Returns:
        Prediction dict with an added "obb_refined" key.
    """
    from inference.postprocess import refine_angle, extend_obb_to_streak_extent

    obb = pred["obb"]

    # Step 1: Radon angle refinement on a padded crop
    crop = _extract_crop(image, obb)
    refined_angle = refine_angle(crop, obb)

    # Step 2: Build an updated OBB with the Radon angle, then extend endpoints
    obb_with_angle = {**obb, "angle_deg": refined_angle}
    obb_refined = extend_obb_to_streak_extent(image, obb_with_angle)

    return {**pred, "obb_refined": obb_refined}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_geometry(
    predictions: list[dict],
    ground_truth: list[dict],
    perp_threshold_px: float = DEFAULT_PERP_THRESHOLD_PX,
    image_arrays: dict[str, np.ndarray] | None = None,
) -> dict:
    """Three-tier streak geometry evaluation.

    Tier 1 — Detection: did the model find each streak at all?
    Tier 2 — Raw geometry: angle and endpoint accuracy of raw model output.
    Tier 3 — Refined geometry: accuracy after Radon + endpoint refinement
              (only computed when ``image_arrays`` is provided).

    The Tier-1 match criterion is strict: the predicted centre must project
    within the GT segment bounds (0 ≤ t ≤ 1) **and** be within
    ``perp_threshold_px`` of the GT centreline.  Predictions whose centre
    lands off the end of the streak are counted as misses — Radon and endpoint
    tracing have nothing to latch onto outside the streak extent.

    Args:
        predictions: List of detection dicts.  Each must have keys:
            "image_id", "confidence", "obb" {cx, cy, w, h, angle_deg},
            "streak_length_px".
        ground_truth: List of annotation dicts with the same schema.
        perp_threshold_px: Maximum perpendicular distance (px) from the
            predicted centre to the GT centreline for a Tier-1 match.
        image_arrays: Optional dict mapping image_id (str) → numpy array for
            the corresponding image.  Required to compute Tier 3.

    Returns:
        Dict with keys "perp_threshold_px", "tier1_detection",
        "tier2_raw_geometry", "tier3_refined_geometry" (None if no images).
    """
    matched_pairs, is_tp, n_gt = _match_all_images(
        predictions, ground_truth, perp_threshold_px
    )

    n_found = sum(is_tp)
    n_fp = len(is_tp) - n_found

    # Per-band detection counts
    gt_band_counts: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    for g in ground_truth:
        gt_band_counts[_band_for(float(g.get("streak_length_px") or g["obb"].get("w", 0)))] += 1

    found_band_counts: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    for _, gt in matched_pairs:
        found_band_counts[_band_for(float(gt.get("streak_length_px") or gt["obb"].get("w", 0)))] += 1

    def _safe_div(a: int, b: int) -> float:
        return round(a / b, 4) if b > 0 else 0.0

    per_band_detection: dict[str, dict] = {}
    for band in ("short", "medium", "long"):
        n_b_gt    = gt_band_counts[band]
        n_b_found = found_band_counts[band]
        per_band_detection[band] = {
            "n_gt":    n_b_gt,
            "n_found": n_b_found,
            "recall":  _safe_div(n_b_found, n_b_gt),
        }

    tier1 = {
        "n_gt": n_gt,
        "n_found": n_found,
        "n_false_positives": n_fp,
        "detection_recall":   _safe_div(n_found, n_gt),
        "detection_precision": _safe_div(n_found, n_found + n_fp),
        "per_band": per_band_detection,
    }

    # Tier 2 — raw geometry
    tier2 = _geometry_stats(matched_pairs, use_refined_pred=False)

    # Tier 3 — refined geometry
    tier3: dict | None = None
    if image_arrays is not None:
        refined_pairs: list[tuple[dict, dict]] = []
        for pred, gt in matched_pairs:
            img_id = str(pred["image_id"])
            image = image_arrays.get(img_id)
            if image is None:
                logger.warning("geometry_metrics: no image array for %s — skipping Tier 3 pair", img_id)
                refined_pairs.append((pred, gt))
                continue
            try:
                pred_refined = _refine_prediction(pred, image)
            except Exception as exc:
                logger.warning("geometry_metrics: refinement failed for %s: %s", img_id, exc)
                refined_pairs.append((pred, gt))
                continue
            refined_pairs.append((pred_refined, gt))

        tier3 = _geometry_stats(refined_pairs, use_refined_pred=True)

        # Improvement deltas (positive = refinement helped)
        tier3["angle_improvement_deg"] = round(
            tier2["angle_err_deg"]["mean"] - tier3["angle_err_deg"]["mean"], 3
        )
        tier3["endpoint_improvement_px"] = round(
            tier2["endpoint_err_px"]["mean"] - tier3["endpoint_err_px"]["mean"], 3
        )

    return {
        "perp_threshold_px": perp_threshold_px,
        "tier1_detection": tier1,
        "tier2_raw_geometry": tier2,
        "tier3_refined_geometry": tier3,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_ground_truth(annotations_path: Path) -> list[dict]:
    """Load COCO-format annotations as geometry-metric ground truth.

    Args:
        annotations_path: Path to COCO JSON annotation file.

    Returns:
        List of GT dicts with image_id, obb, streak_length_px.
    """
    import json
    with open(annotations_path) as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    ground_truth: list[dict] = []
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        obb_raw = ann.get("obb")
        if not obb_raw:
            continue
        if isinstance(obb_raw, dict):
            try:
                cx  = float(obb_raw["cx"])
                cy  = float(obb_raw["cy"])
                w   = float(obb_raw["w"])
                h   = float(obb_raw["h"])
                ang = float(obb_raw["angle_deg"])
            except (KeyError, TypeError, ValueError):
                continue
        else:
            if len(obb_raw) < 5:
                continue
            cx, cy, w, h, ang = [float(v) for v in obb_raw[:5]]
        ground_truth.append({
            "image_id": id_to_filename.get(ann["image_id"], str(ann["image_id"])),
            "obb": {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": ang},
            "streak_length_px": max(w, h),
        })
    return ground_truth


def _load_image_arrays(
    image_ids: list[str],
    images_dir: Path,
) -> dict[str, np.ndarray]:
    """Load FITS images as numpy arrays, keyed by image_id (filename).

    Args:
        image_ids: List of filenames (may be relative paths).
        images_dir: Directory in which to look for each file.

    Returns:
        Dict mapping image_id → numpy array.  Missing files are logged and
        omitted.
    """
    from inference.fits_loader import FITSLoader

    loader = FITSLoader()
    arrays: dict[str, np.ndarray] = {}
    for img_id in image_ids:
        path = images_dir / Path(img_id).name
        if not path.exists():
            logger.warning("geometry_metrics: image not found: %s", path)
            continue
        try:
            tensor, _ = loader.load(path)
            # Convert CHW float tensor to HWC uint8 numpy
            arr = tensor.permute(1, 2, 0).numpy()
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            arrays[img_id] = arr
            logger.debug("Loaded %s  shape=%s", path.name, arr.shape)
        except Exception as exc:
            logger.warning("geometry_metrics: failed to load %s: %s", path, exc)
    return arrays


if __name__ == "__main__":
    import argparse
    import json
    import logging
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="ARGUS three-tier streak geometry evaluation"
    )
    parser.add_argument("--predictions",  required=True, help="Predictions JSON file")
    parser.add_argument("--annotations",  required=True, help="COCO annotations JSON file")
    parser.add_argument("--images-dir",   help="Directory of raw FITS images (enables Tier 3)")
    parser.add_argument("--perp-threshold", type=float, default=DEFAULT_PERP_THRESHOLD_PX,
                        help=f"Perpendicular distance threshold for Tier-1 match (px, default {DEFAULT_PERP_THRESHOLD_PX})")
    parser.add_argument("--output",       help="Where to save the JSON results file")
    args = parser.parse_args()

    with open(args.predictions) as f:
        preds = json.load(f)
    logger.info("Loaded %d predictions", len(preds))

    gt = _load_ground_truth(Path(args.annotations))
    logger.info("Loaded %d ground-truth annotations", len(gt))

    image_arrays: dict[str, np.ndarray] | None = None
    if args.images_dir:
        image_ids = list({p["image_id"] for p in preds})
        image_arrays = _load_image_arrays(image_ids, Path(args.images_dir))
        logger.info("Loaded %d/%d images for Tier 3", len(image_arrays), len(image_ids))

    results = evaluate_geometry(preds, gt, perp_threshold_px=args.perp_threshold,
                                image_arrays=image_arrays)
    results["date_recorded"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    t1 = results["tier1_detection"]
    t2 = results["tier2_raw_geometry"]
    t3 = results["tier3_refined_geometry"]

    print(f"\n── Tier 1: Detection  (perp ≤ {args.perp_threshold} px) ──")
    print(f"  recall    : {t1['detection_recall']:.1%}  ({t1['n_found']}/{t1['n_gt']})")
    print(f"  precision : {t1['detection_precision']:.1%}")
    for band in ("short", "medium", "long"):
        b = t1["per_band"][band]
        print(f"    {band:6s}  recall={b['recall']:.1%}  ({b['n_found']}/{b['n_gt']})")

    print(f"\n── Tier 2: Raw geometry  (n={t2['n_pairs']}) ──")
    ae = t2["angle_err_deg"]
    ep = t2["endpoint_err_px"]
    print(f"  angle  err: mean={ae['mean']}°  median={ae['median']}°  p90={ae['p90']}°")
    print(f"  endpt  err: mean={ep['mean']}px  median={ep['median']}px  p90={ep['p90']}px")

    if t3 is not None:
        print(f"\n── Tier 3: Refined geometry  (n={t3['n_pairs']}) ──")
        ae3 = t3["angle_err_deg"]
        ep3 = t3["endpoint_err_px"]
        print(f"  angle  err: mean={ae3['mean']}°  median={ae3['median']}°  p90={ae3['p90']}°  "
              f"(Δ {t3['angle_improvement_deg']:+.3f}°)")
        print(f"  endpt  err: mean={ep3['mean']}px  median={ep3['median']}px  p90={ep3['p90']}px  "
              f"(Δ {t3['endpoint_improvement_px']:+.3f}px)")
    else:
        print("\n── Tier 3: skipped (no --images-dir) ──")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", out_path)
    else:
        print("\n" + json.dumps(results, indent=2))
