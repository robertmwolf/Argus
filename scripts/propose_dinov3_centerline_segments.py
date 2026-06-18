"""Propose endpoint streak segments from DINOv3 centerline heatmaps.

This script implements the post-training path for the heatmap spike:

1. run the orientation-binned centerline model,
2. extract thresholded seed components,
3. keep orientation-consistent components,
4. refine the seed angle with a local Radon transform, and
5. write line-segment proposals for downstream streak post-processing.

The output intentionally avoids oriented boxes.  The line segment is represented
as two endpoints plus confidence/orientation metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import imagenet_normalize
from scripts.evaluate_dinov3_orientation_centerline import _load_checkpoint_model
from training.dinov3_orientation_centerline_dataset import DINOv3OrientationCenterlineDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Point:
    """A 2D point in image coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class LineSegmentProposal:
    """One endpoint centerline proposal."""

    component_id: int
    score: float
    mean_score: float
    area_px: int
    dominant_bin: int
    orientation_consistency: float
    seed_angle_deg: float
    radon_angle_deg: float
    radon_snr: float
    line_support_ratio: float
    input_start: Point
    input_end: Point
    native_start: Point
    native_end: Point
    input_bbox_xyxy: list[int]


def _angle_for_bin(bin_idx: int, n_bins: int) -> float:
    """Return the centerline angle represented by an orientation bin."""
    return (float(bin_idx) / max(float(n_bins), 1.0) * 180.0) % 180.0


def _wrapped_bin_delta(a: np.ndarray, b: int, n_bins: int) -> np.ndarray:
    """Return circular distance between orientation bins on a half-circle."""
    delta = np.abs(a.astype(np.int32) - int(b))
    return np.minimum(delta, int(n_bins) - delta)


def _to_gray(image_chw: np.ndarray) -> np.ndarray:
    """Convert a CHW float RGB image in [0, 1] to a contrast-stretched gray image."""
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    gray = image_hwc.mean(axis=2).astype(np.float32)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return np.zeros_like(gray, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.7])
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _to_uint8_rgb(image_chw: np.ndarray) -> np.ndarray:
    """Convert a CHW float RGB image in [0, 1] to uint8 RGB."""
    return np.clip(np.transpose(image_chw, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)


def _heat_to_rgb(heat: np.ndarray) -> np.ndarray:
    """Colorize a heatmap as an RGB overlay."""
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def _crop_bounds(
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    pad: int,
) -> tuple[int, int, int, int]:
    """Return padded integer crop bounds as x1, y1, x2, y2."""
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(width, int(xs.max()) + pad + 1)
    y2 = min(height, int(ys.max()) + pad + 1)
    return x1, y1, x2, y2


def _refine_angle_radon(
    gray_crop: np.ndarray,
    seed_angle_deg: float,
    search_degrees: float,
    step_degrees: float,
) -> tuple[float, float]:
    """Refine a seed line angle using local Radon variance.

    Returns:
        Tuple of (refined_angle_deg, radon_snr) where radon_snr is the peak
        column variance divided by the mean column variance.  Values above ~3
        indicate a genuine linear structure; flat sinograms return 1.0.
    """
    from skimage.transform import radon  # type: ignore[import]

    crop = np.asarray(gray_crop, dtype=np.float32)
    if crop.size == 0 or min(crop.shape) < 4 or search_degrees <= 0.0:
        return float(seed_angle_deg % 180.0), 1.0
    crop = np.clip(crop - np.median(crop), 0.0, None)
    if float(crop.max()) <= 0.0:
        return float(seed_angle_deg % 180.0), 1.0

    max_side = 512
    h_crop, w_crop = crop.shape
    if max(h_crop, w_crop) > max_side:
        scale = max_side / max(h_crop, w_crop)
        crop = cv2.resize(
            crop,
            (max(1, int(w_crop * scale)), max(1, int(h_crop * scale))),
            interpolation=cv2.INTER_AREA,
        )

    radon_center = 90.0 - seed_angle_deg
    theta = np.arange(
        radon_center - search_degrees,
        radon_center + search_degrees + step_degrees,
        step_degrees,
        dtype=np.float32,
    )
    if theta.size == 0:
        return float(seed_angle_deg % 180.0), 1.0
    try:
        sinogram = radon(crop, theta=theta, circle=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Radon refinement failed: %s", exc)
        return float(seed_angle_deg % 180.0), 1.0
    col_vars = sinogram.var(axis=0)
    best_idx = int(np.argmax(col_vars))
    best_radon = float(theta[best_idx])
    mean_var = float(col_vars.mean())
    radon_snr = float(col_vars[best_idx] / (mean_var + 1e-8))
    return float((90.0 - best_radon) % 180.0), radon_snr


def _line_support_ratio(
    component_mask: np.ndarray,
    heat: np.ndarray,
    start: Point,
    end: Point,
    tolerance_px: float,
) -> float:
    """Return the heat-weighted fraction of component pixels near the fitted line.

    A genuine streak places almost all of its activation within a few pixels of
    the line; a diffuse blob spreads widely and scores low.

    Args:
        component_mask: Boolean mask of the seed component.
        heat: Full heatmap (same spatial dimensions as component_mask).
        start: Traced segment start in heatmap pixel space.
        end: Traced segment end in heatmap pixel space.
        tolerance_px: Maximum perpendicular distance to count as "on the line".

    Returns:
        Score in [0, 1].  Values below ~0.5 indicate poor line support.
    """
    ys, xs = np.nonzero(component_mask)
    if xs.size == 0:
        return 0.0
    weights = heat[ys, xs].astype(np.float64)
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    px = -dy / length
    py = dx / length
    cross = np.abs((xs.astype(np.float64) - start.x) * px + (ys.astype(np.float64) - start.y) * py)
    within = cross <= tolerance_px
    w_sum = float(weights.sum())
    if w_sum < 1e-9:
        return float(within.mean())
    return float((weights * within).sum() / w_sum)


def _trace_segment_from_heat(
    heat: np.ndarray,
    component_mask: np.ndarray,
    angle_deg: float,
    threshold: float,
    min_length_px: float,
    extension_px: float,
) -> tuple[Point, Point] | None:
    """Fit a line segment to a heatmap component along the refined angle."""
    ys, xs = np.nonzero(component_mask)
    if xs.size == 0:
        return None
    weights = np.maximum(heat[ys, xs], 1e-6)
    cx = float(np.average(xs, weights=weights))
    cy = float(np.average(ys, weights=weights))

    theta = math.radians(angle_deg)
    ux = math.cos(theta)
    uy = math.sin(theta)
    px = -uy
    py = ux

    local_mask = ndimage.binary_dilation(component_mask, iterations=max(1, int(extension_px)))
    candidate_mask = local_mask & (heat >= max(threshold * 0.75, 1e-6))
    cand_y, cand_x = np.nonzero(candidate_mask)
    if cand_x.size == 0:
        cand_x = xs
        cand_y = ys
    along = (cand_x.astype(np.float32) - cx) * ux + (cand_y.astype(np.float32) - cy) * uy
    across = np.abs((cand_x.astype(np.float32) - cx) * px + (cand_y.astype(np.float32) - cy) * py)

    component_along = (xs.astype(np.float32) - cx) * ux + (ys.astype(np.float32) - cy) * uy
    cross_limit = max(3.0, float(np.percentile(across, 15.0)) + 3.0)
    support = across <= cross_limit
    if int(support.sum()) >= max(3, int(xs.size * 0.25)):
        support_along = along[support]
    else:
        support_along = component_along

    t0 = float(np.min(support_along) - extension_px)
    t1 = float(np.max(support_along) + extension_px)
    if t1 - t0 < min_length_px:
        center_t = (t0 + t1) * 0.5
        half = min_length_px * 0.5
        t0 = center_t - half
        t1 = center_t + half

    h, w = heat.shape
    start = Point(
        x=float(np.clip(cx + t0 * ux, 0, w - 1)),
        y=float(np.clip(cy + t0 * uy, 0, h - 1)),
    )
    end = Point(
        x=float(np.clip(cx + t1 * ux, 0, w - 1)),
        y=float(np.clip(cy + t1 * uy, 0, h - 1)),
    )
    return start, end


def _rasterise_segment_mask(
    start: Point,
    end: Point,
    shape: tuple[int, int],
    tolerance_px: int,
) -> np.ndarray:
    """Rasterise a line segment as a distance-tolerant binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(
        mask,
        (int(round(start.x)), int(round(start.y))),
        (int(round(end.x)), int(round(end.y))),
        color=1,
        thickness=max(1, int(tolerance_px) * 2 + 1),
    )
    return mask.astype(bool)


def _proposal_gt_coverage(
    proposal: LineSegmentProposal,
    gt_mask: np.ndarray,
    tolerance_px: int,
) -> float:
    """Return fraction of GT centerline pixels covered by a line proposal."""
    if not gt_mask.any():
        return 0.0
    segment_mask = _rasterise_segment_mask(
        proposal.input_start,
        proposal.input_end,
        gt_mask.shape,
        tolerance_px=tolerance_px,
    )
    return float(np.logical_and(segment_mask, gt_mask).sum() / max(int(gt_mask.sum()), 1))


def _native_point(point: Point, sample: Any, input_size: int) -> Point:
    """Map a model-input point back into the sample's native image coordinates."""
    sx = float(sample.crop_w) / max(float(input_size), 1.0)
    sy = float(sample.crop_h) / max(float(input_size), 1.0)
    return Point(
        x=float(sample.crop_x + point.x * sx),
        y=float(sample.crop_y + point.y * sy),
    )


def _extract_proposals(
    heat: np.ndarray,
    bins: np.ndarray,
    gray: np.ndarray,
    sample: Any,
    threshold: float,
    min_component_pixels: int,
    orientation_neighbor_bins: int,
    min_orientation_consistency: float,
    crop_padding: int,
    radon_search_degrees: float,
    radon_step_degrees: float,
    min_length_px: float,
    extension_px: float,
    max_components: int,
    n_bins: int,
    min_line_support: float,
    line_support_tolerance_px: float,
    min_radon_snr: float,
) -> list[LineSegmentProposal]:
    """Extract orientation-consistent seed components and refine them into segments.

    After tracing each segment, two quality gates prune FPs before the proposal
    is emitted:

    * ``min_line_support`` — minimum heat-weighted fraction of component pixels
      within ``line_support_tolerance_px`` of the fitted line.  Blobs score
      low; genuine streaks score ≥ 0.5.
    * ``min_radon_snr`` — minimum ratio of peak to mean Radon column variance.
      Flat, noisy components score ~1; real lines score ≥ 3.
    """
    seed_mask = heat >= threshold
    labels, label_count = ndimage.label(seed_mask)
    h, w = heat.shape
    proposals: list[LineSegmentProposal] = []

    for label_id in range(1, label_count + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < min_component_pixels:
            continue
        ys, xs = np.nonzero(component)
        component_bins = bins[component]
        hist = np.bincount(component_bins.astype(np.int32), minlength=n_bins)
        dominant_bin = int(hist.argmax())
        consistency = float(
            (_wrapped_bin_delta(component_bins, dominant_bin, n_bins) <= orientation_neighbor_bins).mean()
        )
        if consistency < min_orientation_consistency:
            continue

        seed_angle = _angle_for_bin(dominant_bin, n_bins)
        x1, y1, x2, y2 = _crop_bounds(xs, ys, w, h, crop_padding)
        refined_angle, radon_snr = _refine_angle_radon(
            gray_crop=gray[y1:y2, x1:x2],
            seed_angle_deg=seed_angle,
            search_degrees=radon_search_degrees,
            step_degrees=radon_step_degrees,
        )
        if radon_snr < min_radon_snr:
            continue

        segment = _trace_segment_from_heat(
            heat=heat,
            component_mask=component,
            angle_deg=refined_angle,
            threshold=threshold,
            min_length_px=min_length_px,
            extension_px=extension_px,
        )
        if segment is None:
            continue
        start, end = segment
        support = _line_support_ratio(
            component_mask=component,
            heat=heat,
            start=start,
            end=end,
            tolerance_px=line_support_tolerance_px,
        )
        if support < min_line_support:
            continue

        proposals.append(
            LineSegmentProposal(
                component_id=int(label_id),
                score=float(heat[component].max()),
                mean_score=float(heat[component].mean()),
                area_px=area,
                dominant_bin=dominant_bin,
                orientation_consistency=consistency,
                seed_angle_deg=seed_angle,
                radon_angle_deg=refined_angle,
                radon_snr=radon_snr,
                line_support_ratio=support,
                input_start=start,
                input_end=end,
                native_start=_native_point(start, sample, input_size=w),
                native_end=_native_point(end, sample, input_size=w),
                input_bbox_xyxy=[int(x1), int(y1), int(x2), int(y2)],
            )
        )

    proposals.sort(key=lambda item: (item.line_support_ratio, item.score, item.area_px), reverse=True)
    return proposals[:max_components]


def _proposal_to_dict(proposal: LineSegmentProposal) -> dict[str, Any]:
    """Convert a proposal dataclass into JSON-serializable primitives."""
    payload = asdict(proposal)
    for key in ("input_start", "input_end", "native_start", "native_end"):
        payload[key] = asdict(getattr(proposal, key))
    return payload


def _render_overlay(
    output_path: Path,
    image_chw: np.ndarray,
    heat: np.ndarray,
    proposals: list[LineSegmentProposal],
    coverages: list[float],
) -> None:
    """Render source, heatmap, and line-segment proposals for QA."""
    source = _to_uint8_rgb(image_chw)
    heat_rgb = _heat_to_rgb(heat)
    overlay = np.clip(
        source.astype(np.float32) * 0.55 + heat_rgb.astype(np.float32) * 0.45,
        0,
        255,
    ).astype(np.uint8)
    canvas = overlay.copy()
    palette = [
        (80, 255, 120),
        (255, 230, 80),
        (80, 180, 255),
        (255, 120, 220),
        (255, 140, 80),
    ]
    for idx, proposal in enumerate(proposals):
        color = palette[idx % len(palette)]
        start = (int(round(proposal.input_start.x)), int(round(proposal.input_start.y)))
        end = (int(round(proposal.input_end.x)), int(round(proposal.input_end.y)))
        cv2.line(canvas, start, end, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, start, 3, color=color, thickness=-1)
        cv2.circle(canvas, end, 3, color=color, thickness=-1)
        label = f"{idx + 1}:{proposal.radon_angle_deg:.1f} cov={coverages[idx]:.2f}"
        cv2.putText(
            canvas,
            label,
            (max(0, start[0] + 4), max(14, start[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> int:
    """Run centerline proposal generation from a trained checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output", default="results/dinov3_centerline_segments/proposals.json")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--target-threshold", type=float, default=0.05)
    parser.add_argument("--proposal-tolerance-px", type=int, default=6)
    parser.add_argument("--proposal-gt-coverage", type=float, default=0.10)
    parser.add_argument("--min-component-pixels", type=int, default=4)
    parser.add_argument("--orientation-neighbor-bins", type=int, default=1)
    parser.add_argument("--min-orientation-consistency", type=float, default=0.55)
    parser.add_argument("--crop-padding", type=int, default=48)
    parser.add_argument("--radon-search-degrees", type=float, default=12.0)
    parser.add_argument("--radon-step-degrees", type=float, default=0.5)
    parser.add_argument("--min-length-px", type=float, default=16.0)
    parser.add_argument("--extension-px", type=float, default=8.0)
    parser.add_argument("--max-components-per-image", type=int, default=2)
    parser.add_argument(
        "--min-line-support",
        type=float,
        default=0.50,
        help="Minimum heat-weighted fraction of component pixels within "
        "--line-support-tolerance-px of the traced line.  0.0 disables the gate.",
    )
    parser.add_argument(
        "--line-support-tolerance-px",
        type=float,
        default=3.0,
        help="Perpendicular tolerance (pixels) used for the line-support ratio.",
    )
    parser.add_argument(
        "--min-radon-snr",
        type=float,
        default=1.0,
        help="Minimum Radon peak-variance / mean-variance ratio.  1.0 disables "
        "the gate; genuine lines typically score > 3.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overlay-dir", default=None)
    parser.add_argument("--max-overlays", type=int, default=24)
    parser.add_argument("--preserve-image-bit-depth", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    model, train_args = _load_checkpoint_model(Path(args.checkpoint), args.weights, device)
    annotations = args.annotations or train_args.get("val_annotations", "data/annotations/val.json")
    preserve_bit_depth = bool(args.preserve_image_bit_depth or train_args.get("preserve_image_bit_depth", False))
    orientation_bins = int(train_args.get("orientation_bins", 18))
    dataset = DINOv3OrientationCenterlineDataset(
        annotation_file=annotations,
        split="val",
        tile_size=int(train_args.get("tile_size", 2560)),
        image_size=int(train_args.get("image_size", 1024)),
        orientation_bins=orientation_bins,
        centerline_width=float(train_args.get("centerline_width", 2.0)),
        centerline_sigma=float(train_args.get("centerline_sigma", 1.4)),
        neighbor_bin_weight=float(train_args.get("neighbor_bin_weight", 0.35)),
        second_neighbor_weight=float(train_args.get("second_neighbor_weight", 0.0)),
        positive_tiles=None,
        negative_tiles=None,
        preserve_image_bit_depth=preserve_bit_depth,
        seed=int(train_args.get("seed", 20260524)),
        max_samples=args.max_samples,
    )

    image_records: list[dict[str, Any]] = []
    total_segments = 0
    positive_images = 0
    negative_images = 0
    positive_images_with_segment = 0
    negative_images_with_segment = 0
    positive_images_matched = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            sample = dataset.samples[idx]
            image = item["image"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            logits = model(imagenet_normalize(image))
            if logits.shape[-2:] != image.shape[-2:]:
                logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            heat = probs.max(axis=0).astype(np.float32)
            bins = probs.argmax(axis=0).astype(np.int32)
            gray = _to_gray(item["image"].numpy())  # type: ignore[union-attr]
            proposals = _extract_proposals(
                heat=heat,
                bins=bins,
                gray=gray,
                sample=sample,
                threshold=args.threshold,
                min_component_pixels=args.min_component_pixels,
                orientation_neighbor_bins=args.orientation_neighbor_bins,
                min_orientation_consistency=args.min_orientation_consistency,
                crop_padding=args.crop_padding,
                radon_search_degrees=args.radon_search_degrees,
                radon_step_degrees=args.radon_step_degrees,
                min_length_px=args.min_length_px,
                extension_px=args.extension_px,
                max_components=args.max_components_per_image,
                n_bins=orientation_bins,
                min_line_support=args.min_line_support,
                line_support_tolerance_px=args.line_support_tolerance_px,
                min_radon_snr=args.min_radon_snr,
            )
            target_heat = item["target"].max(dim=0).values.numpy()  # type: ignore[union-attr]
            gt_mask = target_heat > args.target_threshold
            gt_positive = bool(gt_mask.any())
            coverages = [
                _proposal_gt_coverage(
                    proposal,
                    gt_mask,
                    tolerance_px=args.proposal_tolerance_px,
                )
                for proposal in proposals
            ]
            matched = bool(coverages and max(coverages) >= args.proposal_gt_coverage)
            if gt_positive:
                positive_images += 1
                if proposals:
                    positive_images_with_segment += 1
                if matched:
                    positive_images_matched += 1
            else:
                negative_images += 1
                if proposals:
                    negative_images_with_segment += 1
            total_segments += len(proposals)
            image_records.append(
                {
                    "image_id": int(item["image_id"].item()),  # type: ignore[union-attr]
                    "file_name": str(item["file_name"]),
                    "positive": gt_positive,
                    "input_size": int(heat.shape[0]),
                    "native_width": int(sample.crop_w),
                    "native_height": int(sample.crop_h),
                    "max_heat": float(heat.max()),
                    "matched_gt": matched,
                    "best_gt_coverage": float(max(coverages) if coverages else 0.0),
                    "segments": [
                        {
                            **_proposal_to_dict(proposal),
                            "gt_coverage": float(coverage),
                            "matched_gt": bool(coverage >= args.proposal_gt_coverage),
                        }
                        for proposal, coverage in zip(proposals, coverages)
                    ],
                }
            )
            if args.overlay_dir and idx < args.max_overlays:
                _render_overlay(
                    output_path=Path(args.overlay_dir)
                    / f"{idx:04d}_image{int(item['image_id'].item())}.png",  # type: ignore[union-attr]
                    image_chw=item["image"].numpy(),  # type: ignore[union-attr]
                    heat=heat,
                    proposals=proposals,
                    coverages=coverages,
                )
            if (idx + 1) % 25 == 0:
                logger.info("processed %d/%d images, segments=%d", idx + 1, len(dataset), total_segments)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "annotations": str(annotations),
        "threshold": args.threshold,
        "min_component_pixels": args.min_component_pixels,
        "min_orientation_consistency": args.min_orientation_consistency,
        "radon_search_degrees": args.radon_search_degrees,
        "radon_step_degrees": args.radon_step_degrees,
        "min_line_support": args.min_line_support,
        "line_support_tolerance_px": args.line_support_tolerance_px,
        "min_radon_snr": args.min_radon_snr,
        "n_images": len(image_records),
        "n_segments": total_segments,
        "proposal_metrics": {
            "positive_images": positive_images,
            "negative_images": negative_images,
            "positive_images_with_segment": positive_images_with_segment,
            "positive_segment_recall": (
                positive_images_with_segment / positive_images if positive_images else 0.0
            ),
            "positive_images_matched": positive_images_matched,
            "positive_matched_recall": (
                positive_images_matched / positive_images if positive_images else 0.0
            ),
            "negative_images_with_segment": negative_images_with_segment,
            "negative_segment_rate": (
                negative_images_with_segment / negative_images if negative_images else 0.0
            ),
            "proposal_tolerance_px": args.proposal_tolerance_px,
            "proposal_gt_coverage": args.proposal_gt_coverage,
        },
        "images": image_records,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s with %d images and %d segments", output_path, len(image_records), total_segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
