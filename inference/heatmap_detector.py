"""DINOv3 orientation-centerline heatmap detector integration.

This module exposes the no-OBB heatmap model as an ARGUS detector.  Native
geometry is a line segment; an ``obb`` compatibility projection is also emitted
so the existing pipeline, WCS, database, and grouping code can consume the
detector before those layers become line-native.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import (
    DINOv3OrientationCenterline,
    imagenet_normalize,
)

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str | None, str], tuple[DINOv3OrientationCenterline, dict[str, Any], torch.device]] = {}


@dataclass(frozen=True)
class _Point:
    """A 2D point in pixel coordinates."""

    x: float
    y: float


def _default_checkpoint() -> Path:
    """Return the default local heatmap checkpoint path."""
    root = Path(__file__).resolve().parent.parent
    return root / "weights" / "run_dinov3_vitb_orientation_centerline_input512_catchment" / "best.pt"


def _angle_for_bin(bin_idx: int, n_bins: int) -> float:
    """Return the centerline angle represented by an orientation bin."""
    return (float(bin_idx) / max(float(n_bins), 1.0) * 180.0) % 180.0


def _wrapped_bin_delta(values: np.ndarray, center: int, n_bins: int) -> np.ndarray:
    """Return circular bin distance on a 180-degree orientation domain."""
    delta = np.abs(values.astype(np.int32) - int(center))
    return np.minimum(delta, int(n_bins) - delta)


def _to_gray(image_chw: np.ndarray) -> np.ndarray:
    """Convert CHW float RGB in [0, 1] to contrast-stretched gray."""
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    gray = image_hwc.mean(axis=2).astype(np.float32)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return np.zeros_like(gray, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.7])
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _refine_angle_radon(
    gray_crop: np.ndarray,
    seed_angle_deg: float,
    search_degrees: float,
    step_degrees: float,
) -> float:
    """Refine a seed line angle with a local Radon transform."""
    from skimage.transform import radon  # type: ignore[import]

    crop = np.asarray(gray_crop, dtype=np.float32)
    if crop.size == 0 or min(crop.shape) < 4 or search_degrees <= 0.0:
        return float(seed_angle_deg % 180.0)
    crop = np.clip(crop - np.median(crop), 0.0, None)
    if float(crop.max()) <= 0.0:
        return float(seed_angle_deg % 180.0)

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
        return float(seed_angle_deg % 180.0)
    try:
        sinogram = radon(crop, theta=theta, circle=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Heatmap Radon refinement failed: %s", exc)
        return float(seed_angle_deg % 180.0)
    best_radon = float(theta[int(np.argmax(sinogram.var(axis=0)))])
    return float((90.0 - best_radon) % 180.0)


def _line_support_ratio(
    component_mask: np.ndarray,
    heat: np.ndarray,
    start: _Point,
    end: _Point,
    tolerance_px: float = 3.0,
) -> float:
    """Return heat-weighted fraction of component pixels within tolerance_px of the line."""
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


def _trace_segment_from_heat(
    heat: np.ndarray,
    component_mask: np.ndarray,
    angle_deg: float,
    threshold: float,
    min_length_px: float,
    extension_px: float,
) -> tuple[_Point, _Point] | None:
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
    start = _Point(
        x=float(np.clip(cx + t0 * ux, 0, w - 1)),
        y=float(np.clip(cy + t0 * uy, 0, h - 1)),
    )
    end = _Point(
        x=float(np.clip(cx + t1 * ux, 0, w - 1)),
        y=float(np.clip(cy + t1 * uy, 0, h - 1)),
    )
    return start, end


def _line_to_obb(start: _Point, end: _Point, width_px: float) -> dict[str, float]:
    """Project a line segment into a minimal OBB-compatible dict."""
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx)) % 180.0 if length > 0 else 0.0
    return {
        "cx": float((start.x + end.x) * 0.5),
        "cy": float((start.y + end.y) * 0.5),
        "w": float(length),
        "h": float(width_px),
        "angle_deg": float(angle),
    }


def _load_heatmap_model(
    checkpoint_path: Path,
    weights_override: str | None,
) -> tuple[DINOv3OrientationCenterline, dict[str, Any], torch.device]:
    """Load and cache the orientation-centerline checkpoint."""
    device = get_device()
    cache_key = (str(checkpoint_path.resolve()), weights_override, device.type)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Heatmap checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args", {}))
    weights = weights_override or train_args.get("weights", "weights/dinov3_vitb16_lvd1689m.pth")
    model = DINOv3OrientationCenterline(
        model_size=train_args.get("model_size", "base"),
        weights=weights,
        decoder_channels=int(train_args.get("decoder_channels", 192)),
        orientation_bins=int(train_args.get("orientation_bins", 18)),
        last_layers=int(train_args.get("last_layers", 4)),
        freeze_backbone=True,
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        logger.info("Heatmap checkpoint missing %d model keys, first=%s", len(missing), missing[:3])
    if unexpected:
        logger.info("Heatmap checkpoint has %d unexpected model keys, first=%s", len(unexpected), unexpected[:3])
    model.eval()
    loaded = (model, train_args, device)
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def get_heatmap_detector_status() -> dict[str, str]:
    """Return availability metadata for the heatmap centerline detector."""
    checkpoint = Path(os.environ.get("HEATMAP_CENTERLINE_CHECKPOINT", str(_default_checkpoint())))
    return {
        "id": "dinov3_heatmap_centerline",
        "name": "DINOv3 Heatmap Centerline",
        "type": "ml",
        "dataset": "SatStreaks centerline catchment",
        "status": "active" if checkpoint.exists() else "no_weights",
    }


def run_heatmap_centerline_detector(array: np.ndarray) -> list[dict[str, Any]]:
    """Run the no-OBB heatmap detector on an RGB image array.

    Args:
        array: uint8 or float image array with shape ``(H, W, 3)``.

    Returns:
        Pipeline-compatible detections.  Native geometry lives in
        ``line_segment``; ``obb`` is a compatibility projection.
    """
    checkpoint = Path(os.environ.get("HEATMAP_CENTERLINE_CHECKPOINT", str(_default_checkpoint())))
    if not checkpoint.exists():
        logger.debug("Heatmap centerline checkpoint not found at %s; skipping", checkpoint)
        return []

    weights_override = os.environ.get("HEATMAP_DINOV3_WEIGHTS") or None
    model, train_args, device = _load_heatmap_model(checkpoint, weights_override)
    image_size = int(os.environ.get("HEATMAP_IMAGE_SIZE", str(train_args.get("image_size", 512))))
    threshold = float(os.environ.get("HEATMAP_SEGMENT_THRESHOLD", "0.85"))
    min_component_pixels = int(os.environ.get("HEATMAP_MIN_COMPONENT_PIXELS", "4"))
    orientation_neighbor_bins = int(os.environ.get("HEATMAP_ORIENTATION_NEIGHBOR_BINS", "1"))
    min_orientation_consistency = float(os.environ.get("HEATMAP_MIN_ORIENTATION_CONSISTENCY", "0.55"))
    max_components = int(os.environ.get("HEATMAP_MAX_COMPONENTS", "2"))
    min_line_support = float(os.environ.get("HEATMAP_MIN_LINE_SUPPORT", "0.50"))
    crop_padding = int(os.environ.get("HEATMAP_CROP_PADDING", "48"))
    radon_search_degrees = float(os.environ.get("HEATMAP_RADON_SEARCH_DEGREES", "12.0"))
    radon_step_degrees = float(os.environ.get("HEATMAP_RADON_STEP_DEGREES", "0.5"))
    min_length_px = float(os.environ.get("HEATMAP_MIN_LENGTH_PX", "16.0"))
    extension_px = float(os.environ.get("HEATMAP_EXTENSION_PX", "8.0"))
    compat_width_px = float(os.environ.get("HEATMAP_OBB_COMPAT_WIDTH_PX", "6.0"))
    n_bins = int(train_args.get("orientation_bins", 18))

    h_native, w_native = array.shape[:2]
    rgb = np.asarray(array)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image_np = resized.astype(np.float32) / 255.0
    image_chw = np.transpose(image_np, (2, 0, 1))
    image = torch.from_numpy(image_chw).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(imagenet_normalize(image))
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    heat = probs.max(axis=0).astype(np.float32)
    bins = probs.argmax(axis=0).astype(np.int32)
    gray = _to_gray(image_chw)
    seed_mask = heat >= threshold
    labels, label_count = ndimage.label(seed_mask)

    proposals: list[dict[str, Any]] = []
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
        x1, y1, x2, y2 = _crop_bounds(xs, ys, image_size, image_size, crop_padding)
        refined_angle = _refine_angle_radon(
            gray_crop=gray[y1:y2, x1:x2],
            seed_angle_deg=seed_angle,
            search_degrees=radon_search_degrees,
            step_degrees=radon_step_degrees,
        )
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
        input_start, input_end = segment
        if min_line_support > 0.0:
            lsr = _line_support_ratio(component, heat, input_start, input_end)
            if lsr < min_line_support:
                continue
        native_start = _Point(input_start.x * w_native / image_size, input_start.y * h_native / image_size)
        native_end = _Point(input_end.x * w_native / image_size, input_end.y * h_native / image_size)
        obb = _line_to_obb(native_start, native_end, compat_width_px)
        confidence = float(heat[component].max())
        proposals.append(
            {
                "bbox": [
                    float(min(native_start.x, native_end.x)),
                    float(min(native_start.y, native_end.y)),
                    float(max(native_start.x, native_end.x)),
                    float(max(native_start.y, native_end.y)),
                ],
                "confidence": confidence,
                "method": "dinov3_heatmap_centerline",
                "geometry_type": "line_segment",
                "line_segment": {
                    "x1": float(native_start.x),
                    "y1": float(native_start.y),
                    "x2": float(native_end.x),
                    "y2": float(native_end.y),
                    "angle_deg": float(obb["angle_deg"]),
                    "length_px": float(obb["w"]),
                },
                "obb_compat": dict(obb),
                "obb": dict(obb),
                "streak_length_px": float(obb["w"]),
                "heatmap": {
                    "component_id": int(label_id),
                    "score": confidence,
                    "mean_score": float(heat[component].mean()),
                    "area_px": area,
                    "dominant_bin": dominant_bin,
                    "orientation_consistency": consistency,
                    "seed_angle_deg": float(seed_angle),
                    "radon_angle_deg": float(refined_angle),
                },
            }
        )

    proposals.sort(key=lambda item: (item["confidence"], item["heatmap"]["area_px"]), reverse=True)
    logger.debug("Heatmap centerline: %d segment proposal(s)", len(proposals))
    return proposals[:max_components]
