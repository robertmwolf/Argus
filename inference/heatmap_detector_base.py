"""Shared utilities for DINOv3 heatmap detectors.

Provides model loading, tile runners, and detection helpers used by
``vits_window_v9_detector`` and ``vitb_window_v10_detector``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from inference.device import get_device

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, tuple[Any, int, torch.device, bool]] = {}


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def _centerline_head_state(head_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize cached-head keys and retain only the centerline output channel."""
    normalized = {
        (key if key.startswith("net.") else f"net.{key}"): value
        for key, value in head_state.items()
    }
    final_weight = normalized.get("net.4.weight")
    if final_weight is None:
        raise ValueError("checkpoint heatmap head is missing its final convolution")
    normalized["net.4.weight"] = final_weight[:1].clone()
    final_bias = normalized.get("net.4.bias")
    if final_bias is not None:
        normalized["net.4.bias"] = final_bias[:1].clone()
    return normalized


def _load_model(checkpoint_path: Path) -> tuple[Any, int, torch.device, bool]:
    """Load and cache a ViT heatmap model from a cached-head checkpoint.

    Returns:
        Tuple of (model, image_size, device, use_geometry).
    """
    device = get_device()
    cache_key = str(checkpoint_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Heatmap checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_meta = ckpt.get("train_cache_metadata", {})
    weights    = train_meta.get("weights", "weights/dinov3_vits16_lvd1689m.pth")
    model_size = train_meta.get("model_size", "small")
    image_size = int(train_meta.get("image_size", 384))

    from models.plain_dinov3.streak_heatmap import DINOv3StreakHeatmap
    from training.train_dinov3_heatmap_cached import HeatmapHead

    backbone = DINOv3StreakHeatmap(
        model_size=model_size,
        weights=weights,
    ).to(device)

    hidden      = int(ckpt.get("args", {}).get("hidden_channels", 256))
    in_channels = int(ckpt["in_channels"])
    head_state  = ckpt["head"]
    runtime_head_state = _centerline_head_state(head_state)
    head = HeatmapHead(in_channels, hidden, out_channels=1)
    head.load_state_dict(runtime_head_state)
    backbone.head = head.net.to(device)
    backbone.eval()

    result = (backbone, image_size, device, False)
    _MODEL_CACHE[cache_key] = result
    return result


def _filter_peak_topk(dets: list[dict[str, Any]], peak_floor: float, top_k: int) -> list[dict[str, Any]]:
    """Apply peak-floor / top-K noise filter (no-op when both gates are off)."""
    if peak_floor <= 0.0 and top_k <= 0:
        return dets
    from inference.postprocess import filter_peak_topk
    return filter_peak_topk(dets, peak_floor=peak_floor, top_k=top_k)


# ---------------------------------------------------------------------------
# Tile runners
# ---------------------------------------------------------------------------


def _letterbox(array: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize image to a square canvas preserving aspect ratio with letterboxing.

    Returns:
        Tuple of (canvas_rgb_uint8, scale, pad_x, pad_y).
    """
    h, w = array.shape[:2]
    scale = min(size / w, size / h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    resized = np.array(Image.fromarray(array).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = round(pad_y), round(pad_x)
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, float(scale), float(pad_x), float(pad_y)


def _component_to_segment(
    mask: np.ndarray,
    score_map: np.ndarray,
    patch_size: int,
    image_size: int,
) -> dict[str, Any] | None:
    """Fit a line segment to a connected heatmap component via PCA.

    Args:
        mask: Boolean mask of active feature-map pixels for this component.
        score_map: Per-pixel probability map from the heatmap head.
        patch_size: Feature-map patch stride in source pixels.
        image_size: Square canvas side length (used to de-normalise geometry).

    Returns:
        Detection dict, or None if the component has fewer than 2 pixels.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    pts = np.column_stack([(xs + 0.5) * patch_size, (ys + 0.5) * patch_size]).astype(np.float32)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    major = vecs[:, order[0]]
    rel = pts - center
    length = max(float((rel @ major).ptp()) + patch_size, patch_size)
    angle  = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0

    half = length / 2.0
    x1 = float(center[0]) - half * float(major[0])
    y1 = float(center[1]) - half * float(major[1])
    x2 = float(center[0]) + half * float(major[0])
    y2 = float(center[1]) + half * float(major[1])

    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "confidence": float(score_map[mask].mean()),
        "peak_confidence": float(score_map[mask].max()),
        "streak_length_px": length,
        "cx": float(center[0]),
        "cy": float(center[1]),
        "angle_deg": angle,
    }


def _run_single_tile(
    array: np.ndarray,
    model: Any,
    image_size: int,
    device: torch.device,
    threshold: float,
    min_pixels: int,
    use_geometry: bool = False,
) -> list[dict[str, Any]]:
    """Run the detector on one tile; return detections in tile-local coordinates."""
    from models.plain_dinov3.streak_heatmap import imagenet_normalize

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    canvas, scale, pad_x, pad_y = _letterbox(array, image_size)
    img_tensor = (
        torch.from_numpy(canvas.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        output = model(imagenet_normalize(img_tensor))
        probs  = torch.sigmoid(output[:, :1])[0, 0].cpu().numpy().astype(np.float32)

    patch_size = 16
    binary = probs >= threshold
    labels, n_labels = ndimage.label(binary)
    detections: list[dict[str, Any]] = []

    for label_id in range(1, n_labels + 1):
        mask = labels == label_id
        if int(mask.sum()) < min_pixels:
            continue
        det = _component_to_segment(mask, probs, patch_size, image_size)
        if det is None:
            continue
        det["x1"] = (det["x1"] - pad_x) / scale
        det["y1"] = (det["y1"] - pad_y) / scale
        det["x2"] = (det["x2"] - pad_x) / scale
        det["y2"] = (det["y2"] - pad_y) / scale
        from inference.streak_segment import apply_segment_geometry
        apply_segment_geometry(det)
        detections.append({
            "confidence":       det["confidence"],
            "peak_confidence":  det["peak_confidence"],
            "method":           "heatmap",
            "x1":               det["x1"],
            "y1":               det["y1"],
            "x2":               det["x2"],
            "y2":               det["y2"],
            "cx":               det["cx"],
            "cy":               det["cy"],
            "angle_deg":        det["angle_deg"],
            "streak_length_px": det["streak_length_px"],
        })
    return detections


def _run_single_tile_probs(
    array: np.ndarray,
    model: Any,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float, float, float]:
    """Run model on one tile; return the raw probability map in tile-pixel space.

    Returns:
        Tuple of (heat_tile, scale, pad_x, pad_y).
    """
    import cv2 as _cv2
    from models.plain_dinov3.streak_heatmap import imagenet_normalize

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    h, w = array.shape[:2]
    canvas, scale, pad_x, pad_y = _letterbox(array, image_size)
    img_tensor = (
        torch.from_numpy(canvas.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        output = model(imagenet_normalize(img_tensor))
        probs = torch.sigmoid(output[:, :1])[0, 0].cpu().numpy().astype(np.float32)

    probs_canvas = _cv2.resize(probs, (image_size, image_size), interpolation=_cv2.INTER_LINEAR)
    y0r, x0r = round(pad_y), round(pad_x)
    new_h, new_w = round(h * scale), round(w * scale)
    probs_content = probs_canvas[y0r:y0r + new_h, x0r:x0r + new_w]
    heat_tile = _cv2.resize(probs_content, (w, h), interpolation=_cv2.INTER_LINEAR)
    return heat_tile, scale, pad_x, pad_y


def _remap_detection(det: dict[str, Any], x0: int, y0: int) -> dict[str, Any]:
    """Shift tile-local detection coordinates to full-image coordinates."""
    det = dict(det)
    det["x1"] += x0
    det["y1"] += y0
    det["x2"] += x0
    det["y2"] += y0
    from inference.streak_segment import apply_segment_geometry
    apply_segment_geometry(det)
    return det


def _rescale_detections(dets: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    """Scale detection coordinates from a downscaled image back to original size."""
    out = []
    for det in dets:
        det = dict(det)
        det["x1"] *= scale
        det["y1"] *= scale
        det["x2"] *= scale
        det["y2"] *= scale
        from inference.streak_segment import apply_segment_geometry
        apply_segment_geometry(det)
        out.append(det)
    return out
