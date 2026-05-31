"""ConvNeXt-S Stage-2 heatmap detector for the ARGUS pipeline.

Loads the Run 5 frozen ConvNeXt-Small backbone + trained HeatmapHead and
converts the output probability map to pipeline-compatible OBB detections.

Checkpoint format is the cached-head format produced by
``training/train_dinov3_heatmap_cached.py``.

Environment variables
---------------------
CONVNEXT_HEATMAP_CHECKPOINT
    Path to the ``best.pt`` checkpoint.
    Default: ``weights/run5_convnext_small_s2_heatmap/best.pt``
CONVNEXT_HEATMAP_THRESHOLD
    Heatmap binarisation threshold (float, default 0.5).
CONVNEXT_HEATMAP_IMAGE_SIZE
    Square input size in pixels (int, default 384).
CONVNEXT_HEATMAP_MIN_PIXELS
    Minimum component size in feature-map pixels (int, default 2).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import (
    ConvNeXtStreakHeatmap,
    decode_geometry,
    imagenet_normalize,
)

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, tuple[Any, int, torch.device]] = {}

_DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "weights" / "run5_convnext_small_s2_heatmap_pretiled" / "best.pt"


def _default_checkpoint() -> Path:
    return Path(os.environ.get("CONVNEXT_HEATMAP_CHECKPOINT", str(_DEFAULT_CHECKPOINT)))


def get_convnext_heatmap_status() -> dict[str, str]:
    """Return availability metadata for the ConvNeXt heatmap detector."""
    ckpt = _default_checkpoint()
    return {
        "id":      "convnext_heatmap",
        "name":    "ConvNeXt-S HeatMap",
        "type":    "ml",
        "dataset": "Atwood+Frigate Run5",
        "status":  "active" if ckpt.exists() else "no_weights",
    }


def _load_model(checkpoint_path: Path) -> tuple[Any, int, torch.device]:
    """Load and cache the ConvNeXt heatmap model."""
    device = get_device()
    cache_key = str(checkpoint_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ConvNeXt heatmap checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_meta = ckpt.get("train_cache_metadata", {})
    weights    = train_meta.get("weights", "weights/dinov3_convnext_small_pretrain_lvd1689m.pth")
    model_size = train_meta.get("model_size", "small")
    stage      = int(train_meta.get("convnext_stage") or 3)
    image_size = int(train_meta.get("image_size", 384))

    from training.train_dinov3_heatmap_cached import HeatmapHead

    backbone = ConvNeXtStreakHeatmap(
        model_size=model_size,
        weights=weights,
        extract_stage=stage,
        freeze_backbone=True,
    ).to(device)

    hidden     = int(ckpt.get("args", {}).get("hidden_channels", 256))
    in_channels = int(ckpt["in_channels"])
    head = HeatmapHead(in_channels, hidden)
    head.load_state_dict(ckpt["head"])
    backbone.head = head.net.to(device)
    backbone.eval()

    result = (backbone, image_size, device)
    _MODEL_CACHE[cache_key] = result
    return result


def _letterbox(array: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize image to square canvas with aspect-ratio-preserving letterbox.

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


def _component_to_obb(
    mask: np.ndarray,
    score_map: np.ndarray,
    patch_size: int,
    geometry_map: np.ndarray | None,
    image_size: int,
) -> dict[str, Any] | None:
    """Fit an OBB to a connected heatmap component via PCA."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    pts = np.column_stack([(xs + 0.5) * patch_size, (ys + 0.5) * patch_size]).astype(np.float32)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    major = vecs[:, order[0]]
    minor = vecs[:, order[1]]
    rel = pts - center
    length = max(float((rel @ major).ptp()) + patch_size, patch_size)
    width  = max(float((rel @ minor).ptp()) + patch_size, patch_size)
    angle  = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0

    if geometry_map is not None:
        geom = geometry_map[:, mask]
        if geom.shape[1] > 0:
            cos2, sin2 = float(geom[0].mean()), float(geom[1].mean())
            if abs(cos2) + abs(sin2) > 1e-3:
                angle = (0.5 * math.degrees(math.atan2(sin2, cos2))) % 180.0
            length = max(float(geom[2].mean()) * image_size, patch_size)
            width  = max(float(geom[3].mean()) * image_size, patch_size)

    return {
        "confidence": float(score_map[mask].mean()),
        "obb": {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "w":  length,
            "h":  width,
            "angle_deg": angle,
        },
        "streak_length_px": length,
    }


def _run_single_tile(
    array: np.ndarray,
    model: Any,
    image_size: int,
    device: torch.device,
    threshold: float,
    min_pixels: int,
) -> list[dict[str, Any]]:
    """Run detector on one tile; return detections in tile-local coordinates."""
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    canvas, scale, pad_x, pad_y = _letterbox(array, image_size)
    img_tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output   = model(imagenet_normalize(img_tensor))
        logits   = output[:, :1]
        probs    = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
        geometry = decode_geometry(output[:, 1:5])[0].cpu().numpy() if output.shape[1] >= 5 else None

    patch_size = 16  # ConvNeXt stage-2 stride equals ViT-S/16 patch stride
    binary = probs >= threshold
    labels, n_labels = ndimage.label(binary)
    detections: list[dict[str, Any]] = []

    for label_id in range(1, n_labels + 1):
        mask = labels == label_id
        if int(mask.sum()) < min_pixels:
            continue
        det = _component_to_obb(mask, probs, patch_size, geometry, image_size)
        if det is None:
            continue
        obb = det["obb"]
        obb["cx"]  = (obb["cx"] - pad_x) / scale
        obb["cy"]  = (obb["cy"] - pad_y) / scale
        obb["w"]  /= scale
        obb["h"]  /= scale
        det["streak_length_px"] = max(float(obb["w"]), float(obb["h"]))
        cx, cy = obb["cx"], obb["cy"]
        hw, hh = obb["w"] / 2, obb["h"] / 2
        detections.append({
            "bbox":             [cx - hw, cy - hh, cx + hw, cy + hh],
            "confidence":       det["confidence"],
            "method":           "convnext_heatmap",
            "obb":              obb,
            "streak_length_px": det["streak_length_px"],
        })
    return detections


def _remap_detection(det: dict[str, Any], x0: int, y0: int) -> dict[str, Any]:
    """Shift tile-local detection coordinates to full-image coordinates."""
    det = dict(det)
    b = det["bbox"]
    det["bbox"] = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
    obb = {**det["obb"], "cx": det["obb"]["cx"] + x0, "cy": det["obb"]["cy"] + y0}
    det["obb"] = obb
    return det


def run_convnext_heatmap_detector(array: np.ndarray) -> list[dict[str, Any]]:
    """Run the ConvNeXt heatmap detector on a single image with tiling.

    Tiles the image using ``CONVNEXT_HEATMAP_NATIVE_TILE_SIZE`` (default 1562 px)
    so that medium streaks (~150–400 px native) span 3–8 feature patches rather
    than <2 patches in a full-image 384 px resize.  Uses 50 % overlap between
    tiles and NMS to deduplicate cross-tile detections.

    Args:
        array: uint8 RGB array, shape ``(H, W, 3)``.

    Returns:
        Pipeline-compatible detection dicts with ``obb``, ``confidence``,
        ``bbox``, ``streak_length_px``, and ``method`` keys.
    """
    checkpoint = _default_checkpoint()
    if not checkpoint.exists():
        logger.debug("ConvNeXt heatmap checkpoint not found at %s; skipping", checkpoint)
        return []

    threshold        = float(os.environ.get("CONVNEXT_HEATMAP_THRESHOLD", "0.5"))
    min_pixels       = int(os.environ.get("CONVNEXT_HEATMAP_MIN_PIXELS", "2"))
    # 1562 px tiles on 6248 px Atwood images → 4 tiles/row, giving a 300 px
    # medium streak ~74 px at model input (4.6 patches vs 1.2 without tiling).
    native_tile_size = int(os.environ.get("CONVNEXT_HEATMAP_NATIVE_TILE_SIZE", "1562"))
    tile_overlap     = float(os.environ.get("CONVNEXT_HEATMAP_TILE_OVERLAP", "0.5"))

    try:
        model, image_size, device = _load_model(checkpoint)
    except Exception as exc:
        logger.warning("ConvNeXt heatmap model load failed: %s", exc)
        return []

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    h, w = array.shape[:2]

    if max(h, w) <= native_tile_size:
        dets = _run_single_tile(array, model, image_size, device, threshold, min_pixels)
        logger.debug("ConvNeXt heatmap (single shot): %d detection(s)", len(dets))
        return dets

    from inference.tiled_pipeline import tile_image, _torchvision_nms, _numpy_nms

    all_dets: list[dict[str, Any]] = []
    for tile, x0, y0 in tile_image(array, native_tile_size, tile_overlap):
        for det in _run_single_tile(tile, model, image_size, device, threshold, min_pixels):
            all_dets.append(_remap_detection(det, x0, y0))

    if len(all_dets) <= 1:
        logger.debug("ConvNeXt heatmap (tiled): %d detection(s)", len(all_dets))
        return all_dets

    preds_xywh = [
        {
            "bbox":        [d["bbox"][0], d["bbox"][1],
                            d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]],
            "score":       float(d["confidence"]),
            "category_id": 1,
        }
        for d in all_dets
    ]
    try:
        kept = _torchvision_nms(preds_xywh, iou_threshold=0.3)
    except Exception:
        kept = _numpy_nms(preds_xywh, iou_threshold=0.3)

    result = [all_dets[i] for i in kept]
    logger.debug("ConvNeXt heatmap (tiled): %d raw → %d after NMS", len(all_dets), len(result))
    return result
