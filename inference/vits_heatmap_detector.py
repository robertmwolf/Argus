"""ViT-S/16 cached-head heatmap detector for the ARGUS pipeline.

Parallel to ``inference/convnext_heatmap_detector.py`` but uses the DINOv3
ViT-S/16 backbone instead of ConvNeXt-S.  The HeatmapHead format, tile loop,
and NMS logic are identical — only the feature encoder differs.

Environment variables
---------------------
VITS_HEATMAP_CHECKPOINT
    Path to the ``best.pt`` checkpoint produced by
    ``training/train_dinov3_heatmap_cached.py`` with ``--backbone vit``.
    Default: ``weights/run5_vits_heatmap_cached/best.pt``
VITS_HEATMAP_THRESHOLD
    Heatmap binarisation threshold (float, default 0.5).
VITS_HEATMAP_IMAGE_SIZE
    Square input size in pixels (int, default 384).
VITS_HEATMAP_MIN_PIXELS
    Minimum component size in feature-map pixels (int, default 2).
VITS_HEATMAP_NATIVE_TILE_SIZE
    Native-pixel tile size used during tiled inference (int, default 400).
    Must match the ``native_tile_size`` used when caching training features.
VITS_HEATMAP_TILE_OVERLAP
    Fractional overlap between adjacent tiles (float, default 0.5).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference.device import get_device
# Re-use the backbone-agnostic helpers from the ConvNeXt detector
from inference.convnext_heatmap_detector import (
    _run_single_tile,
    _remap_detection,
)

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, tuple[Any, int, torch.device]] = {}

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "weights"
    / "run5_vits_heatmap_cached"
    / "best.pt"
)


def _default_checkpoint() -> Path:
    return Path(os.environ.get("VITS_HEATMAP_CHECKPOINT", str(_DEFAULT_CHECKPOINT)))


def _load_model(checkpoint_path: Path) -> tuple[Any, int, torch.device]:
    """Load and cache the ViT-S heatmap model."""
    device = get_device()
    cache_key = str(checkpoint_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"ViT-S heatmap checkpoint not found: {checkpoint_path}"
        )

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
    head = HeatmapHead(in_channels, hidden)
    head.load_state_dict(ckpt["head"])
    backbone.head = head.net.to(device)
    backbone.eval()

    result = (backbone, image_size, device)
    _MODEL_CACHE[cache_key] = result
    return result


def run_vits_heatmap_detector(
    array: np.ndarray,
    checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the ViT-S heatmap detector on a single image with tiling.

    Tiles the image using ``VITS_HEATMAP_NATIVE_TILE_SIZE`` (default 400 px)
    to match the scale the model was trained on.  Uses 50% overlap between
    tiles and NMS to deduplicate cross-tile detections.

    Args:
        array: uint8 RGB array, shape ``(H, W, 3)``.
        checkpoint: Override checkpoint path (else uses env var / default).

    Returns:
        Pipeline-compatible detection dicts with ``obb``, ``confidence``,
        ``bbox``, ``streak_length_px``, and ``method`` keys.
    """
    ckpt_path = Path(checkpoint) if checkpoint else _default_checkpoint()
    threshold        = float(os.environ.get("VITS_HEATMAP_THRESHOLD", "0.5"))
    min_pixels       = int(os.environ.get("VITS_HEATMAP_MIN_PIXELS", "2"))
    native_tile_size = int(os.environ.get("VITS_HEATMAP_NATIVE_TILE_SIZE", "400"))
    tile_overlap     = float(os.environ.get("VITS_HEATMAP_TILE_OVERLAP", "0.5"))

    try:
        model, image_size, device = _load_model(ckpt_path)
    except Exception as exc:
        logger.warning("ViT-S heatmap model load failed: %s", exc)
        return []

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    h, w = array.shape[:2]

    if max(h, w) <= native_tile_size:
        dets = _run_single_tile(array, model, image_size, device, threshold, min_pixels)
        for d in dets:
            d["method"] = "vits_heatmap"
        return dets

    from inference.tiled_pipeline import tile_image, _torchvision_nms, _numpy_nms

    all_dets: list[dict[str, Any]] = []
    for tile, x0, y0 in tile_image(array, native_tile_size, tile_overlap):
        for det in _run_single_tile(tile, model, image_size, device, threshold, min_pixels):
            d = _remap_detection(det, x0, y0)
            d["method"] = "vits_heatmap"
            all_dets.append(d)

    if len(all_dets) <= 1:
        return all_dets

    preds_xywh = [
        {
            "bbox":        [d["bbox"][0], d["bbox"][1],
                            d["bbox"][2] - d["bbox"][0],
                            d["bbox"][3] - d["bbox"][1]],
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
    logger.debug(
        "ViT-S heatmap (tiled): %d raw → %d after NMS", len(all_dets), len(result)
    )
    return result
