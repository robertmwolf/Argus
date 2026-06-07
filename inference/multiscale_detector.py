"""Multi-scale heatmap detector for ARGUS.

Runs the ViT-S or ConvNeXt-S heatmap detector at multiple native tile sizes and
merges detections via IoU NMS.  This handles images where streaks may appear at
different apparent lengths depending on sensor resolution.

Scale selection rationale
-------------------------
A streak of native length L appears at the model input as:

    apparent_px = L * (model_input_size / native_tile_size)

Run 12 was trained on 1800px tiles → 518px model input (scale = 0.288×):
  - 900px Atwood trail  → 259px apparent  (long)
  - 300px short trail   →  86px apparent  (medium)
  - 60px Frigate streak → 282px apparent  (long, via 110px zoom)

For inference, use multiple tile sizes so the model sees streaks in the
size range it was trained on regardless of sensor:

  1800px tiles  → long Atwood trails appear as medium/long (200–400px)
   518px tiles  → medium trails fully contained (80–518px)
   110px tiles  → short Frigate streaks magnified to 38–280px apparent

Usage
-----
    from pathlib import Path
    from inference.multiscale_detector import run_multiscale_detector

    dets = run_multiscale_detector(
        array,
        checkpoint=Path("weights/run12_vits/best.pt"),
        backbone="vit",
        scales=[1800, 518],
    )

    # Auto-select scales based on image size:
    dets = run_multiscale_detector(array, checkpoint=..., backbone="vit")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def choose_scales(image_shape: tuple[int, ...], model_input_size: int = 518) -> list[int]:
    """Auto-select native tile sizes for multi-scale inference.

    Args:
        image_shape:      (H, W, ...) of the input image.
        model_input_size: The model's input resolution (from checkpoint metadata).

    Returns:
        Tile sizes in descending order (large tiles first so long streaks are
        found first; NMS then suppresses lower-confidence duplicates).
    """
    h, w = image_shape[:2]
    max_dim = max(h, w)

    scales: set[int] = set()

    # Large images (Atwood-style ≥ 2500px): 1800px tiles for medium/long streaks
    if max_dim >= 2500:
        scales.add(1800)

    # Mid-range: native model input size covers 80–518px apparent streaks
    scales.add(model_input_size)

    # Small images (< 2000px) or when short-streak detection is needed: 110px zoom
    if max_dim < 2000:
        scales.add(110)

    return sorted(scales, reverse=True)


def run_multiscale_detector(
    array: np.ndarray,
    checkpoint: Path | str,
    backbone: str = "vit",
    scales: list[int] | None = None,
    threshold: float = 0.5,
    min_pixels: int = 2,
    iou_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Run the heatmap detector at multiple native tile sizes and merge via NMS.

    Each scale runs the full single-scale detector (tiling, per-scale NMS).
    All resulting detections — already in image-space coordinates — are then
    pooled and de-duplicated with a final cross-scale NMS pass.

    Args:
        array:         Image array (H×W or H×W×3), any dtype.
        checkpoint:    Path to ``best.pt`` produced by
                       ``train_dinov3_heatmap_cached.py``.
        backbone:      ``"vit"`` or ``"convnext"``.
        scales:        Native tile sizes in pixels.  ``None`` → auto via
                       :func:`choose_scales` based on image dimensions and
                       the checkpoint's ``image_size`` metadata.
        threshold:     Heatmap binarisation threshold (default 0.5).
        min_pixels:    Minimum component size in feature-map patches (default 2).
        iou_threshold: IoU threshold for cross-scale NMS (default 0.3).

    Returns:
        Detection dicts with ``obb``, ``confidence``, ``bbox`` [x1,y1,x2,y2],
        ``streak_length_px``, ``method``, and ``native_scale`` (tile size that
        produced the detection) keys.
    """
    import torch
    from inference.tiled_pipeline import nms_predictions

    checkpoint = Path(checkpoint)

    if scales is None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_input_size = int(
            ckpt.get("train_cache_metadata", {}).get("image_size", 518)
        )
        scales = choose_scales(array.shape, model_input_size)
        logger.info(
            "Auto-selected scales %s for %dx%d image (model_input=%dpx)",
            scales, array.shape[1], array.shape[0], model_input_size,
        )

    # Save and restore env vars we temporarily override
    _saved: dict[str, str | None] = {}

    def _setenv(key: str, val: str) -> None:
        _saved.setdefault(key, os.environ.get(key))
        os.environ[key] = val

    def _restore() -> None:
        for key, val in _saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    all_dets: list[dict[str, Any]] = []

    try:
        for scale in scales:
            if backbone == "vit":
                _setenv("VITS_HEATMAP_NATIVE_TILE_SIZE", str(scale))
                _setenv("VITS_HEATMAP_THRESHOLD",        str(threshold))
                _setenv("VITS_HEATMAP_MIN_PIXELS",       str(min_pixels))
                from inference.vits_heatmap_detector import run_vits_heatmap_detector
                dets = run_vits_heatmap_detector(array, checkpoint=checkpoint)
            elif backbone == "convnext":
                _setenv("CONVNEXT_HEATMAP_NATIVE_TILE_SIZE", str(scale))
                _setenv("CONVNEXT_HEATMAP_THRESHOLD",        str(threshold))
                _setenv("CONVNEXT_HEATMAP_MIN_PIXELS",       str(min_pixels))
                _setenv("CONVNEXT_HEATMAP_CHECKPOINT",       str(checkpoint))
                from inference.convnext_heatmap_detector import run_convnext_heatmap_detector
                dets = run_convnext_heatmap_detector(array)
            else:
                raise ValueError(f"Unknown backbone: {backbone!r}. Expected 'vit' or 'convnext'.")

            for d in dets:
                d["native_scale"] = scale
            all_dets.extend(dets)
            logger.debug("Scale %dpx → %d detections", scale, len(dets))
    finally:
        _restore()

    if len(all_dets) <= 1:
        return all_dets

    # Convert bbox from [x1,y1,x2,y2] to [x,y,w,h] for nms_predictions
    preds_for_nms = [
        {
            "bbox":        [
                d["bbox"][0],
                d["bbox"][1],
                d["bbox"][2] - d["bbox"][0],
                d["bbox"][3] - d["bbox"][1],
            ],
            "score":       float(d["confidence"]),
            "category_id": 1,
        }
        for d in all_dets
    ]

    kept_indices = nms_predictions(preds_for_nms, iou_threshold=iou_threshold)
    result = [all_dets[i] for i in kept_indices]

    logger.info(
        "Multi-scale NMS: scales=%s  %d raw → %d after NMS",
        scales, len(all_dets), len(result),
    )
    return result
