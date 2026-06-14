"""ViT-B/16 cached-head heatmap detector for the ARGUS pipeline.

Parallel to ``inference/vits_heatmap_detector.py`` but targets the ViT-B/16
checkpoint.  Model loading is delegated to the same
``vits_heatmap_detector._load_model`` since that function reads backbone size
and weights path from the checkpoint's ``train_cache_metadata`` block.

Environment variables
---------------------
VITB_HEATMAP_CHECKPOINT
    Path to the ``best.pt`` checkpoint produced by train_window_v2_pipeline.sh (ViT-B).
    Default: ``weights/vitb_window_v2/best.pt``
VITB_HEATMAP_THRESHOLD
    Heatmap binarisation threshold (float, default 0.5).
VITB_HEATMAP_MIN_PIXELS
    Minimum component size in feature-map pixels (int, default 2).
VITB_HEATMAP_NATIVE_TILE_SIZE
    Native-pixel tile size used during tiled inference (int, default 400).
    Must match the ``native_tile_size`` used when caching training features.
VITB_HEATMAP_TILE_OVERLAP
    Fractional overlap between adjacent tiles (float, default 0.5).
VITB_HEATMAP_MAX_LONG_EDGE
    Downscale the image so its longest edge does not exceed this value before
    tiling (int, default 0 = disabled).  Reduces tile count quadratically;
    e.g. halving a 6k-wide frame cuts ~620 tiles to ~155.  Detections are
    rescaled back to original coordinates after inference.
VITB_HEATMAP_TILE_BATCH_SIZE
    Number of tiles processed per model forward call (int, default 4).
    Higher values reduce MPS kernel-launch overhead but increase peak memory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

# Re-use the model loader and backbone-agnostic tile helpers from ViT-S module.
# _load_model is checkpoint-agnostic: it reads model_size from the checkpoint's
# train_cache_metadata, so ViT-B checkpoints load correctly via the same path.
from inference.vits_heatmap_detector import _load_model, _filter_peak_topk
from inference.convnext_heatmap_detector import (
    _run_single_tile,
    _run_single_tile_probs,
    _remap_detection,
    _rescale_detections,
    _run_tile_batch_full,
)

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "weights"
    / "vitb_window_v2"
    / "best.pt"
)


def _default_checkpoint() -> Path:
    return Path(os.environ.get("VITB_HEATMAP_CHECKPOINT", str(_DEFAULT_CHECKPOINT)))


def get_vitb_heatmap_status() -> dict[str, str]:
    """Return availability metadata for the ViT-B heatmap detector."""
    ckpt = _default_checkpoint()
    return {
        "id":      "vitb_heatmap",
        "name":    "DINOv3 ViT-B HeatMap",
        "type":    "ml",
        "dataset": "Atwood+Frigate Run17",
        "status":  "active" if ckpt.exists() else "no_weights",
    }


def run_vitb_heatmap_detector(
    array: np.ndarray,
    checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the ViT-B heatmap detector on a single image with tiling.

    Args:
        array: uint8 RGB array, shape ``(H, W, 3)``.
        checkpoint: Override checkpoint path (else uses env var / default).

    Returns:
        Pipeline-compatible detection dicts.
    """
    ckpt_path = Path(checkpoint) if checkpoint else _default_checkpoint()
    threshold        = float(os.environ.get("VITB_HEATMAP_THRESHOLD", "0.5"))
    min_pixels       = int(os.environ.get("VITB_HEATMAP_MIN_PIXELS", "2"))
    native_tile_size = int(os.environ.get("VITB_HEATMAP_NATIVE_TILE_SIZE", "400"))
    tile_overlap     = float(os.environ.get("VITB_HEATMAP_TILE_OVERLAP", "0.5"))
    peak_floor       = float(os.environ.get("VITB_HEATMAP_PEAK_FLOOR", "0.0"))
    top_k            = int(os.environ.get("VITB_HEATMAP_TOPK", "0"))
    max_long_edge    = int(os.environ.get("VITB_HEATMAP_MAX_LONG_EDGE", "0"))

    try:
        model, image_size, device, use_geometry = _load_model(ckpt_path)
    except Exception as exc:
        logger.warning("ViT-B heatmap model load failed: %s", exc)
        return []

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    h, w = array.shape[:2]
    prescale = 1.0
    if max_long_edge > 0 and max(h, w) > max_long_edge:
        import cv2
        prescale = max_long_edge / max(h, w)
        array = cv2.resize(array, (int(w * prescale), int(h * prescale)), interpolation=cv2.INTER_AREA)
        h, w = array.shape[:2]
        logger.info("ViT-B heatmap: prescaled to %dx%d (scale=%.3f)", w, h, prescale)

    if max(h, w) <= native_tile_size:
        dets = _run_single_tile(array, model, image_size, device, threshold, min_pixels,
                                use_geometry=use_geometry)
        for d in dets:
            d["method"] = "vitb_heatmap"
        dets = _filter_peak_topk(dets, peak_floor, top_k)
        return _rescale_detections(dets, 1.0 / prescale) if prescale != 1.0 else dets

    from inference.tiled_pipeline import tile_image
    from inference.postprocess import nms_detections

    all_dets: list[dict[str, Any]] = []
    for tile, x0, y0 in tile_image(array, native_tile_size, tile_overlap):
        for det in _run_single_tile(tile, model, image_size, device, threshold, min_pixels,
                                    use_geometry=use_geometry):
            d = _remap_detection(det, x0, y0)
            d["method"] = "vitb_heatmap"
            all_dets.append(d)

    if len(all_dets) <= 1:
        dets = _filter_peak_topk(all_dets, peak_floor, top_k)
        return _rescale_detections(dets, 1.0 / prescale) if prescale != 1.0 else dets

    result = _filter_peak_topk(nms_detections(all_dets), peak_floor, top_k)
    logger.debug(
        "ViT-B heatmap (tiled): %d raw → %d after segment NMS + peak/top-K",
        len(all_dets), len(result)
    )
    return _rescale_detections(result, 1.0 / prescale) if prescale != 1.0 else result


def run_vitb_heatmap_detector_and_heatmap(
    array: np.ndarray,
    checkpoint: Path | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """Run the ViT-B heatmap detector and return both detections and a stitched
    full-image probability heatmap for the overlay.

    Returns:
        Tuple of (detections, heat) where ``heat`` is a float32 array shaped
        ``(H, W)`` with values in [0, 1], or None on model-load failure.
    """
    ckpt_path = Path(checkpoint) if checkpoint else _default_checkpoint()
    threshold        = float(os.environ.get("VITB_HEATMAP_THRESHOLD", "0.5"))
    min_pixels       = int(os.environ.get("VITB_HEATMAP_MIN_PIXELS", "2"))
    native_tile_size = int(os.environ.get("VITB_HEATMAP_NATIVE_TILE_SIZE", "400"))
    tile_overlap     = float(os.environ.get("VITB_HEATMAP_TILE_OVERLAP", "0.5"))
    peak_floor       = float(os.environ.get("VITB_HEATMAP_PEAK_FLOOR", "0.0"))
    top_k            = int(os.environ.get("VITB_HEATMAP_TOPK", "0"))
    max_long_edge    = int(os.environ.get("VITB_HEATMAP_MAX_LONG_EDGE", "0"))
    tile_batch_size  = int(os.environ.get("VITB_HEATMAP_TILE_BATCH_SIZE", "4"))

    try:
        model, image_size, device, use_geometry = _load_model(ckpt_path)
    except Exception as exc:
        logger.warning("ViT-B heatmap model load failed (heatmap): %s", exc)
        return [], None

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    h_orig, w_orig = array.shape[:2]
    prescale = 1.0
    if max_long_edge > 0 and max(h_orig, w_orig) > max_long_edge:
        import cv2
        prescale = max_long_edge / max(h_orig, w_orig)
        array = cv2.resize(array, (int(w_orig * prescale), int(h_orig * prescale)), interpolation=cv2.INTER_AREA)
        logger.info("ViT-B heatmap: prescaled to %dx%d (scale=%.3f)", array.shape[1], array.shape[0], prescale)

    h_full, w_full = array.shape[:2]
    heat_full = np.zeros((h_full, w_full), dtype=np.float32)

    if max(h_full, w_full) <= native_tile_size:
        tile_results = _run_tile_batch_full(
            [(array, 0, 0)], model, image_size, device, threshold, min_pixels, use_geometry, tile_batch_size
        )
        dets_tile, heat_full, _, _, _, _ = tile_results[0]
        for d in dets_tile:
            d["method"] = "vitb_heatmap"
        dets_tile = _filter_peak_topk(dets_tile, peak_floor, top_k)
        if prescale != 1.0:
            import cv2
            heat_full = cv2.resize(heat_full, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            dets_tile = _rescale_detections(dets_tile, 1.0 / prescale)
        return dets_tile, heat_full

    from inference.tiled_pipeline import tile_image
    from inference.postprocess import nms_detections

    tiles = list(tile_image(array, native_tile_size, tile_overlap))
    all_dets: list[dict[str, Any]] = []
    for tile_dets, heat_tile, x0, y0, th, tw in _run_tile_batch_full(
        tiles, model, image_size, device, threshold, min_pixels, use_geometry, tile_batch_size
    ):
        for det in tile_dets:
            d = _remap_detection(det, x0, y0)
            d["method"] = "vitb_heatmap"
            all_dets.append(d)
        y1e, x1e = min(y0 + th, h_full), min(x0 + tw, w_full)
        np.maximum(heat_full[y0:y1e, x0:x1e], heat_tile[:y1e - y0, :x1e - x0],
                   out=heat_full[y0:y1e, x0:x1e])

    result = _filter_peak_topk(nms_detections(all_dets) if len(all_dets) > 1 else all_dets, peak_floor, top_k)
    if prescale != 1.0:
        import cv2
        heat_full = cv2.resize(heat_full, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        result = _rescale_detections(result, 1.0 / prescale)
    return result, heat_full
