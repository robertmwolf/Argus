"""ViT-B/16 window_v4 heatmap detector.

Identical runtime to ``inference/vitb_window_v3_detector.py`` but targets the
window_v4 checkpoint and uses the ``VITB_V4_*`` env-var namespace.

Environment variables
---------------------
VITB_V4_HEATMAP_CHECKPOINT
    Default: ``weights/vitb_window_v4/best.pt``
VITB_V4_HEATMAP_THRESHOLD
    Heatmap binarisation threshold (float, default 0.65).
VITB_V4_HEATMAP_MIN_PIXELS
    Minimum component size in feature-map pixels (int, default 2).
VITB_V4_HEATMAP_NATIVE_TILE_SIZE
    Native-pixel tile size (int, default 400).
VITB_V4_HEATMAP_TILE_OVERLAP
    Fractional overlap between adjacent tiles (float, default 0.5).
VITB_V4_HEATMAP_PEAK_FLOOR
    Minimum peak heatmap value for a detection to survive (float, default 0.85).
VITB_V4_HEATMAP_NORM
    Normalisation mode applied to raw FITS before inference (default ``zscore``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from inference.vits_heatmap_detector import _load_model, _filter_peak_topk
from inference.convnext_heatmap_detector import (
    _run_single_tile,
    _run_single_tile_probs,
    _remap_detection,
)

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent / "weights" / "vitb_window_v4" / "best.pt"
)


def _default_checkpoint() -> Path:
    return Path(os.environ.get("VITB_V4_HEATMAP_CHECKPOINT", str(_DEFAULT_CHECKPOINT)))


def get_vitb_v4_heatmap_status() -> dict[str, str]:
    ckpt = _default_checkpoint()
    return {
        "id":      "vitb_heatmap_v4",
        "name":    "DINOv3 ViT-B HeatMap v4",
        "type":    "ml",
        "dataset": "Atwood window_v4",
        "status":  "active" if ckpt.exists() else "no_weights",
    }


def run_vitb_v4_heatmap_detector(
    array: np.ndarray,
    checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    ckpt_path = Path(checkpoint) if checkpoint else _default_checkpoint()
    threshold        = float(os.environ.get("VITB_V4_HEATMAP_THRESHOLD", "0.65"))
    min_pixels       = int(os.environ.get("VITB_V4_HEATMAP_MIN_PIXELS", "2"))
    native_tile_size = int(os.environ.get("VITB_V4_HEATMAP_NATIVE_TILE_SIZE", "400"))
    tile_overlap     = float(os.environ.get("VITB_V4_HEATMAP_TILE_OVERLAP", "0.5"))
    peak_floor       = float(os.environ.get("VITB_V4_HEATMAP_PEAK_FLOOR", "0.85"))
    top_k            = int(os.environ.get("VITB_V4_HEATMAP_TOPK", "0"))

    try:
        model, image_size, device, use_geometry = _load_model(ckpt_path)
    except Exception as exc:
        logger.warning("ViT-B v4 heatmap model load failed: %s", exc)
        return []

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    h, w = array.shape[:2]

    if max(h, w) <= native_tile_size:
        dets = _run_single_tile(array, model, image_size, device, threshold, min_pixels,
                                use_geometry=use_geometry)
        for d in dets:
            d["method"] = "vitb_heatmap_v4"
        return _filter_peak_topk(dets, peak_floor, top_k)

    from inference.tiled_pipeline import tile_image
    from inference.postprocess import nms_detections

    all_dets: list[dict[str, Any]] = []
    for tile, x0, y0 in tile_image(array, native_tile_size, tile_overlap):
        for det in _run_single_tile(tile, model, image_size, device, threshold, min_pixels,
                                    use_geometry=use_geometry):
            d = _remap_detection(det, x0, y0)
            d["method"] = "vitb_heatmap_v4"
            all_dets.append(d)

    if len(all_dets) <= 1:
        return _filter_peak_topk(all_dets, peak_floor, top_k)

    result = nms_detections(all_dets)
    result = _filter_peak_topk(result, peak_floor, top_k)
    logger.debug("ViT-B v4 heatmap (tiled): %d raw → %d after NMS + peak/top-K",
                 len(all_dets), len(result))
    return result


def run_vitb_v4_heatmap_detector_and_heatmap(
    array: np.ndarray,
    checkpoint: Path | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    ckpt_path = Path(checkpoint) if checkpoint else _default_checkpoint()
    threshold        = float(os.environ.get("VITB_V4_HEATMAP_THRESHOLD", "0.65"))
    min_pixels       = int(os.environ.get("VITB_V4_HEATMAP_MIN_PIXELS", "2"))
    native_tile_size = int(os.environ.get("VITB_V4_HEATMAP_NATIVE_TILE_SIZE", "400"))
    tile_overlap     = float(os.environ.get("VITB_V4_HEATMAP_TILE_OVERLAP", "0.5"))
    peak_floor       = float(os.environ.get("VITB_V4_HEATMAP_PEAK_FLOOR", "0.85"))
    top_k            = int(os.environ.get("VITB_V4_HEATMAP_TOPK", "0"))

    try:
        model, image_size, device, use_geometry = _load_model(ckpt_path)
    except Exception as exc:
        logger.warning("ViT-B v4 heatmap model load failed (heatmap): %s", exc)
        return [], None

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    h_full, w_full = array.shape[:2]
    heat_full = np.zeros((h_full, w_full), dtype=np.float32)

    if max(h_full, w_full) <= native_tile_size:
        dets = _run_single_tile(array, model, image_size, device, threshold, min_pixels,
                                use_geometry=use_geometry)
        for d in dets:
            d["method"] = "vitb_heatmap_v4"
        heat_tile, _, _, _ = _run_single_tile_probs(array, model, image_size, device)
        return _filter_peak_topk(dets, peak_floor, top_k), heat_tile

    from inference.tiled_pipeline import tile_image
    from inference.postprocess import nms_detections

    all_dets: list[dict[str, Any]] = []
    for tile, x0, y0 in tile_image(array, native_tile_size, tile_overlap):
        th, tw = tile.shape[:2]
        for det in _run_single_tile(tile, model, image_size, device, threshold, min_pixels,
                                    use_geometry=use_geometry):
            d = _remap_detection(det, x0, y0)
            d["method"] = "vitb_heatmap_v4"
            all_dets.append(d)
        heat_tile, _, _, _ = _run_single_tile_probs(tile, model, image_size, device)
        y1e, x1e = min(y0 + th, h_full), min(x0 + tw, w_full)
        np.maximum(heat_full[y0:y1e, x0:x1e], heat_tile[:y1e - y0, :x1e - x0],
                   out=heat_full[y0:y1e, x0:x1e])

    if len(all_dets) <= 1:
        return _filter_peak_topk(all_dets, peak_floor, top_k), heat_full

    return _filter_peak_topk(nms_detections(all_dets), peak_floor, top_k), heat_full
