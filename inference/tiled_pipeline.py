"""Tiling helpers for endpoint-native streak detectors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from inference.postprocess import nms_detections, stitch_collinear_segments
from inference.streak_segment import apply_segment_geometry


def tile_image(
    image: np.ndarray,
    tile_size: int,
    overlap: float = 0.5,
    resize_to: int | None = None,
    interp: str = "bilinear",
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield overlapping square tiles and native-pixel offsets.

    Edge tiles are shifted inward to remain square, so every source pixel is
    covered without padding.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if interp not in {"bilinear", "lanczos", "area"}:
        raise ValueError("interp must be one of: bilinear, lanczos, area")

    height, width = image.shape[:2]
    stride = max(1, round(tile_size * (1.0 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, length - tile_size + 1, stride))
        final = length - tile_size
        if values[-1] != final:
            values.append(final)
        return values

    for y0 in starts(height):
        for x0 in starts(width):
            tile = image[y0:min(y0 + tile_size, height), x0:min(x0 + tile_size, width)]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                pad_shape = (tile_size, tile_size, *tile.shape[2:])
                padded = np.zeros(pad_shape, dtype=image.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            if resize_to is not None and resize_to != tile_size:
                import cv2
                modes = {
                    "bilinear": cv2.INTER_LINEAR,
                    "lanczos": cv2.INTER_LANCZOS4,
                    "area": cv2.INTER_AREA,
                }
                tile = cv2.resize(tile, (resize_to, resize_to), interpolation=modes[interp])
            yield tile, x0, y0


def remap_predictions(
    predictions: list[dict[str, Any]],
    x0: int,
    y0: int,
    magnification: float = 1.0,
) -> list[dict[str, Any]]:
    """Map tile-local endpoint predictions into source-image coordinates."""
    output: list[dict[str, Any]] = []
    for prediction in predictions:
        item = dict(prediction)
        item["x1"] = float(item["x1"]) / magnification + x0
        item["y1"] = float(item["y1"]) / magnification + y0
        item["x2"] = float(item["x2"]) / magnification + x0
        item["y2"] = float(item["y2"]) / magnification + y0
        output.append(apply_segment_geometry(item))
    return output


def nms_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Suppress duplicate endpoint predictions."""
    for prediction in predictions:
        prediction.setdefault("confidence", prediction.get("score", 0.0))
    return nms_detections(predictions)


def stitch_collinear_fragments(
    predictions: list[dict[str, Any]],
    max_gap_px: float = 200.0,
    max_growth_ratio: float = 3.0,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Compatibility name for endpoint fragment stitching."""
    return stitch_collinear_segments(
        predictions,
        max_gap_px=max_gap_px,
        max_growth_ratio=max_growth_ratio,
        **kwargs,
    )


def select_tile_params(
    image_shape: tuple[int, int],
    expected_streak_native_px: float,
    target_streak_model_px: float = 150.0,
    model_input_size: int = 400,
    min_native_tile: int = 80,
    max_downscale: float = 3.0,
) -> dict[str, float | int]:
    """Choose native tile size from the expected endpoint separation."""
    del image_shape
    native = int(model_input_size * expected_streak_native_px / target_streak_model_px)
    native = max(min_native_tile, min(native, int(model_input_size * max_downscale)))
    return {"native_tile_size": native, "overlap": 0.5}


if __name__ == "__main__":
    sample = np.zeros((10, 10), dtype=np.uint8)
    assert list(tile_image(sample, 4, 0.5))
