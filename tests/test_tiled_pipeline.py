"""Tests for endpoint tiling helpers."""

from __future__ import annotations

import numpy as np

from inference.tiled_pipeline import (
    nms_predictions,
    remap_predictions,
    select_tile_params,
    stitch_collinear_fragments,
    tile_image,
)


def _segment(x1: float, x2: float, confidence: float = 0.9) -> dict:
    return {"x1": x1, "y1": 20.0, "x2": x2, "y2": 20.0, "confidence": confidence}


def test_tile_image_covers_edges() -> None:
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    tiles = list(tile_image(image, 4, 0.5))
    assert [(x, y) for _, x, y in tiles] == [(0, 0), (2, 0), (3, 0), (0, 1), (2, 1), (3, 1)]
    assert all(tile.shape == (4, 4, 3) for tile, _, _ in tiles)


def test_tile_resize_preserves_native_offsets() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    native = [(x, y) for _, x, y in tile_image(image, 4, 0.5)]
    resized = [(x, y) for _, x, y in tile_image(image, 4, 0.5, resize_to=8)]
    assert resized == native


def test_remap_predictions_scales_endpoints() -> None:
    original = [_segment(20.0, 100.0)]
    remapped = remap_predictions(original, x0=10, y0=5, magnification=2.0)
    assert remapped[0]["x1"] == 20.0
    assert remapped[0]["x2"] == 60.0
    assert remapped[0]["y1"] == 15.0
    assert "bbox" not in remapped[0]


def test_nms_predictions_suppresses_duplicate_segments() -> None:
    predictions = [_segment(0, 100, 0.9), _segment(1, 101, 0.8)]
    assert nms_predictions(predictions) == [predictions[0]]


def test_stitch_collinear_fragments_joins_gap() -> None:
    stitched = stitch_collinear_fragments([_segment(0, 100), _segment(120, 200, 0.8)])
    assert len(stitched) == 1
    assert stitched[0]["x1"] == 0.0
    assert stitched[0]["x2"] == 200.0


def test_select_tile_params_is_bounded() -> None:
    result = select_tile_params((1000, 1000), 5.0, min_native_tile=80)
    assert result["native_tile_size"] == 80
