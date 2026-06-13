from __future__ import annotations

import math

import numpy as np
import pytest

from inference.tiled_pipeline import (
    _union_find_components,
    nms_predictions,
    remap_predictions,
    select_tile_params,
    stitch_collinear_fragments,
    tile_image,
)


# ---------------------------------------------------------------------------
# tile_image — original 1:1 behaviour
# ---------------------------------------------------------------------------

def test_tile_image_pads_and_covers_edges() -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    tiles = list(tile_image(image, tile_size=4, overlap=0.5))

    assert [(x0, y0) for _, x0, y0 in tiles] == [
        (0, 0), (2, 0), (4, 0),
        (0, 2), (2, 2), (4, 2),
    ]
    assert all(tile.shape == (4, 4, 3) for tile, _, _ in tiles)
    assert np.array_equal(tiles[-1][0][-1, -1], image[-1, -1])


def test_tile_image_no_resize_when_resize_to_none() -> None:
    """Without resize_to, tiles should be exactly tile_size × tile_size."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for tile, _, _ in tile_image(image, tile_size=4, overlap=0.0):
        assert tile.shape == (4, 4, 3)


def test_tile_image_no_resize_when_resize_to_equals_tile_size() -> None:
    """resize_to == tile_size should be a no-op (no cv2 import needed)."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for tile, _, _ in tile_image(image, tile_size=4, overlap=0.0, resize_to=4):
        assert tile.shape == (4, 4, 3)


def test_tile_image_resize_upsamples_correctly() -> None:
    """resize_to > tile_size should upscale each crop."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for tile, _, _ in tile_image(image, tile_size=4, overlap=0.0, resize_to=8):
        assert tile.shape == (8, 8, 3)


def test_tile_image_resize_downsamples_correctly() -> None:
    """resize_to < tile_size should downscale each crop."""
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    for tile, _, _ in tile_image(image, tile_size=10, overlap=0.0, resize_to=5):
        assert tile.shape == (5, 5, 3)


def test_tile_image_offsets_remain_in_native_pixels_after_resize() -> None:
    """x0 / y0 must always be native source pixels even when resize_to is set."""
    image = np.zeros((10, 10), dtype=np.uint8)
    offsets_resized = [(x0, y0) for _, x0, y0 in tile_image(image, tile_size=4, overlap=0.5, resize_to=8)]
    offsets_native  = [(x0, y0) for _, x0, y0 in tile_image(image, tile_size=4, overlap=0.5)]
    assert offsets_resized == offsets_native


# ---------------------------------------------------------------------------
# remap_predictions — with and without magnification
# ---------------------------------------------------------------------------

def test_remap_predictions_offsets_xywh_boxes() -> None:
    preds = [{"bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9, "category_id": 1}]

    remapped = remap_predictions(preds, x0=10, y0=20)

    assert remapped == [{"bbox": [11.0, 22.0, 3.0, 4.0], "score": 0.9, "category_id": 1}]
    # Input must not be mutated.
    assert preds[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_remap_predictions_magnification_1_is_identity() -> None:
    """magnification=1.0 should give identical results to the default."""
    preds = [{"bbox": [5.0, 10.0, 20.0, 15.0], "score": 0.5, "category_id": 1}]
    assert remap_predictions(preds, x0=3, y0=7, magnification=1.0) == \
           remap_predictions(preds, x0=3, y0=7)


def test_remap_predictions_magnification_scales_then_offsets() -> None:
    """With magnification=2.0 the bbox should be halved then shifted."""
    # Model returns bbox in 400×400 tile space.
    # Native tile = 200×200.  magnification = 400/200 = 2.0.
    # Expected: bbox/2 + (x0, y0)
    preds = [{"bbox": [100.0, 80.0, 40.0, 20.0], "score": 0.8, "category_id": 1}]
    remapped = remap_predictions(preds, x0=50, y0=30, magnification=2.0)
    expected_bbox = [100.0 / 2 + 50, 80.0 / 2 + 30, 40.0 / 2, 20.0 / 2]
    result_bbox = remapped[0]["bbox"]
    assert len(result_bbox) == 4
    for got, want in zip(result_bbox, expected_bbox):
        assert abs(got - want) < 1e-6


def test_remap_predictions_frigate_magnification() -> None:
    """Simulate Frigate: native_tile=110, model_input=400 → mag≈3.636."""
    mag = 400 / 110
    # A detection at (150, 100) in model coords with size (80, 50).
    preds = [{"bbox": [150.0, 100.0, 80.0, 50.0], "score": 0.7, "category_id": 1}]
    remapped = remap_predictions(preds, x0=0, y0=0, magnification=mag)
    result_bbox = remapped[0]["bbox"]
    assert abs(result_bbox[0] - 150.0 / mag) < 0.5
    assert abs(result_bbox[2] - 80.0  / mag) < 0.5


# ---------------------------------------------------------------------------
# nms_predictions (unchanged API)
# ---------------------------------------------------------------------------

def test_nms_predictions_suppresses_same_category_duplicates() -> None:
    preds = [
        {"bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9, "category_id": 1},
        {"bbox": [1.0, 1.0, 10.0, 10.0], "score": 0.8, "category_id": 1},
        {"bbox": [1.0, 1.0, 10.0, 10.0], "score": 0.7, "category_id": 2},
        {"bbox": [30.0, 30.0, 5.0, 5.0], "score": 0.6, "category_id": 1},
    ]

    kept = nms_predictions(preds, iou_threshold=0.4)

    assert kept == [preds[0], preds[2], preds[3]]


# ---------------------------------------------------------------------------
# select_tile_params
# ---------------------------------------------------------------------------

def test_select_tile_params_returns_required_keys() -> None:
    result = select_tile_params((1555, 2325), expected_streak_native_px=40.0)
    assert "native_tile_size" in result
    assert "overlap" in result


def test_select_tile_params_frigate_regime() -> None:
    """40 px native streaks → ~106–107 px tile (plan §2.2 example).

    The formula is ``int(400 * 40 / 150) = int(106.67) = 106`` due to int()
    truncation.  The plan quotes "≈107" but the implementation uses int(), so
    we accept 106 or 107 — both are within 1 % of the expected 3.6× target.
    """
    result = select_tile_params(
        image_shape=(1555, 2325),
        expected_streak_native_px=40.0,
        target_streak_model_px=150.0,
        model_input_size=400,
    )
    # int(400 * 40 / 150) = int(106.67) = 106
    assert 106 <= result["native_tile_size"] <= 107
    assert 0.25 <= result["overlap"] <= 0.5


def test_select_tile_params_respects_min_native_tile() -> None:
    """Very short streaks should be clipped to min_native_tile."""
    result = select_tile_params(
        image_shape=(1000, 1000),
        expected_streak_native_px=5.0,
        min_native_tile=80,
    )
    assert result["native_tile_size"] >= 80


def test_select_tile_params_respects_max_downscale() -> None:
    """Very long streaks should be clipped to max_downscale * model_input_size."""
    result = select_tile_params(
        image_shape=(4000, 6000),
        expected_streak_native_px=2000.0,
        model_input_size=400,
        max_downscale=3.0,
    )
    assert result["native_tile_size"] <= 400 * 3


def test_select_tile_params_normal_satstreaks_regime() -> None:
    """Standard SatStreaks (200–800 px native) → native_tile ~400–1200 px."""
    result = select_tile_params(
        image_shape=(2000, 3000),
        expected_streak_native_px=400.0,
        target_streak_model_px=150.0,
        model_input_size=400,
    )
    # 400 * 400 / 150 ≈ 1067 → capped at 400 * 3 = 1200
    assert result["native_tile_size"] <= 1200
    assert result["native_tile_size"] >= 80


# ---------------------------------------------------------------------------
# tile_image — interpolation parameter
# ---------------------------------------------------------------------------

def test_tile_image_rejects_unknown_interp() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="interp must be one of"):
        list(tile_image(image, tile_size=4, overlap=0.0, interp="spline"))


def test_tile_image_lanczos_produces_correct_shape() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for tile, _, _ in tile_image(image, tile_size=4, overlap=0.0,
                                  resize_to=8, interp="lanczos"):
        assert tile.shape == (8, 8, 3)


def test_tile_image_bilinear_and_lanczos_same_shape() -> None:
    """Both interpolation modes must produce tiles of the same shape."""
    image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
    bilinear = [(t.shape, x0, y0) for t, x0, y0 in
                tile_image(image, tile_size=5, overlap=0.0, resize_to=10, interp="bilinear")]
    lanczos   = [(t.shape, x0, y0) for t, x0, y0 in
                 tile_image(image, tile_size=5, overlap=0.0, resize_to=10, interp="lanczos")]
    assert bilinear == lanczos


# ---------------------------------------------------------------------------
# _union_find_components
# ---------------------------------------------------------------------------

def test_union_find_all_singletons() -> None:
    groups = _union_find_components(4, [])
    assert sorted(sorted(g) for g in groups) == [[0], [1], [2], [3]]


def test_union_find_single_component() -> None:
    groups = _union_find_components(3, [(0, 1), (1, 2)])
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_union_find_two_components() -> None:
    groups = _union_find_components(4, [(0, 1), (2, 3)])
    assert len(groups) == 2
    grouped = sorted(sorted(g) for g in groups)
    assert grouped == [[0, 1], [2, 3]]


# ---------------------------------------------------------------------------
# stitch_collinear_fragments
# ---------------------------------------------------------------------------

def _horiz_streak(x: float, y: float, w: float, h: float, score: float = 0.8) -> dict:
    """Helper: make a horizontal-ish streak prediction."""
    return {"bbox": [x, y, w, h], "score": score, "category_id": 1}


def test_stitch_passthrough_singleton() -> None:
    preds = [_horiz_streak(0, 0, 200, 10)]
    result = stitch_collinear_fragments(preds)
    assert len(result) == 1
    assert result[0]["bbox"] == [0, 0, 200, 10]


def test_stitch_passthrough_empty() -> None:
    assert stitch_collinear_fragments([]) == []


def test_stitch_merges_two_horizontal_fragments() -> None:
    """Two aligned horizontal boxes with a small gap should become one."""
    # Box A: x=0..200, y=95..105 (w=200, h=10 → aspect 20:1, angle≈3°)
    # Box B: x=250..450, y=95..105 (gap = 50 px)
    a = _horiz_streak(0,   95, 200, 10, score=0.9)
    b = _horiz_streak(250, 95, 200, 10, score=0.7)
    result = stitch_collinear_fragments([a, b], max_gap_px=100.0)
    assert len(result) == 1
    merged_bbox = result[0]["bbox"]
    # Merged box should span x=0..450
    assert merged_bbox[0] == 0.0
    assert abs(merged_bbox[0] + merged_bbox[2] - 450.0) < 1.0
    # Score = max
    assert result[0]["score"] == 0.9


def test_stitch_does_not_merge_gap_too_large() -> None:
    """Gap exceeding max_gap_px → no merge."""
    a = _horiz_streak(0,   95, 200, 10)
    b = _horiz_streak(500, 95, 200, 10)
    result = stitch_collinear_fragments([a, b], max_gap_px=100.0)
    assert len(result) == 2


def test_stitch_does_not_merge_perpendicular_boxes() -> None:
    """A horizontal and a vertical box at the same x should not be stitched."""
    horiz = _horiz_streak(0,   50, 200, 10)   # angle ≈ 3° (horizontal)
    vert  = _horiz_streak(0, -100, 10, 200)   # angle ≈ 87° (vertical)
    result = stitch_collinear_fragments([horiz, vert])
    assert len(result) == 2


def test_stitch_does_not_merge_laterally_offset_boxes() -> None:
    """Two parallel boxes that are far apart perpendicular to streak → no merge."""
    # Both horizontal, but offset by 500 px vertically
    a = _horiz_streak(0, 0,   200, 10)
    b = _horiz_streak(250, 500, 200, 10)
    result = stitch_collinear_fragments([a, b], max_gap_px=400.0)
    assert len(result) == 2


def test_stitch_merges_vertical_fragments() -> None:
    """Two aligned vertical boxes with a small gap should become one."""
    # Box A: x=95..105, y=0..300 (w=10, h=300 → aspect 30:1, angle≈88°)
    # Box B: x=95..105, y=350..650 (gap = 50 px vertically)
    a = {"bbox": [95, 0,   10, 300], "score": 0.85, "category_id": 1}
    b = {"bbox": [95, 350, 10, 300], "score": 0.75, "category_id": 1}
    result = stitch_collinear_fragments([a, b], max_gap_px=100.0)
    assert len(result) == 1
    merged_bbox = result[0]["bbox"]
    # Should span y=0..650
    assert merged_bbox[1] == 0.0
    assert abs(merged_bbox[1] + merged_bbox[3] - 650.0) < 1.0


def test_stitch_ignores_low_aspect_ratio_boxes() -> None:
    """Near-square boxes (aspect < min_aspect_ratio) should pass through unmerged."""
    # Both boxes are 100×80 (aspect = 1.25, below default threshold 1.5)
    a = {"bbox": [0,   0, 100, 80], "score": 0.8, "category_id": 1}
    b = {"bbox": [150, 0, 100, 80], "score": 0.8, "category_id": 1}
    result = stitch_collinear_fragments([a, b], max_gap_px=400.0, min_aspect_ratio=1.5)
    assert len(result) == 2


def test_stitch_three_collinear_fragments() -> None:
    """Three collinear fragments should merge into a single box."""
    a = _horiz_streak(0,   95, 200, 10, score=0.9)
    b = _horiz_streak(250, 95, 200, 10, score=0.7)
    c = _horiz_streak(500, 95, 200, 10, score=0.8)
    result = stitch_collinear_fragments([a, b, c], max_gap_px=100.0)
    assert len(result) == 1
    merged_bbox = result[0]["bbox"]
    assert merged_bbox[0] == 0.0
    assert abs(merged_bbox[0] + merged_bbox[2] - 700.0) < 1.0
    assert result[0]["score"] == 0.9  # max of all three


def test_stitch_does_not_bridge_across_large_gap_transitively() -> None:
    """Anti-transitivity: a true streak must not absorb a far collinear fragment.

    A and B are a real 2-tile streak (gap 50). C is a far collinear fragment
    600 px past B — beyond max_gap_px. Old union-find merged {A,B,C} into one
    frame-spanning blob; the guarded greedy merge must keep C separate.
    """
    a = _horiz_streak(0,    95, 200, 10, score=0.9)
    b = _horiz_streak(250,  95, 200, 10, score=0.9)   # gap 50 from A → merges
    c = _horiz_streak(1050, 95, 200, 10, score=0.9)   # gap 600 from B → must NOT chain
    result = stitch_collinear_fragments([a, b, c], max_gap_px=200.0)
    assert len(result) == 2
    spans = sorted(r["bbox"][2] for r in result)
    assert abs(spans[0] - 200.0) < 1.0   # lone C
    assert abs(spans[1] - 450.0) < 1.0   # merged A+B (x=0..450)


def test_stitch_excludes_low_confidence_fragments() -> None:
    """Fragments below conf_floor do not seed/extend chains (pass through)."""
    a = _horiz_streak(0,   95, 200, 10, score=0.9)
    b = _horiz_streak(250, 95, 200, 10, score=0.2)   # below floor 0.5
    result = stitch_collinear_fragments([a, b], max_gap_px=200.0, conf_floor=0.5)
    assert len(result) == 2          # not merged — b is ineligible
    # Lowering the floor lets them merge.
    merged = stitch_collinear_fragments([a, b], max_gap_px=200.0, conf_floor=0.0)
    assert len(merged) == 1


def test_select_tile_params_overlap_covers_streak() -> None:
    """Overlap should ensure the tile stride ≤ max estimated streak length."""
    streak_px = 40.0
    result = select_tile_params(
        image_shape=(1555, 2325),
        expected_streak_native_px=streak_px,
    )
    tile = result["native_tile_size"]
    stride = tile * (1.0 - result["overlap"])
    # Stride must be ≤ 3× streak length (conservative upper bound from plan).
    assert stride <= streak_px * 3 + 1  # +1 for float rounding
