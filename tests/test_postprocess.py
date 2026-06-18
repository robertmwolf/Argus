"""Tests for endpoint-native streak postprocessing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from inference.postprocess import (
    QUALITY_EDGE,
    QUALITY_GOOD,
    QUALITY_LOW_CONF,
    QUALITY_NO_WCS,
    QUALITY_TOO_SHORT,
    classify_detection_quality,
    extend_segment_to_streak_extent,
    fuse_group_geometries,
    group_detections,
    nms_detections,
    refine_segment_angle,
)
from inference.streak_segment import apply_segment_geometry


def _segment(
    confidence: float = 0.9,
    x1: float = 20.0,
    y1: float = 64.0,
    x2: float = 100.0,
    y2: float = 64.0,
) -> dict:
    det = {
        "confidence": confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "ra_tip1_deg": 10.0,
        "dec_tip1_deg": 20.0,
        "ra_tip2_deg": 10.1,
        "dec_tip2_deg": 20.1,
    }
    return apply_segment_geometry(det)


def _streak_image(angle_deg: float, size: int = 128) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float32)
    center = size / 2
    radians = math.radians(angle_deg)
    for t in np.linspace(-45, 45, 181):
        x = int(round(center + t * math.cos(radians)))
        y = int(round(center + t * math.sin(radians)))
        if 0 <= x < size and 0 <= y < size:
            image[max(0, y - 1):min(size, y + 2), max(0, x - 1):min(size, x + 2)] = 255
    return image


@pytest.mark.parametrize("angle", [20.0, 45.0, 90.0, 135.0, 160.0])
def test_refine_segment_angle_recovers_streak(angle: float) -> None:
    refined = refine_segment_angle(_streak_image(angle), angle + 4.0, 15.0)
    error = abs(refined - angle) % 180.0
    assert min(error, 180.0 - error) <= 5.0


def test_refine_segment_angle_zero_range_returns_initial() -> None:
    assert refine_segment_angle(np.zeros((16, 16)), 37.0, 0.0) == 37.0


def test_extend_segment_never_shrinks() -> None:
    image = _streak_image(0.0)
    original = _segment(x1=14.0, y1=64.0, x2=114.0, y2=64.0)
    extended = extend_segment_to_streak_extent(image, original)
    assert extended["streak_length_px"] >= original["streak_length_px"]
    assert set(("x1", "y1", "x2", "y2")).issubset(extended)


def test_nms_suppresses_duplicate_segments() -> None:
    high = _segment(0.9)
    low = _segment(0.5, 21.0, 64.5, 101.0, 64.5)
    assert nms_detections([low, high]) == [high]


def test_nms_keeps_separate_segments() -> None:
    first = _segment(0.9)
    second = _segment(0.8, 20.0, 100.0, 100.0, 100.0)
    assert len(nms_detections([first, second])) == 2


def test_group_detections_groups_partial_overlap() -> None:
    full = _segment(0.9, 0.0, 64.0, 120.0, 64.0)
    partial = _segment(0.7, 30.0, 65.0, 90.0, 65.0)
    grouped = group_detections([full, partial])
    assert len({det["streak_id"] for det in grouped}) == 1


def test_group_detections_keeps_crossing_streaks_separate() -> None:
    horizontal = _segment(0.9, 10.0, 64.0, 118.0, 64.0)
    vertical = _segment(0.8, 64.0, 10.0, 64.0, 118.0)
    grouped = group_detections([horizontal, vertical])
    assert len({det["streak_id"] for det in grouped}) == 2


def test_fuse_group_geometries_spans_outer_endpoints() -> None:
    first = _segment(0.9, 10.0, 50.0, 70.0, 50.0)
    second = _segment(0.8, 60.0, 51.0, 130.0, 51.0)
    first["streak_id"] = second["streak_id"] = 1
    fused = fuse_group_geometries([first, second])
    assert fused[0]["streak_length_px"] == pytest.approx(120.0, rel=0.02)
    assert fused[0]["x1"] == pytest.approx(10.0, abs=1.0)
    assert fused[0]["x2"] == pytest.approx(130.0, abs=1.0)
    assert "obb" not in fused[0]


def test_quality_good() -> None:
    assert classify_detection_quality(_segment(x1=100, y1=100, x2=300, y2=100), (512, 512)) == QUALITY_GOOD


def test_quality_edge_has_priority() -> None:
    det = _segment(0.01, 5, 100, 205, 100)
    assert classify_detection_quality(det, (512, 512)) == QUALITY_EDGE


def test_quality_low_confidence() -> None:
    det = _segment(0.1, 100, 100, 300, 100)
    assert classify_detection_quality(det, (512, 512)) == QUALITY_LOW_CONF


def test_quality_too_short() -> None:
    det = _segment(0.9, 100, 100, 120, 100)
    assert classify_detection_quality(det, (512, 512)) == QUALITY_TOO_SHORT


def test_quality_no_wcs() -> None:
    det = _segment(0.9, 100, 100, 300, 100)
    det["ra_tip1_deg"] = det["dec_tip1_deg"] = None
    det["ra_tip2_deg"] = det["dec_tip2_deg"] = None
    assert classify_detection_quality(det, (512, 512)) == QUALITY_NO_WCS
