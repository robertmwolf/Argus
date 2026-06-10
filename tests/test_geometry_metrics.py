"""Tests for eval/geometry_metrics.py — three-tier streak geometry evaluation."""

from __future__ import annotations

import math
import numpy as np
import pytest

from eval.geometry_metrics import (
    _angle_error_deg,
    _centerline_endpoints,
    _centerline_match,
    _endpoint_error_px,
    _geometry_stats,
    _point_to_segment,
    evaluate_geometry,
)


# ---------------------------------------------------------------------------
# Helper: make a minimal prediction/GT dict
# ---------------------------------------------------------------------------

def _make_det(cx: float, cy: float, w: float, h: float, angle_deg: float,
              image_id: str = "img1", confidence: float = 0.9) -> dict:
    return {
        "image_id": image_id,
        "confidence": confidence,
        "obb": {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle_deg},
        "streak_length_px": w,
    }


# ---------------------------------------------------------------------------
# _centerline_endpoints
# ---------------------------------------------------------------------------

def test_centerline_endpoints_horizontal():
    obb = {"cx": 100.0, "cy": 50.0, "w": 200.0, "h": 4.0, "angle_deg": 0.0}
    p1, p2 = _centerline_endpoints(obb)
    assert p1 == pytest.approx([0.0, 50.0], abs=1e-6)
    assert p2 == pytest.approx([200.0, 50.0], abs=1e-6)


def test_centerline_endpoints_vertical():
    obb = {"cx": 50.0, "cy": 100.0, "w": 200.0, "h": 4.0, "angle_deg": 90.0}
    p1, p2 = _centerline_endpoints(obb)
    # cos(90°) ≈ 0, sin(90°) = 1
    assert p1[0] == pytest.approx(50.0, abs=1e-4)
    assert p2[0] == pytest.approx(50.0, abs=1e-4)
    assert abs(p1[1] - p2[1]) == pytest.approx(200.0, abs=1e-4)


# ---------------------------------------------------------------------------
# _point_to_segment
# ---------------------------------------------------------------------------

def test_point_to_segment_midpoint():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([100.0, 0.0])
    dist, t = _point_to_segment(np.array([50.0, 5.0]), p1, p2)
    assert t == pytest.approx(0.5, abs=1e-6)
    assert dist == pytest.approx(5.0, abs=1e-6)


def test_point_to_segment_off_end():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([100.0, 0.0])
    dist, t = _point_to_segment(np.array([150.0, 0.0]), p1, p2)
    assert t == pytest.approx(1.5, abs=1e-6)


def test_point_to_segment_zero_length():
    p1 = p2 = np.array([50.0, 50.0])
    dist, t = _point_to_segment(np.array([60.0, 60.0]), p1, p2)
    assert dist == pytest.approx(math.sqrt(200), abs=1e-4)


# ---------------------------------------------------------------------------
# _angle_error_deg
# ---------------------------------------------------------------------------

def test_angle_error_symmetry():
    assert _angle_error_deg(0.0, 180.0) == pytest.approx(0.0)
    assert _angle_error_deg(5.0, 175.0) == pytest.approx(10.0)
    assert _angle_error_deg(90.0, 0.0) == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# _centerline_match
# ---------------------------------------------------------------------------

def test_centerline_match_hit_on_line():
    # Horizontal GT streak from (0, 0) to (200, 0), center at (100, 0)
    gt = _make_det(100.0, 0.0, 200.0, 4.0, 0.0)
    # Pred center at midpoint, perfect position
    pred = _make_det(100.0, 0.0, 50.0, 4.0, 0.0)
    assert _centerline_match(pred, gt, perp_threshold_px=10.0)


def test_centerline_match_perp_just_inside():
    gt = _make_det(100.0, 0.0, 200.0, 4.0, 0.0)
    pred = _make_det(100.0, 9.9, 50.0, 4.0, 0.0)  # 9.9 px off line
    assert _centerline_match(pred, gt, perp_threshold_px=10.0)


def test_centerline_match_perp_just_outside():
    gt = _make_det(100.0, 0.0, 200.0, 4.0, 0.0)
    pred = _make_det(100.0, 10.1, 50.0, 4.0, 0.0)  # 10.1 px off line
    assert not _centerline_match(pred, gt, perp_threshold_px=10.0)


def test_centerline_match_off_end_is_miss():
    # GT streak from (0,0) to (200,0). Pred centre at (250, 0) — past the endpoint.
    gt = _make_det(100.0, 0.0, 200.0, 4.0, 0.0)
    pred = _make_det(250.0, 0.0, 50.0, 4.0, 0.0)
    assert not _centerline_match(pred, gt, perp_threshold_px=10.0)


def test_centerline_match_off_start_is_miss():
    gt = _make_det(100.0, 0.0, 200.0, 4.0, 0.0)
    pred = _make_det(-50.0, 0.0, 50.0, 4.0, 0.0)
    assert not _centerline_match(pred, gt, perp_threshold_px=10.0)


def test_centerline_match_at_endpoint_is_hit():
    # Pred centre exactly at p2 endpoint — t=1.0, should still be a match
    gt = _make_det(100.0, 0.0, 200.0, 4.0, 0.0)
    pred = _make_det(200.0, 0.0, 50.0, 4.0, 0.0)
    assert _centerline_match(pred, gt, perp_threshold_px=10.0)


# ---------------------------------------------------------------------------
# _endpoint_error_px
# ---------------------------------------------------------------------------

def test_endpoint_error_perfect():
    obb = {"cx": 100.0, "cy": 0.0, "w": 200.0, "h": 4.0, "angle_deg": 0.0}
    assert _endpoint_error_px(obb, obb) == pytest.approx(0.0, abs=1e-6)


def test_endpoint_error_reversed_orientation():
    # Same streak but "reversed" — undirected, so error should be ~0
    gt  = {"cx": 100.0, "cy": 0.0, "w": 200.0, "h": 4.0, "angle_deg": 0.0}
    pred = {"cx": 100.0, "cy": 0.0, "w": 200.0, "h": 4.0, "angle_deg": 180.0}
    assert _endpoint_error_px(pred, gt) == pytest.approx(0.0, abs=1e-4)


def test_endpoint_error_shifted():
    gt   = {"cx": 100.0, "cy": 0.0, "w": 200.0, "h": 4.0, "angle_deg": 0.0}
    pred = {"cx": 100.0, "cy": 0.0, "w": 100.0, "h": 4.0, "angle_deg": 0.0}
    # GT endpoints: (0,0) and (200,0). Pred endpoints: (50,0) and (150,0).
    # Forward: mean of |0-50| and |200-150| = 50
    # Reverse: same by symmetry
    assert _endpoint_error_px(pred, gt) == pytest.approx(50.0, abs=1e-4)


# ---------------------------------------------------------------------------
# evaluate_geometry — end-to-end
# ---------------------------------------------------------------------------

def test_evaluate_geometry_perfect_detection():
    gt   = [_make_det(100.0, 0.0, 200.0, 4.0, 0.0)]
    pred = [_make_det(100.0, 0.0, 200.0, 4.0, 0.0)]
    result = evaluate_geometry(pred, gt, perp_threshold_px=10.0)

    t1 = result["tier1_detection"]
    assert t1["n_gt"] == 1
    assert t1["n_found"] == 1
    assert t1["detection_recall"] == 1.0
    assert t1["detection_precision"] == 1.0

    t2 = result["tier2_raw_geometry"]
    assert t2["n_pairs"] == 1
    assert t2["angle_err_deg"]["mean"] == pytest.approx(0.0, abs=1e-4)
    assert t2["endpoint_err_px"]["mean"] == pytest.approx(0.0, abs=1e-4)

    assert result["tier3_refined_geometry"] is None


def test_evaluate_geometry_off_end_miss():
    gt   = [_make_det(100.0, 0.0, 200.0, 4.0, 0.0)]
    # Pred centre at x=300, well past the right endpoint
    pred = [_make_det(300.0, 0.0, 50.0, 4.0, 0.0)]
    result = evaluate_geometry(pred, gt, perp_threshold_px=10.0)
    assert result["tier1_detection"]["n_found"] == 0
    assert result["tier1_detection"]["detection_recall"] == 0.0


def test_evaluate_geometry_false_positive():
    gt   = [_make_det(100.0, 0.0, 200.0, 4.0, 0.0)]
    # Two predictions: one matches, one is elsewhere
    pred = [
        _make_det(100.0, 0.0, 200.0, 4.0, 0.0, confidence=0.9),
        _make_det(500.0, 500.0, 50.0, 4.0, 0.0, confidence=0.5),
    ]
    result = evaluate_geometry(pred, gt, perp_threshold_px=10.0)
    t1 = result["tier1_detection"]
    assert t1["n_found"] == 1
    assert t1["n_false_positives"] == 1
    assert t1["detection_precision"] == pytest.approx(0.5, abs=1e-4)


def test_evaluate_geometry_no_predictions():
    gt   = [_make_det(100.0, 0.0, 200.0, 4.0, 0.0)]
    result = evaluate_geometry([], gt, perp_threshold_px=10.0)
    assert result["tier1_detection"]["detection_recall"] == 0.0
    assert result["tier2_raw_geometry"]["n_pairs"] == 0


def test_evaluate_geometry_no_ground_truth():
    pred = [_make_det(100.0, 0.0, 200.0, 4.0, 0.0)]
    result = evaluate_geometry(pred, [], perp_threshold_px=10.0)
    assert result["tier1_detection"]["n_gt"] == 0
    assert result["tier1_detection"]["n_false_positives"] == 1


def test_evaluate_geometry_per_band():
    gt = [
        _make_det(100.0, 0.0, 80.0,  4.0, 0.0, image_id="img1"),   # short
        _make_det(100.0, 0.0, 250.0, 4.0, 0.0, image_id="img2"),   # medium
        _make_det(100.0, 0.0, 500.0, 4.0, 0.0, image_id="img3"),   # long
    ]
    pred = [
        _make_det(100.0, 0.0, 80.0,  4.0, 0.0, image_id="img1"),
        _make_det(100.0, 0.0, 250.0, 4.0, 0.0, image_id="img2"),
        # long streak missed on purpose
    ]
    result = evaluate_geometry(pred, gt, perp_threshold_px=10.0)
    t1 = result["tier1_detection"]
    assert t1["per_band"]["short"]["recall"]  == 1.0
    assert t1["per_band"]["medium"]["recall"] == 1.0
    assert t1["per_band"]["long"]["recall"]   == 0.0


def test_geometry_stats_empty():
    stats = _geometry_stats([])
    assert stats["n_pairs"] == 0
    assert stats["angle_err_deg"]["mean"] == 0.0
    assert stats["endpoint_err_px"]["median"] == 0.0
