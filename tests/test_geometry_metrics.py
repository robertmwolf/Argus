"""Tests for canonical endpoint geometry evaluation."""

from __future__ import annotations

import pytest

from eval.geometry_metrics import _endpoint_error_px, evaluate_geometry
from inference.streak_segment import StreakSegment


def _segment(image_id: str = "1", y: float = 10.0, confidence: float = 0.9) -> dict:
    return {
        "image_id": image_id,
        "confidence": confidence,
        "x1": 0.0,
        "y1": y,
        "x2": 200.0,
        "y2": y,
        "streak_length_px": 200.0,
    }


def test_endpoint_error_is_order_invariant() -> None:
    first = StreakSegment(0, 0, 100, 0, 1.0, "1")
    reversed_segment = StreakSegment(100, 0, 0, 0, 1.0, "1")
    assert _endpoint_error_px(first, reversed_segment) == pytest.approx(0.0)


def test_perfect_geometry_evaluation() -> None:
    result = evaluate_geometry([_segment()], [_segment()])
    assert result["tier1_detection"]["detection_recall"] == 1.0
    assert result["tier2_raw_geometry"]["angle_err_deg"]["mean"] == 0.0
    assert result["tier2_raw_geometry"]["endpoint_err_px"]["mean"] == 0.0


def test_perpendicular_miss() -> None:
    result = evaluate_geometry([_segment(y=30.0)], [_segment(y=10.0)])
    assert result["tier1_detection"]["detection_recall"] == 0.0


def test_false_positive_counted() -> None:
    result = evaluate_geometry([_segment(), _segment(image_id="2")], [_segment()])
    assert result["tier1_detection"]["n_false_positives"] == 1
