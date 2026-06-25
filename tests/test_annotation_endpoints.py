"""Tests for training/annotation_endpoints.py.

Covers all conversion paths in annotation_to_endpoints:
  native x1/y1/x2/y2, line_segment dict, line_segment list, endpoints dict,
  obb dict, obb list/tuple, and bbox fallback (horizontal and vertical).
"""

from __future__ import annotations

import math

import pytest

from training.annotation_endpoints import annotation_to_endpoints


# ---------------------------------------------------------------------------
# Native endpoint passthrough
# ---------------------------------------------------------------------------


def test_native_endpoints_passthrough() -> None:
    ann = {"x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 80.0}
    assert annotation_to_endpoints(ann) == pytest.approx((10.0, 20.0, 110.0, 80.0))


def test_native_endpoints_coerces_to_float() -> None:
    ann = {"x1": 5, "y1": 0, "x2": 55, "y2": 0}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert isinstance(x1, float)
    assert x1 == pytest.approx(5.0)


def test_native_endpoints_partial_missing_falls_through() -> None:
    # Only x1/y1 present — should not match native path, fall to bbox.
    ann = {"x1": 0.0, "y1": 0.0, "bbox": [0, 0, 100, 10]}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert x2 == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# line_segment dict
# ---------------------------------------------------------------------------


def test_line_segment_dict() -> None:
    ann = {"line_segment": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}}
    assert annotation_to_endpoints(ann) == pytest.approx((1.0, 2.0, 3.0, 4.0))


def test_endpoints_dict_alias() -> None:
    ann = {"endpoints": {"x1": 5.0, "y1": 6.0, "x2": 7.0, "y2": 8.0}}
    assert annotation_to_endpoints(ann) == pytest.approx((5.0, 6.0, 7.0, 8.0))


# ---------------------------------------------------------------------------
# line_segment list / tuple
# ---------------------------------------------------------------------------


def test_line_segment_list() -> None:
    ann = {"line_segment": [10.0, 20.0, 30.0, 40.0]}
    assert annotation_to_endpoints(ann) == pytest.approx((10.0, 20.0, 30.0, 40.0))


def test_line_segment_tuple() -> None:
    ann = {"line_segment": (0.5, 1.5, 2.5, 3.5)}
    assert annotation_to_endpoints(ann) == pytest.approx((0.5, 1.5, 2.5, 3.5))


# ---------------------------------------------------------------------------
# obb dict
# ---------------------------------------------------------------------------


def test_obb_dict_horizontal() -> None:
    ann = {"obb": {"cx": 50.0, "cy": 50.0, "w": 100.0, "angle_deg": 0.0}}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert x1 == pytest.approx(0.0)
    assert y1 == pytest.approx(50.0)
    assert x2 == pytest.approx(100.0)
    assert y2 == pytest.approx(50.0)


def test_obb_dict_vertical() -> None:
    ann = {"obb": {"cx": 50.0, "cy": 100.0, "w": 80.0, "angle_deg": 90.0}}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    # cos(90°)≈0, sin(90°)=1
    assert x1 == pytest.approx(50.0, abs=1e-4)
    assert y1 == pytest.approx(60.0, abs=1e-4)
    assert x2 == pytest.approx(50.0, abs=1e-4)
    assert y2 == pytest.approx(140.0, abs=1e-4)


def test_obb_dict_diagonal_length_preserved() -> None:
    cx, cy, length = 100.0, 200.0, 60.0
    ann = {"obb": {"cx": cx, "cy": cy, "w": length, "angle_deg": 45.0}}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    reconstructed_length = math.hypot(x2 - x1, y2 - y1)
    assert reconstructed_length == pytest.approx(length, rel=1e-5)


def test_obb_dict_centre_preserved() -> None:
    ann = {"obb": {"cx": 80.0, "cy": 120.0, "w": 40.0, "angle_deg": 30.0}}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert (x1 + x2) / 2 == pytest.approx(80.0, abs=1e-5)
    assert (y1 + y2) / 2 == pytest.approx(120.0, abs=1e-5)


def test_obb_dict_defaults_angle_zero() -> None:
    ann = {"obb": {"cx": 50.0, "cy": 50.0, "w": 20.0}}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert y1 == pytest.approx(50.0)
    assert y2 == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# obb list / tuple
# ---------------------------------------------------------------------------


def test_obb_list_horizontal() -> None:
    # (cx, cy, length, _, angle_deg)
    ann = {"obb": [50.0, 50.0, 100.0, 10.0, 0.0]}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert x1 == pytest.approx(0.0)
    assert x2 == pytest.approx(100.0)
    assert y1 == pytest.approx(50.0)
    assert y2 == pytest.approx(50.0)


def test_obb_list_length_preserved() -> None:
    cx, cy, length = 200.0, 300.0, 80.0
    ann = {"obb": [cx, cy, length, 5.0, 60.0]}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(length, rel=1e-5)


# ---------------------------------------------------------------------------
# bbox fallback — horizontal (w >= h)
# ---------------------------------------------------------------------------


def test_bbox_horizontal_segment() -> None:
    ann = {"bbox": [10.0, 20.0, 80.0, 5.0]}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert x1 == pytest.approx(10.0)
    assert x2 == pytest.approx(90.0)
    assert y1 == pytest.approx(22.5)
    assert y2 == pytest.approx(22.5)


def test_bbox_vertical_segment() -> None:
    ann = {"bbox": [10.0, 20.0, 5.0, 80.0]}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert x1 == pytest.approx(12.5)
    assert x2 == pytest.approx(12.5)
    assert y1 == pytest.approx(20.0)
    assert y2 == pytest.approx(100.0)


def test_bbox_square_treated_as_horizontal() -> None:
    ann = {"bbox": [0.0, 0.0, 50.0, 50.0]}
    x1, y1, x2, y2 = annotation_to_endpoints(ann)
    assert x1 == pytest.approx(0.0)
    assert x2 == pytest.approx(50.0)
    assert y1 == y2


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_all_geometry_raises() -> None:
    with pytest.raises(KeyError):
        annotation_to_endpoints({"category_id": 1})
