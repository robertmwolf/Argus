"""Tests for pure-numpy helpers in inference/heatmap_detector_base.py.

Covers: _letterbox, _component_to_segment, _refine_half_length_by_profile,
_remap_detection, _rescale_detections, and _centerline_head_state.
Model loading (_load_model, _run_single_tile*) is excluded — requires weights.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from inference.heatmap_detector_base import (
    _centerline_head_state,
    _component_to_segment,
    _letterbox,
    _refine_half_length_by_profile,
    _remap_detection,
    _rescale_detections,
)
from inference.streak_segment import apply_segment_geometry


# ---------------------------------------------------------------------------
# _letterbox
# ---------------------------------------------------------------------------


class TestLetterbox:
    def test_square_input_no_padding(self) -> None:
        array = np.zeros((64, 64, 3), dtype=np.uint8)
        canvas, scale, pad_x, pad_y = _letterbox(array, 64)
        assert canvas.shape == (64, 64, 3)
        assert scale == pytest.approx(1.0)
        assert pad_x == pytest.approx(0.0)
        assert pad_y == pytest.approx(0.0)

    def test_wide_input_pads_vertically(self) -> None:
        array = np.zeros((32, 64, 3), dtype=np.uint8)
        canvas, scale, pad_x, pad_y = _letterbox(array, 64)
        assert canvas.shape == (64, 64, 3)
        assert scale == pytest.approx(1.0)
        assert pad_x == pytest.approx(0.0)
        assert pad_y > 0.0

    def test_tall_input_pads_horizontally(self) -> None:
        array = np.zeros((64, 32, 3), dtype=np.uint8)
        canvas, scale, pad_x, pad_y = _letterbox(array, 64)
        assert canvas.shape == (64, 64, 3)
        assert pad_x > 0.0
        assert pad_y == pytest.approx(0.0)

    def test_scale_determined_by_larger_dimension(self) -> None:
        array = np.zeros((40, 80, 3), dtype=np.uint8)
        _, scale, _, _ = _letterbox(array, 80)
        assert scale == pytest.approx(1.0)

    def test_upscale_small_image(self) -> None:
        array = np.zeros((16, 16, 3), dtype=np.uint8)
        canvas, scale, pad_x, pad_y = _letterbox(array, 64)
        assert canvas.shape == (64, 64, 3)
        assert scale == pytest.approx(4.0)
        assert pad_x == pytest.approx(0.0)
        assert pad_y == pytest.approx(0.0)

    def test_grayscale_input_3d_required(self) -> None:
        # _letterbox expects HxWxC; callers must stack channels first
        array = np.zeros((32, 64, 3), dtype=np.uint8)
        canvas, _, _, _ = _letterbox(array, 64)
        assert canvas.ndim == 3

    def test_content_placed_inside_canvas(self) -> None:
        array = np.full((32, 64, 3), 128, dtype=np.uint8)
        canvas, scale, pad_x, pad_y = _letterbox(array, 64)
        y0 = round(pad_y)
        new_h = round(32 * scale)
        assert canvas[y0 : y0 + new_h].mean() > 0


# ---------------------------------------------------------------------------
# _component_to_segment
# ---------------------------------------------------------------------------


def _horizontal_mask(rows: int = 4, cols: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal streak mask in the top half of a (rows*2) x (cols) grid."""
    score_map = np.zeros((rows * 2, cols), dtype=np.float32)
    mask = np.zeros_like(score_map, dtype=bool)
    mask[rows // 2, :] = True
    score_map[mask] = 0.9
    return mask, score_map


class TestComponentToSegment:
    def test_single_pixel_returns_none(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[4, 4] = True
        score_map = np.zeros((8, 8), dtype=np.float32)
        assert _component_to_segment(mask, score_map, 16, 128) is None

    def test_horizontal_streak_angle_near_zero(self) -> None:
        mask, score_map = _horizontal_mask(rows=4, cols=20)
        det = _component_to_segment(mask, score_map, 16, 320)
        assert det is not None
        assert det["angle_deg"] == pytest.approx(0.0, abs=5.0)

    def test_returns_required_keys(self) -> None:
        mask, score_map = _horizontal_mask()
        det = _component_to_segment(mask, score_map, 16, 320)
        assert det is not None
        for key in ("x1", "y1", "x2", "y2", "confidence", "peak_confidence",
                    "streak_length_px", "cx", "cy", "angle_deg"):
            assert key in det

    def test_confidence_is_score_mean(self) -> None:
        mask, score_map = _horizontal_mask()
        det = _component_to_segment(mask, score_map, 16, 320)
        assert det is not None
        assert det["confidence"] == pytest.approx(score_map[mask].mean(), rel=1e-4)

    def test_peak_confidence_is_score_max(self) -> None:
        mask, score_map = _horizontal_mask()
        det = _component_to_segment(mask, score_map, 16, 320)
        assert det is not None
        assert det["peak_confidence"] == pytest.approx(score_map[mask].max(), rel=1e-4)

    def test_centre_is_midpoint_of_endpoints(self) -> None:
        mask, score_map = _horizontal_mask()
        det = _component_to_segment(mask, score_map, 16, 320)
        assert det is not None
        assert det["cx"] == pytest.approx((det["x1"] + det["x2"]) / 2, abs=1.0)
        assert det["cy"] == pytest.approx((det["y1"] + det["y2"]) / 2, abs=1.0)

    def test_diagonal_streak_angle_near_45(self) -> None:
        mask = np.eye(12, dtype=bool)
        score_map = np.zeros_like(mask, dtype=np.float32)
        score_map[mask] = 0.8
        det = _component_to_segment(mask, score_map, 16, 192)
        assert det is not None
        assert det["angle_deg"] == pytest.approx(45.0, abs=5.0)

    def test_profile_refinement_does_not_change_direction(self) -> None:
        mask, score_map = _horizontal_mask(rows=4, cols=16)
        det_plain = _component_to_segment(mask, score_map, 16, 256)
        det_refined = _component_to_segment(mask, score_map, 16, 256, profile_peak_fraction=0.85)
        assert det_plain is not None
        assert det_refined is not None
        assert det_refined["angle_deg"] == pytest.approx(det_plain["angle_deg"], abs=5.0)


# ---------------------------------------------------------------------------
# _refine_half_length_by_profile
# ---------------------------------------------------------------------------


class TestRefineHalfLengthByProfile:
    def _uniform_heatmap_horizontal(
        self, n_cols: int = 10, patch_size: int = 16
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        score_map = np.zeros((4, n_cols), dtype=np.float32)
        row = 2
        score_map[row, :] = 1.0
        center = np.array([(n_cols / 2) * patch_size, (row + 0.5) * patch_size])
        major = np.array([1.0, 0.0])
        return score_map, center, major

    def test_returns_positive_half_length_for_active_streak(self) -> None:
        score_map, center, major = self._uniform_heatmap_horizontal()
        half, _ = _refine_half_length_by_profile(score_map, center, major, 16, 0.5)
        assert half > 0.0

    def test_zero_peak_returns_zero(self) -> None:
        score_map = np.zeros((4, 8), dtype=np.float32)
        center = np.array([64.0, 32.0])
        major = np.array([1.0, 0.0])
        half, shift = _refine_half_length_by_profile(score_map, center, major, 16, 0.5)
        assert half == 0.0
        assert shift == 0.0

    def test_single_active_pixel_returns_zero(self) -> None:
        score_map = np.zeros((4, 8), dtype=np.float32)
        score_map[2, 4] = 1.0
        center = np.array([4.5 * 16, 2.5 * 16])
        major = np.array([1.0, 0.0])
        half, shift = _refine_half_length_by_profile(score_map, center, major, 16, 0.5)
        assert half == 0.0

    def test_symmetric_streak_zero_centre_shift(self) -> None:
        score_map, center, major = self._uniform_heatmap_horizontal(n_cols=10)
        _, shift = _refine_half_length_by_profile(score_map, center, major, 16, 0.5)
        assert abs(shift) < 16.0  # shift within one patch of centre


# ---------------------------------------------------------------------------
# _remap_detection
# ---------------------------------------------------------------------------


class TestRemapDetection:
    def _make_det(self, x1=10.0, y1=5.0, x2=90.0, y2=5.0) -> dict:
        d = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.8,
             "peak_confidence": 0.9, "streak_length_px": 80.0,
             "cx": 50.0, "cy": 5.0, "angle_deg": 0.0, "method": "heatmap"}
        return apply_segment_geometry(d)

    def test_shifts_x_coordinates(self) -> None:
        det = _remap_detection(self._make_det(), x0=100, y0=0)
        assert det["x1"] == pytest.approx(110.0)
        assert det["x2"] == pytest.approx(190.0)

    def test_shifts_y_coordinates(self) -> None:
        det = _remap_detection(self._make_det(), x0=0, y0=200)
        assert det["y1"] == pytest.approx(205.0)
        assert det["y2"] == pytest.approx(205.0)

    def test_does_not_mutate_original(self) -> None:
        original = self._make_det()
        original_x1 = original["x1"]
        _remap_detection(original, x0=50, y0=50)
        assert original["x1"] == original_x1

    def test_zero_offset_is_identity(self) -> None:
        det = self._make_det()
        result = _remap_detection(det, x0=0, y0=0)
        assert result["x1"] == pytest.approx(det["x1"])
        assert result["y1"] == pytest.approx(det["y1"])


# ---------------------------------------------------------------------------
# _rescale_detections
# ---------------------------------------------------------------------------


class TestRescaleDetections:
    def _make_det(self, x1=10.0, y1=20.0, x2=110.0, y2=20.0) -> dict:
        d = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.7,
             "peak_confidence": 0.85, "streak_length_px": 100.0,
             "cx": 60.0, "cy": 20.0, "angle_deg": 0.0, "method": "heatmap"}
        return apply_segment_geometry(d)

    def test_scale_up(self) -> None:
        dets = _rescale_detections([self._make_det()], scale=2.0)
        assert dets[0]["x1"] == pytest.approx(20.0)
        assert dets[0]["x2"] == pytest.approx(220.0)

    def test_scale_down(self) -> None:
        dets = _rescale_detections([self._make_det()], scale=0.5)
        assert dets[0]["x1"] == pytest.approx(5.0)
        assert dets[0]["x2"] == pytest.approx(55.0)

    def test_does_not_mutate_originals(self) -> None:
        original = self._make_det()
        original_x1 = original["x1"]
        _rescale_detections([original], scale=3.0)
        assert original["x1"] == original_x1

    def test_empty_list(self) -> None:
        assert _rescale_detections([], scale=2.0) == []

    def test_scale_one_is_identity(self) -> None:
        det = self._make_det()
        result = _rescale_detections([det], scale=1.0)
        assert result[0]["x1"] == pytest.approx(det["x1"])
        assert result[0]["y2"] == pytest.approx(det["y2"])


# ---------------------------------------------------------------------------
# _centerline_head_state
# ---------------------------------------------------------------------------


def _fake_head_state(prefix: str = "net.") -> dict[str, torch.Tensor]:
    return {
        f"{prefix}4.weight": torch.zeros(2, 3, 1, 1),
        f"{prefix}4.bias": torch.zeros(2),
        f"{prefix}0.weight": torch.ones(3, 1, 1, 1),
    }


class TestCenterlineHeadState:
    def test_slices_to_one_channel(self) -> None:
        state = _centerline_head_state(_fake_head_state("net."))
        assert state["net.4.weight"].shape[0] == 1

    def test_adds_net_prefix_when_missing(self) -> None:
        state = _centerline_head_state(_fake_head_state(""))
        assert "net.4.weight" in state
        assert state["net.4.weight"].shape[0] == 1

    def test_preserves_existing_net_prefix(self) -> None:
        state = _centerline_head_state(_fake_head_state("net."))
        assert "net.4.weight" in state

    def test_slices_bias_to_one_element(self) -> None:
        state = _centerline_head_state(_fake_head_state("net."))
        assert state["net.4.bias"].shape[0] == 1

    def test_missing_final_conv_raises(self) -> None:
        bad_state = {"net.0.weight": torch.ones(3, 1, 1, 1)}
        with pytest.raises(ValueError, match="missing its final convolution"):
            _centerline_head_state(bad_state)
