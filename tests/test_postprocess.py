"""Tests for inference/postprocess.py — Radon angle refinement and OBB NMS."""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_streak_image(
    height: int = 256,
    width: int = 256,
    angle_deg: float = 45.0,
    brightness: float = 2000.0,
    streak_width: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Return a float32 greyscale image with one synthetic streak at *angle_deg*.

    Uses a larger canvas and Gaussian cross-section for a clean, detectable
    streak signal at all angles.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 8.0, size=(height, width)).astype(np.float32)
    theta_rad = math.radians(angle_deg)
    cx, cy = width // 2, height // 2
    half_len = min(width, height) // 2 - 10
    for t in np.linspace(-half_len, half_len, half_len * 6):
        for dperp in np.linspace(-streak_width * 2, streak_width * 2, 9):
            px = int(round(cx + t * math.cos(theta_rad) - dperp * math.sin(theta_rad)))
            py = int(round(cy + t * math.sin(theta_rad) + dperp * math.cos(theta_rad)))
            if 0 <= px < width and 0 <= py < height:
                weight = math.exp(-0.5 * (dperp / streak_width) ** 2)
                img[py, px] += brightness * weight
    return np.clip(img, 0, 65535).astype(np.float32)


def _make_obb(cx=64.0, cy=64.0, w=100.0, h=4.0, angle_deg=45.0) -> dict:
    return {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle_deg}


def _make_det(confidence: float, cx=64.0, cy=64.0, w=80.0, h=5.0, angle_deg=45.0) -> dict:
    return {
        "confidence": confidence,
        "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        "obb": _make_obb(cx=cx, cy=cy, w=w, h=h, angle_deg=angle_deg),
    }


# ---------------------------------------------------------------------------
# bbox_to_obb
# ---------------------------------------------------------------------------

class TestBboxToObb:
    def test_center_is_bbox_midpoint(self):
        from inference.postprocess import bbox_to_obb
        obb = bbox_to_obb([10, 20, 50, 80], angle_deg=30.0)
        assert obb["cx"] == pytest.approx(30.0)
        assert obb["cy"] == pytest.approx(50.0)

    def test_returns_required_keys(self):
        from inference.postprocess import bbox_to_obb
        obb = bbox_to_obb([0, 0, 100, 20], angle_deg=0.0)
        assert set(obb.keys()) == {"cx", "cy", "w", "h", "angle_deg"}

    def test_w_is_long_axis(self):
        from inference.postprocess import bbox_to_obb
        obb = bbox_to_obb([0, 0, 100, 10], angle_deg=0.0)
        assert obb["w"] >= obb["h"]

    def test_square_bbox_zero_angle(self):
        """For a 100×100 bbox at 0°, w and h should both be 100."""
        from inference.postprocess import bbox_to_obb
        obb = bbox_to_obb([0, 0, 100, 100], angle_deg=0.0)
        assert obb["w"] == pytest.approx(100.0)
        assert obb["h"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# refine_angle
# ---------------------------------------------------------------------------

class TestRefineAngle:
    @pytest.mark.parametrize("true_angle", [20.0, 45.0, 90.0, 135.0, 160.0])
    def test_returns_within_5deg_of_ground_truth(self, true_angle):
        from inference.postprocess import refine_angle
        img = _make_streak_image(angle_deg=true_angle)
        obb = _make_obb(angle_deg=true_angle + 8.0)  # deliberately offset by 8°
        refined = refine_angle(img, obb, angle_search_range=15.0)
        # Allow for 180° ambiguity in streak direction
        err = min(
            abs(refined - true_angle),
            abs(refined - (true_angle + 180) % 180),
            abs(refined - (true_angle - 180) % 180),
        )
        assert err <= 5.0, f"angle={true_angle}°  refined={refined:.1f}°  err={err:.1f}°"

    def test_zero_search_range_returns_initial(self):
        from inference.postprocess import refine_angle
        img = _make_streak_image(angle_deg=45.0)
        obb = _make_obb(angle_deg=30.0)
        result = refine_angle(img, obb, angle_search_range=0.0)
        assert result == pytest.approx(30.0)

    def test_returns_float(self):
        from inference.postprocess import refine_angle
        img = _make_streak_image(angle_deg=45.0)
        obb = _make_obb(angle_deg=45.0)
        result = refine_angle(img, obb, angle_search_range=10.0)
        assert isinstance(result, float)

    def test_angle_in_0_to_180(self):
        from inference.postprocess import refine_angle
        img = _make_streak_image(angle_deg=170.0)
        obb = _make_obb(angle_deg=170.0)
        result = refine_angle(img, obb, angle_search_range=15.0)
        assert 0.0 <= result < 180.0

    def test_tiny_crop_returns_initial_no_crash(self):
        """Very small crop should fall back to initial angle, not crash."""
        from inference.postprocess import refine_angle
        tiny = np.zeros((3, 3), dtype=np.uint8)
        obb = _make_obb(angle_deg=45.0)
        result = refine_angle(tiny, obb, angle_search_range=10.0)
        assert isinstance(result, float)

    def test_3channel_crop_accepted(self):
        """3-channel uint8 crop should work (channels averaged internally)."""
        from inference.postprocess import refine_angle
        img_gray = _make_streak_image(angle_deg=45.0)
        img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
        obb = _make_obb(angle_deg=45.0)
        result = refine_angle(img_rgb, obb, angle_search_range=10.0)
        assert isinstance(result, float)

    def test_wide_search_recovers_mirrored_bbox_angle(self):
        """A bbox seed at +40° should recover a -40°/140° streak when allowed."""
        from inference.postprocess import refine_angle
        img = _make_streak_image(angle_deg=140.0)
        obb = _make_obb(angle_deg=40.0)
        refined = refine_angle(img, obb, angle_search_range=90.0)
        err = min(abs(refined - 140.0), abs(refined - (140.0 - 180.0)))
        assert err <= 5.0

    def test_wide_search_recovers_loose_tall_bbox_seed(self):
        """Loose DINO boxes can seed near 67° for a descending ~112° streak."""
        from inference.postprocess import refine_angle

        img = _make_streak_image(height=410, width=170, angle_deg=112.0)
        obb = _make_obb(cx=85.0, cy=205.0, w=444.0, h=308.0, angle_deg=67.5)

        refined = refine_angle(img, obb, angle_search_range=60.0)

        err = min(abs(refined - 112.0), abs(refined - (112.0 - 180.0)))
        assert err <= 5.0, f"refined={refined:.1f}°  err={err:.1f}°"


# ---------------------------------------------------------------------------
# extend_obb_to_streak_extent
# ---------------------------------------------------------------------------

class TestExtendObbToStreakExtent:
    def test_never_shrinks_existing_obb(self):
        from inference.postprocess import extend_obb_to_streak_extent

        image = np.full((128, 128), 100, dtype=np.uint8)
        image[64, 63:66] = 255
        obb = _make_obb(cx=64.0, cy=64.0, w=80.0, h=4.0, angle_deg=0.0)

        extended = extend_obb_to_streak_extent(image, obb)

        assert extended["w"] == pytest.approx(obb["w"])

    def test_does_not_jump_to_distant_bright_feature(self):
        """OBB centre must not teleport to a far-away star when t=0 is dim.

        Regression for the bug where extend_obb picked the globally longest
        bright run (a star 300 px away) instead of staying near the streak,
        causing the drawn box to appear hundreds of pixels from the actual
        streak in the preview.
        """
        from inference.postprocess import extend_obb_to_streak_extent

        h, w = 512, 512
        bg = np.full((h, w), 80, dtype=np.uint8)

        # Horizontal streak at y=256, x=220..300 (centre at x=260, length=80)
        streak_cx, streak_cy = 260.0, 256.0
        bg[256, 220:301] = 220

        # Distant bright star at (x=20, y=256) — along the same horizontal axis
        # but 240 px to the left of the streak centre.
        bg[256, 18:23] = 240

        # OBB centre is at the streak (t=0 is bright)
        obb = _make_obb(cx=streak_cx, cy=streak_cy, w=80.0, h=4.0, angle_deg=0.0)
        result = extend_obb_to_streak_extent(bg, obb)

        # Centre must stay near the streak — not jump to the star at x≈20
        assert abs(result["cx"] - streak_cx) < 80.0, (
            f"cx jumped from {streak_cx} to {result['cx']}: "
            "extend_obb latched onto a distant star"
        )

    def test_does_not_jump_when_obb_centre_is_dim(self):
        """When the OBB centre itself is dim but a far star is bright,
        the function should return the OBB unchanged rather than jumping.

        Regression: previously the 'longest run' fallback would pick the
        star, moving cx/cy hundreds of pixels from the actual streak location.
        """
        from inference.postprocess import extend_obb_to_streak_extent

        h, w = 512, 512
        bg = np.full((h, w), 80, dtype=np.uint8)

        # A short streak at y=256, x=220..240 (length=20)
        bg[256, 220:241] = 220

        # Distant bright star at x=450, y=256 — same horizontal row, 210 px away
        bg[256, 448:453] = 240

        # OBB centred between the streak and the star at x=330 — t=0 is dim
        obb = _make_obb(cx=330.0, cy=256.0, w=50.0, h=4.0, angle_deg=0.0)
        result = extend_obb_to_streak_extent(bg, obb)

        # The star is ~120 px from the OBB centre, which is more than
        # 2× the OBB half-width (25 px).  The function should not jump there.
        assert abs(result["cx"] - 330.0) < 100.0, (
            f"cx jumped to {result['cx']}: extend_obb should not latch onto "
            "a distant star when the OBB centre is dim"
        )


# ---------------------------------------------------------------------------
# nms_detections
# ---------------------------------------------------------------------------

class TestNmsDetections:
    def test_empty_input_returns_empty(self):
        from inference.postprocess import nms_detections
        assert nms_detections([]) == []

    def test_single_detection_always_kept(self):
        from inference.postprocess import nms_detections
        dets = [_make_det(0.9)]
        assert len(nms_detections(dets)) == 1

    def test_non_overlapping_detections_all_kept(self):
        """Two streaks far apart should both survive NMS."""
        from inference.postprocess import nms_detections
        det1 = _make_det(0.9, cx=50.0, cy=50.0, angle_deg=45.0)
        det2 = _make_det(0.8, cx=400.0, cy=400.0, angle_deg=10.0)
        result = nms_detections([det1, det2], iou_threshold=0.5)
        assert len(result) == 2

    def test_heavily_overlapping_lower_confidence_suppressed(self):
        """Two nearly identical detections → only the higher-confidence one survives."""
        from inference.postprocess import nms_detections
        det_high = _make_det(0.9, cx=64.0, cy=64.0, w=80.0, h=5.0, angle_deg=45.0)
        det_low  = _make_det(0.5, cx=64.5, cy=64.5, w=80.0, h=5.0, angle_deg=45.0)
        result = nms_detections([det_high, det_low], iou_threshold=0.5)
        assert len(result) == 1
        assert result[0]["confidence"] == pytest.approx(0.9)

    def test_result_sorted_by_confidence_descending(self):
        from inference.postprocess import nms_detections
        dets = [
            _make_det(0.5, cx=10.0, cy=10.0),
            _make_det(0.9, cx=200.0, cy=200.0),
            _make_det(0.7, cx=400.0, cy=400.0),
        ]
        result = nms_detections(dets)
        confidences = [d["confidence"] for d in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_all_suppressed_except_highest(self):
        """All detections at the same location → only rank-1 kept."""
        from inference.postprocess import nms_detections
        dets = [_make_det(conf, cx=64.0, cy=64.0) for conf in [0.9, 0.7, 0.5, 0.3]]
        result = nms_detections(dets, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0]["confidence"] == pytest.approx(0.9)

    def test_detections_without_obb_not_suppressed(self):
        """Detections missing 'obb' key are kept and don't cause crashes."""
        from inference.postprocess import nms_detections
        det_no_obb = {"confidence": 0.8}
        det_with_obb = _make_det(0.9)
        result = nms_detections([det_no_obb, det_with_obb])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# classify_detection_quality
# Source: SkyTrack (colleague) — StreakProcess reject flags
# ---------------------------------------------------------------------------

def _make_quality_det(
    cx: float = 256.0,
    cy: float = 256.0,
    w: float = 200.0,
    angle_deg: float = 0.0,
    confidence: float = 0.85,
    streak_length_px: float = 200.0,
    ra_tip1: float | None = 10.0,
    dec_tip1: float | None = 20.0,
    ra_tip2: float | None = 10.1,
    dec_tip2: float | None = 20.1,
) -> dict:
    """Build a minimal detection dict for quality classification tests."""
    return {
        "obb": {"cx": cx, "cy": cy, "w": w, "h": 8.0, "angle_deg": angle_deg},
        "confidence": confidence,
        "streak_length_px": streak_length_px,
        "ra_tip1_deg": ra_tip1,
        "dec_tip1_deg": dec_tip1,
        "ra_tip2_deg": ra_tip2,
        "dec_tip2_deg": dec_tip2,
    }


class TestClassifyDetectionQuality:
    IMAGE_SHAPE = (512, 512)

    def test_good_detection_returns_zero(self):
        from inference.postprocess import classify_detection_quality, QUALITY_GOOD
        det = _make_quality_det()
        assert classify_detection_quality(det, self.IMAGE_SHAPE) == QUALITY_GOOD

    def test_tip_on_left_edge_returns_edge_flag(self):
        """Tip within edge_margin_px of left border → QUALITY_EDGE."""
        from inference.postprocess import classify_detection_quality, QUALITY_EDGE
        # angle=0 → tip1_x = cx - w/2 = 5 (inside 20 px margin)
        det = _make_quality_det(cx=105.0, cy=256.0, w=200.0, angle_deg=0.0)
        assert classify_detection_quality(det, self.IMAGE_SHAPE, edge_margin_px=20) == QUALITY_EDGE
        assert det["edge_clipped"] is True
        assert det["edge_contacts"] == ["left"]

    def test_tip_on_right_edge_returns_edge_flag(self):
        from inference.postprocess import classify_detection_quality, QUALITY_EDGE
        # tip2_x = cx + w/2 = 402 + 100 = 502 > 512-20=492
        det = _make_quality_det(cx=402.0, cy=256.0, w=200.0, angle_deg=0.0)
        assert classify_detection_quality(det, self.IMAGE_SHAPE, edge_margin_px=20) == QUALITY_EDGE

    def test_tip_on_top_edge_returns_edge_flag(self):
        from inference.postprocess import classify_detection_quality, QUALITY_EDGE
        # vertical streak: angle=90 → tip_y = cy ± w/2 = 50 ± 100 → tip1_y = -50 (off edge)
        det = _make_quality_det(cx=256.0, cy=50.0, w=200.0, angle_deg=90.0)
        assert classify_detection_quality(det, self.IMAGE_SHAPE, edge_margin_px=20) == QUALITY_EDGE

    def test_low_confidence_returns_flag_two(self):
        from inference.postprocess import classify_detection_quality, QUALITY_LOW_CONF
        det = _make_quality_det(confidence=0.10)
        assert classify_detection_quality(det, self.IMAGE_SHAPE, min_confidence=0.30) == QUALITY_LOW_CONF

    def test_confidence_exactly_at_threshold_is_low_conf(self):
        """Confidence strictly below threshold → flag 2."""
        from inference.postprocess import classify_detection_quality, QUALITY_LOW_CONF
        det = _make_quality_det(confidence=0.299)
        assert classify_detection_quality(det, self.IMAGE_SHAPE, min_confidence=0.30) == QUALITY_LOW_CONF

    def test_confidence_at_threshold_is_good(self):
        """Confidence exactly equal to threshold passes."""
        from inference.postprocess import classify_detection_quality, QUALITY_GOOD
        det = _make_quality_det(confidence=0.30)
        assert classify_detection_quality(det, self.IMAGE_SHAPE, min_confidence=0.30) == QUALITY_GOOD
        assert det["edge_clipped"] is False
        assert det["edge_contacts"] == []

    def test_short_streak_returns_flag_three(self):
        from inference.postprocess import classify_detection_quality, QUALITY_TOO_SHORT
        det = _make_quality_det(streak_length_px=10.0)
        assert classify_detection_quality(det, self.IMAGE_SHAPE, min_length_px=50.0) == QUALITY_TOO_SHORT

    def test_no_sky_coords_returns_flag_four(self):
        from inference.postprocess import classify_detection_quality, QUALITY_NO_WCS
        det = _make_quality_det(ra_tip1=None, dec_tip1=None, ra_tip2=None, dec_tip2=None)
        assert classify_detection_quality(det, self.IMAGE_SHAPE) == QUALITY_NO_WCS

    def test_only_tip1_sky_coords_is_good(self):
        """If either tip has sky coords, WCS is present → not QUALITY_NO_WCS."""
        from inference.postprocess import classify_detection_quality, QUALITY_GOOD
        det = _make_quality_det(ra_tip2=None, dec_tip2=None)
        assert classify_detection_quality(det, self.IMAGE_SHAPE) == QUALITY_GOOD

    def test_edge_check_takes_priority_over_low_confidence(self):
        """Edge failure (flag 1) should be returned even if confidence is also low."""
        from inference.postprocess import classify_detection_quality, QUALITY_EDGE
        det = _make_quality_det(cx=105.0, cy=256.0, w=200.0, angle_deg=0.0, confidence=0.05)
        assert classify_detection_quality(det, self.IMAGE_SHAPE) == QUALITY_EDGE

    def test_low_conf_takes_priority_over_too_short(self):
        """Flag 2 returned before flag 3 when both conditions fail."""
        from inference.postprocess import classify_detection_quality, QUALITY_LOW_CONF
        det = _make_quality_det(confidence=0.05, streak_length_px=5.0)
        assert classify_detection_quality(det, self.IMAGE_SHAPE) == QUALITY_LOW_CONF


# ---------------------------------------------------------------------------
# group_detections
# ---------------------------------------------------------------------------

class TestGroupDetections:
    def test_empty_input_returns_empty(self):
        from inference.postprocess import group_detections
        assert group_detections([]) == []

    def test_all_dets_have_streak_id(self):
        from inference.postprocess import group_detections
        dets = [_make_det(0.9), _make_det(0.8, cx=400.0, cy=400.0)]
        result = group_detections(dets)
        assert all("streak_id" in d for d in result)

    def test_non_overlapping_get_different_streak_ids(self):
        from inference.postprocess import group_detections
        det1 = _make_det(0.9, cx=50.0, cy=50.0)
        det2 = _make_det(0.8, cx=400.0, cy=400.0)
        result = group_detections([det1, det2])
        ids = {d["streak_id"] for d in result}
        assert len(ids) == 2

    def test_collinear_fragments_share_streak_id(self):
        """Separated boxes on the same line should be one physical streak."""
        from inference.postprocess import group_detections

        det1 = {"confidence": 0.9, "method": "dinov3",
                "obb": {"cx": 150.0, "cy": 100.0, "w": 160.0, "h": 6.0, "angle_deg": 0.0}}
        det2 = {"confidence": 0.8, "method": "dinov3",
                "obb": {"cx": 360.0, "cy": 104.0, "w": 150.0, "h": 7.0, "angle_deg": 2.0}}

        result = group_detections([det1, det2])

        assert len({d["streak_id"] for d in result}) == 1

    def test_heavily_overlapping_same_streak_id(self):
        """Nearly identical detections from two methods → same streak_id."""
        from inference.postprocess import group_detections
        det1 = _make_det(0.9, cx=64.0, cy=64.0, angle_deg=45.0)
        det2 = _make_det(0.7, cx=64.5, cy=64.5, angle_deg=45.0)
        result = group_detections([det1, det2])
        ids = [d["streak_id"] for d in result]
        assert ids[0] == ids[1], "Heavily overlapping detections must share a streak_id"

    def test_thin_streak_lateral_offset_groups_correctly(self):
        """Two thin 5×500 OBBs shifted 3 px perpendicular have IoU~0.25 but IoM~0.7.

        This is the core regression: before IoM was added, these would get
        different streak_ids even though they clearly see the same streak.
        """
        from inference.postprocess import group_detections
        # Both OBBs represent the same 45° streak, one shifted 3 px perpendicular
        det1 = {"confidence": 0.9, "method": "ml",
                 "obb": {"cx": 250.0, "cy": 250.0, "w": 500.0, "h": 5.0, "angle_deg": 45.0}}
        # shift 3 px along the perpendicular direction (−sin45, cos45)
        perp = math.cos(math.radians(45.0))
        det2 = {"confidence": 0.7, "method": "opencv",
                 "obb": {"cx": 250.0 - 3 * perp, "cy": 250.0 + 3 * perp,
                         "w": 500.0, "h": 5.0, "angle_deg": 45.0}}
        result = group_detections([det1, det2])
        ids = [d["streak_id"] for d in result]
        assert ids[0] == ids[1], (
            "Thin streaks with small perpendicular offset must share a streak_id "
            "(IoU is low but IoM is high)"
        )

    def test_partial_detection_groups_with_full(self):
        """A 200 px partial detection should group with a 500 px full detection.

        IoU ≈ 0.4 → would fail with IoU-only grouping.
        IoM ≈ 1.0 → correctly groups via IoM.
        """
        from inference.postprocess import group_detections
        full  = {"confidence": 0.9, "method": "ml",
                  "obb": {"cx": 250.0, "cy": 128.0, "w": 500.0, "h": 5.0, "angle_deg": 0.0}}
        # Partial: same centre x, aligned, but only 200 px long
        partial = {"confidence": 0.6, "method": "classical",
                    "obb": {"cx": 250.0, "cy": 128.0, "w": 200.0, "h": 5.0, "angle_deg": 0.0}}
        result = group_detections([full, partial])
        ids = [d["streak_id"] for d in result]
        assert ids[0] == ids[1], (
            "Partial detection must group with the full detection of the same streak"
        )


class TestFuseGroupGeometries:
    def test_fuses_group_fragments_to_outer_endpoints(self):
        from inference.postprocess import fuse_group_geometries

        dets = [
            {"streak_id": 1, "confidence": 0.9,
             "obb": {"cx": 150.0, "cy": 100.0, "w": 160.0, "h": 6.0, "angle_deg": 0.0},
             "streak_length_px": 160.0},
            {"streak_id": 1, "confidence": 0.8,
             "obb": {"cx": 360.0, "cy": 104.0, "w": 150.0, "h": 8.0, "angle_deg": 2.0},
             "streak_length_px": 150.0},
        ]

        fused = fuse_group_geometries(dets)

        # New endpoint-based fusion projects OBB-derived endpoints onto the axis,
        # giving a result within 1% of the exact OBB-interval value.
        assert fused[0]["obb"]["w"] == pytest.approx(365.0, rel=0.01)
        assert fused[1]["obb"]["w"] == pytest.approx(365.0, rel=0.01)
        assert fused[0]["obb"]["cx"] == pytest.approx(252.5, rel=0.01)
        assert fused[1]["streak_length_px"] == pytest.approx(365.0, rel=0.01)

    def test_fuse_uses_longest_member_axis_over_highest_confidence(self):
        from inference.postprocess import fuse_group_geometries

        dets = [
            {"streak_id": 1, "confidence": 0.99,
             "obb": {"cx": 100.0, "cy": 100.0, "w": 100.0, "h": 5.0, "angle_deg": 90.0}},
            {"streak_id": 1, "confidence": 0.50,
             "obb": {"cx": 150.0, "cy": 100.0, "w": 300.0, "h": 8.0, "angle_deg": 112.0}},
        ]

        fused = fuse_group_geometries(dets)

        assert fused[0]["obb"]["angle_deg"] == pytest.approx(112.0)
        assert fused[1]["obb"]["angle_deg"] == pytest.approx(112.0)
