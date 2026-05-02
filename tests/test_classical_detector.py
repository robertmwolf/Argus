"""Tests for src/detection/classical_detector.py."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from src.detection.classical_detector import StreakDetection, detect_streaks
from src.ingest.fits_parser import FITSImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fits_image(
    data: np.ndarray,
    obs_time: datetime | None = None,
) -> FITSImage:
    """Build a minimal FITSImage from a numpy array."""
    if obs_time is None:
        obs_time = datetime(2022, 8, 7, 21, 0, 0, tzinfo=timezone.utc)
    h, w = data.shape
    header = fits.Header()
    header["DATE-OBS"] = obs_time.isoformat()
    header["NAXIS1"] = w
    header["NAXIS2"] = h
    return FITSImage(
        filepath=Path("/fake/test.fits"),
        obs_time=obs_time,
        ra_center=None,
        dec_center=None,
        width_px=w,
        height_px=h,
        pixscale_arcsec=1.238,
        exptime_sec=10.0,
        sitelat=None,
        sitelong=None,
        siteelev=None,
        data=data.astype(np.float32),
        header=header,
    )


def _blank_image(height: int = 256, width: int = 256) -> np.ndarray:
    """Poisson noise background — no streak."""
    rng = np.random.default_rng(0)
    return rng.poisson(100, size=(height, width)).astype(np.float32)


def _image_with_streak(
    height: int = 256,
    width: int = 256,
    streak_brightness: int = 8000,
) -> np.ndarray:
    """Noise background with a bright horizontal streak."""
    rng = np.random.default_rng(1)
    data = rng.poisson(100, size=(height, width)).astype(np.float32)
    y_row = height // 2
    # Bright streak across the full width
    data[y_row, 20 : width - 20] = streak_brightness
    data[y_row + 1, 20 : width - 20] = streak_brightness // 2
    return data


# ---------------------------------------------------------------------------
# Tests: no streak image
# ---------------------------------------------------------------------------

class TestNoStreak:
    def test_returns_list(self):
        img = _make_fits_image(_blank_image())
        result = detect_streaks(img)
        assert isinstance(result, list)

    def test_blank_image_returns_empty_or_few(self):
        """Pure noise should yield zero or very few detections."""
        img = _make_fits_image(_blank_image())
        result = detect_streaks(img, contour_threshold=3.0, min_length_px=20)
        # Noise images occasionally produce very short artefacts;
        # nothing longer than min_length_px should survive.
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: image with a streak
# ---------------------------------------------------------------------------

class TestWithStreak:
    @pytest.fixture(scope="class")
    def detections(self):
        img = _make_fits_image(_image_with_streak())
        return detect_streaks(img, contour_threshold=3.0, min_length_px=20)

    def test_finds_at_least_one_streak(self, detections):
        assert len(detections) >= 1

    def test_returns_streak_detection_instances(self, detections):
        for d in detections:
            assert isinstance(d, StreakDetection)

    def test_length_px_positive(self, detections):
        for d in detections:
            assert d.length_px > 0

    def test_angle_deg_in_range(self, detections):
        for d in detections:
            assert -180.0 < d.angle_deg <= 180.0

    def test_area_px_positive(self, detections):
        for d in detections:
            assert d.area_px > 0

    def test_width_px_non_negative(self, detections):
        for d in detections:
            assert d.width_px >= 0

    def test_sky_coords_are_none(self, detections):
        """Sky fields must be None until plate solver runs."""
        for d in detections:
            assert d.ra_start is None
            assert d.dec_start is None
            assert d.ra_end is None
            assert d.dec_end is None
            assert d.ra_center is None
            assert d.dec_center is None
            assert d.angular_velocity_arcsec_s is None
            assert d.position_angle_deg is None

    def test_center_is_midpoint_of_endpoints(self, detections):
        for d in detections:
            assert d.x_center == pytest.approx((d.x_start + d.x_end) / 2, abs=0.5)
            assert d.y_center == pytest.approx((d.y_start + d.y_end) / 2, abs=0.5)

    def test_streak_roughly_horizontal(self, detections):
        """The injected streak is horizontal — angle should be near 0 or ±180."""
        longest = max(detections, key=lambda d: d.length_px)
        assert abs(longest.angle_deg) < 10 or abs(abs(longest.angle_deg) - 180) < 10


# ---------------------------------------------------------------------------
# Tests: tunable parameters
# ---------------------------------------------------------------------------

class TestParameters:
    def test_higher_contour_threshold_finds_fewer_or_equal(self):
        """Stricter threshold should not find more streaks than a loose one."""
        img = _make_fits_image(_image_with_streak())
        loose = detect_streaks(img, contour_threshold=2.0, min_length_px=20)
        strict = detect_streaks(img, contour_threshold=5.0, min_length_px=20)
        assert len(strict) <= len(loose)

    def test_min_length_filters_short_detections(self):
        """Increasing min_length_px should reduce or keep the detection count."""
        img = _make_fits_image(_image_with_streak())
        short_allowed = detect_streaks(img, contour_threshold=3.0, min_length_px=5)
        long_required = detect_streaks(img, contour_threshold=3.0, min_length_px=100)
        assert len(long_required) <= len(short_allowed)

    def test_all_detections_exceed_min_length(self):
        min_len = 50.0
        img = _make_fits_image(_image_with_streak())
        dets = detect_streaks(img, contour_threshold=3.0, min_length_px=min_len)
        for d in dets:
            assert d.length_px >= min_len
