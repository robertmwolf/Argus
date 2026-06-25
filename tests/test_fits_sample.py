"""Tests against the committed Brent/Atwood FITS sample in tests/data/test/.

streak_brent_sample.fits is a 512×512 uint16 crop from a real ZWO ASI2600MM Pro
observation (2026-05-16, Atwood observatory).  It has DATE-OBS, EXPOSURE,
SITELAT/SITELONG/SITEELEV, and no WCS solution — representative of the
unplatesolved images that flow through the production inference pipeline.

These tests always run (no real_data marker) because the file is committed.
"""

from __future__ import annotations

import os
from datetime import timezone
from pathlib import Path

import numpy as np
import pytest

SAMPLE = Path(__file__).parent / "data" / "test" / "streak_brent_sample.fits"


def _load_fits(norm: str) -> dict:
    """Load the sample file with a specific normalisation mode."""
    import inference.fits_loader as fits_loader_mod
    from inference.fits_loader import FITSLoader
    old = os.environ.get("ARGUS_NORM")
    os.environ["ARGUS_NORM"] = norm
    try:
        return FITSLoader().load(SAMPLE)
    finally:
        if old is None:
            os.environ.pop("ARGUS_NORM", None)
        else:
            os.environ["ARGUS_NORM"] = old


# ---------------------------------------------------------------------------
# src.ingest.fits_parser
# ---------------------------------------------------------------------------


class TestFitsParser:
    @pytest.fixture(scope="class")
    def image(self):
        from src.ingest.fits_parser import parse_fits
        return parse_fits(SAMPLE)

    def test_dimensions_match_crop(self, image) -> None:
        assert image.width_px == 512
        assert image.height_px == 512

    def test_data_is_float32_2d(self, image) -> None:
        assert image.data.ndim == 2
        assert image.data.dtype == np.float32

    def test_obs_time_is_utc_aware(self, image) -> None:
        assert image.obs_time is not None
        assert image.obs_time.tzinfo is not None
        assert image.obs_time.tzinfo == timezone.utc

    def test_obs_time_value(self, image) -> None:
        assert image.obs_time.year == 2026
        assert image.obs_time.month == 5
        assert image.obs_time.day == 16

    def test_exposure_time_parsed(self, image) -> None:
        assert image.exptime_sec == pytest.approx(0.5)

    def test_observer_coords_present(self, image) -> None:
        assert image.sitelat == pytest.approx(43.6735556, rel=1e-4)
        assert image.sitelong == pytest.approx(-81.0204722, rel=1e-4)
        assert image.siteelev == pytest.approx(365.0, abs=1.0)

    def test_pixel_values_positive(self, image) -> None:
        assert image.data.min() >= 0.0

    def test_pixel_values_nonzero_variance(self, image) -> None:
        assert image.data.std() > 0.0

    def test_wcs_absent(self, image) -> None:
        # No plate-solve on this file — ra_center and dec_center should be None.
        assert image.ra_center is None
        assert image.dec_center is None


# ---------------------------------------------------------------------------
# inference.fits_loader (zscore normalisation — default)
# ---------------------------------------------------------------------------


class TestFitsLoaderZscore:
    @pytest.fixture(scope="class")
    def result(self):
        return _load_fits("zscore")

    def test_array_is_uint8_rgb(self, result) -> None:
        arr = result["array"]
        assert arr.ndim == 3
        assert arr.shape == (512, 512, 3)
        assert arr.dtype == np.uint8

    def test_raw_float32_shape(self, result) -> None:
        assert result["raw_float32"].shape == (512, 512)
        assert result["raw_float32"].dtype == np.float32

    def test_array_has_nonzero_variance(self, result) -> None:
        assert result["array"].std() > 0

    def test_obs_time_present(self, result) -> None:
        assert result["obs_time"] is not None

    def test_observer_lat_present(self, result) -> None:
        assert result["observer_lat"] == pytest.approx(43.6735556, rel=1e-4)

    def test_observer_lon_present(self, result) -> None:
        assert result["observer_lon"] == pytest.approx(-81.0204722, rel=1e-4)

    def test_filename_field(self, result) -> None:
        assert result["filename"] == "streak_brent_sample.fits"

    def test_shape_field(self, result) -> None:
        assert result["shape"] == (512, 512)

    def test_norm_mode_recorded(self, result) -> None:
        assert result["norm_mode"] == "zscore"

    def test_wcs_is_none_without_plate_solve(self, result) -> None:
        assert result["wcs"] is None


# ---------------------------------------------------------------------------
# inference.fits_loader (autostretch normalisation)
# ---------------------------------------------------------------------------


class TestFitsLoaderAutostretch:
    @pytest.fixture(scope="class")
    def result(self):
        return _load_fits("autostretch")

    def test_array_is_uint8_rgb(self, result) -> None:
        arr = result["array"]
        assert arr.shape == (512, 512, 3)
        assert arr.dtype == np.uint8

    def test_array_has_nonzero_variance(self, result) -> None:
        assert result["array"].std() > 0

    def test_norm_mode_recorded(self, result) -> None:
        assert result["norm_mode"] == "autostretch"


# ---------------------------------------------------------------------------
# inference.fits_loader (zscale normalisation)
# ---------------------------------------------------------------------------


class TestFitsLoaderZscale:
    @pytest.fixture(scope="class")
    def result(self):
        return _load_fits("zscale")

    def test_array_is_uint8_rgb(self, result) -> None:
        arr = result["array"]
        assert arr.shape == (512, 512, 3)
        assert arr.dtype == np.uint8

    def test_norm_mode_recorded(self, result) -> None:
        assert result["norm_mode"] == "zscale"


# ---------------------------------------------------------------------------
# Cross-check: both loaders agree on obs_time and dimensions
# ---------------------------------------------------------------------------


def test_parser_and_loader_agree_on_obs_time() -> None:
    from src.ingest.fits_parser import parse_fits

    parsed = parse_fits(SAMPLE)
    loaded = _load_fits("zscore")

    assert parsed.obs_time == loaded["obs_time"]


def test_parser_and_loader_agree_on_dimensions() -> None:
    from src.ingest.fits_parser import parse_fits

    parsed = parse_fits(SAMPLE)
    loaded = _load_fits("zscore")

    h, w = loaded["shape"]
    assert parsed.width_px == w
    assert parsed.height_px == h
