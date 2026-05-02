"""Tests for src/ingest/fits_parser.py."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from src.ingest.fits_parser import FITSImage, parse_fits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_minimal_fits(path: Path, extra_headers: dict | None = None, data=None) -> Path:
    """Write a minimal valid FITS file for testing."""
    if data is None:
        data = np.zeros((64, 64), dtype=np.uint16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["DATE-OBS"] = "2024-04-02T02:55:24.38"
    hdu.header["NAXIS1"] = data.shape[1]
    hdu.header["NAXIS2"] = data.shape[0]
    if extra_headers:
        for k, v in extra_headers.items():
            hdu.header[k] = v
    hdu.writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseFitsSuccess:
    def test_returns_fits_image_instance(self, tmp_path):
        f = _write_minimal_fits(tmp_path / "test.fits")
        result = parse_fits(f)
        assert isinstance(result, FITSImage)

    def test_image_data_is_float32(self, tmp_path):
        f = _write_minimal_fits(tmp_path / "test.fits")
        result = parse_fits(f)
        assert result.data.dtype == np.float32

    def test_image_shape_matches_naxis(self, tmp_path):
        data = np.zeros((32, 48), dtype=np.uint16)
        f = _write_minimal_fits(tmp_path / "test.fits", data=data)
        result = parse_fits(f)
        assert result.width_px == 48
        assert result.height_px == 32

    def test_filepath_is_absolute(self, tmp_path):
        f = _write_minimal_fits(tmp_path / "test.fits")
        result = parse_fits(f)
        assert result.filepath.is_absolute()


class TestDateObsParsing:
    def test_t_separated_format(self, tmp_path):
        f = _write_minimal_fits(tmp_path / "t.fits", {"DATE-OBS": "2024-04-02T02:55:24.38"})
        result = parse_fits(f)
        assert result.obs_time.year == 2024
        assert result.obs_time.month == 4
        assert result.obs_time.day == 2
        assert result.obs_time.hour == 2
        assert result.obs_time.minute == 55

    def test_space_separated_format(self, tmp_path):
        f = _write_minimal_fits(tmp_path / "s.fits", {"DATE-OBS": "2024-04-02 02:55:24"})
        result = parse_fits(f)
        assert result.obs_time.year == 2024
        assert result.obs_time.second == 24

    def test_obs_time_is_utc_aware(self, tmp_path):
        f = _write_minimal_fits(tmp_path / "tz.fits")
        result = parse_fits(f)
        assert result.obs_time.tzinfo is not None
        assert result.obs_time.tzinfo == timezone.utc

    def test_explicit_utc_offset_preserved(self, tmp_path):
        f = _write_minimal_fits(tmp_path / "utc.fits", {"DATE-OBS": "2024-04-02T02:55:24+00:00"})
        result = parse_fits(f)
        assert result.obs_time == datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)


class TestRequiredFieldValidation:
    def test_missing_date_obs_raises_value_error(self, tmp_path):
        data = np.zeros((64, 64), dtype=np.uint16)
        hdu = fits.PrimaryHDU(data)
        hdu.header["NAXIS1"] = 64
        hdu.header["NAXIS2"] = 64
        # Deliberately omit DATE-OBS
        path = tmp_path / "no_date.fits"
        hdu.writeto(path)
        with pytest.raises(ValueError, match="DATE-OBS"):
            parse_fits(path)

    def test_missing_naxis1_raises_value_error(self, tmp_path):
        # data=None → NAXIS=0, so neither NAXIS1 nor NAXIS2 are written by astropy
        hdu = fits.PrimaryHDU()
        hdu.header["DATE-OBS"] = "2024-04-02T02:55:24"
        path = tmp_path / "no_naxis1.fits"
        hdu.writeto(path)
        with pytest.raises(ValueError, match="NAXIS1"):
            parse_fits(path)

    def test_missing_naxis2_raises_value_error(self, tmp_path):
        # 1-D array → astropy writes NAXIS=1, NAXIS1 only; NAXIS2 is absent
        hdu = fits.PrimaryHDU(np.zeros(64, dtype=np.uint16))
        hdu.header["DATE-OBS"] = "2024-04-02T02:55:24"
        path = tmp_path / "no_naxis2.fits"
        hdu.writeto(path)
        with pytest.raises(ValueError, match="NAXIS2"):
            parse_fits(path)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_fits(tmp_path / "nonexistent.fits")


class TestOptionalFields:
    def test_missing_optional_fields_return_none(self, tmp_path):
        f = _write_minimal_fits(tmp_path / "minimal.fits")
        result = parse_fits(f)
        assert result.ra_center is None
        assert result.dec_center is None
        assert result.pixscale_arcsec is None
        assert result.exptime_sec is None
        assert result.sitelat is None
        assert result.sitelong is None
        assert result.siteelev is None

    def test_optional_fields_parsed_when_present(self, tmp_path):
        extra = {
            "CRVAL1": 123.456,
            "CRVAL2": -45.678,
            "PIXSCALE": 1.23,
            "EXPTIME": 30.0,
            "SITELAT": 37.5,
            "SITELONG": -122.0,
            "SITEELEV": 100.0,
        }
        f = _write_minimal_fits(tmp_path / "full.fits", extra)
        result = parse_fits(f)
        assert result.ra_center == pytest.approx(123.456)
        assert result.dec_center == pytest.approx(-45.678)
        assert result.pixscale_arcsec == pytest.approx(1.23)
        assert result.exptime_sec == pytest.approx(30.0)
        assert result.sitelat == pytest.approx(37.5)
        assert result.sitelong == pytest.approx(-122.0)
        assert result.siteelev == pytest.approx(100.0)

    def test_pixscale_falls_back_to_cdelt1(self, tmp_path):
        # CDELT1 in degrees/pixel → arcsec/pixel = abs(CDELT1) * 3600
        f = _write_minimal_fits(tmp_path / "cdelt.fits", {"CDELT1": -0.0005})
        result = parse_fits(f)
        assert result.pixscale_arcsec == pytest.approx(0.0005 * 3600)

    def test_pixscale_prefers_pixscale_over_cdelt1(self, tmp_path):
        f = _write_minimal_fits(tmp_path / "both.fits", {"PIXSCALE": 2.0, "CDELT1": -0.0005})
        result = parse_fits(f)
        assert result.pixscale_arcsec == pytest.approx(2.0)
