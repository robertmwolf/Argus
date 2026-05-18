"""Tests for inference.fits_loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from inference.fits_loader import FITSLoader


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_fits(tmp_path: Path, with_wcs: bool = False, exptime: float | None = 10.0) -> Path:
    """Create a minimal synthetic FITS file for testing."""
    rng = np.random.default_rng(42)
    data = rng.normal(1000.0, 100.0, (64, 64)).astype(np.float32)

    hdu = fits.PrimaryHDU(data)
    hdu.header["NAXIS1"] = 64
    hdu.header["NAXIS2"] = 64

    if exptime is not None:
        hdu.header["EXPTIME"] = exptime

    if with_wcs:
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
        hdu.header["CRVAL1"] = 180.0
        hdu.header["CRVAL2"] = 45.0
        hdu.header["CRPIX1"] = 32.0
        hdu.header["CRPIX2"] = 32.0
        hdu.header["CDELT1"] = -0.001
        hdu.header["CDELT2"] = 0.001

    out = tmp_path / "test_image.fits"
    hdu.writeto(out, overwrite=True)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFITSLoaderLoad:
    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path)
        loader = FITSLoader()
        result = loader.load(fits_path)
        assert set(result.keys()) >= {"array", "wcs", "exposure_time", "filename", "shape",
                                       "obs_time", "observer_lat", "observer_lon", "observer_alt_m"}

    def test_array_dtype_uint8(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path)
        result = FITSLoader().load(fits_path)
        assert result["array"].dtype == np.uint8

    def test_array_shape_h_w_3(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path)
        result = FITSLoader().load(fits_path)
        h, w = result["shape"]
        assert result["array"].shape == (h, w, 3)
        assert h == 64
        assert w == 64

    def test_zscale_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fits_path = _make_fits(tmp_path)
        monkeypatch.setenv("ARGUS_NORM", "zscale")

        result = FITSLoader().load(fits_path)

        assert result["norm_mode"] == "zscale"
        assert result["array"].dtype == np.uint8
        assert result["array"].shape == (64, 64, 3)

    def test_shape_tuple(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path)
        result = FITSLoader().load(fits_path)
        assert result["shape"] == (64, 64)

    def test_filename_basename_only(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path)
        result = FITSLoader().load(fits_path)
        assert result["filename"] == "test_image.fits"
        assert "/" not in result["filename"]

    def test_exposure_time_present(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path, exptime=30.0)
        result = FITSLoader().load(fits_path)
        assert result["exposure_time"] == pytest.approx(30.0)

    def test_exposure_time_absent_is_none(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path, exptime=None)
        result = FITSLoader().load(fits_path)
        assert result["exposure_time"] is None

    def test_wcs_none_when_no_wcs_headers(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path, with_wcs=False)
        result = FITSLoader().load(fits_path)
        assert result["wcs"] is None

    def test_wcs_present_when_wcs_headers_exist(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path, with_wcs=True)
        result = FITSLoader().load(fits_path)
        assert result["wcs"] is not None
        assert isinstance(result["wcs"], WCS)
        assert result["wcs_source"] == "fits"

    def test_wcs_loaded_from_sidecar_when_fits_header_has_none(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path, with_wcs=False)
        sidecar_header = fits.Header()
        sidecar_header["SIMPLE"] = True
        sidecar_header["BITPIX"] = 16
        sidecar_header["NAXIS"] = 0
        sidecar_header["CTYPE1"] = "RA---TAN"
        sidecar_header["CTYPE2"] = "DEC--TAN"
        sidecar_header["CRVAL1"] = 180.0
        sidecar_header["CRVAL2"] = 45.0
        sidecar_header["CRPIX1"] = 32.0
        sidecar_header["CRPIX2"] = 32.0
        sidecar_header["CDELT1"] = -0.001
        sidecar_header["CDELT2"] = 0.001
        fits_path.with_suffix(".wcs").write_text(
            sidecar_header.tostring(sep="\n", endcard=True, padding=False)
        )

        result = FITSLoader().load(fits_path)

        assert result["wcs"] is not None
        assert isinstance(result["wcs"], WCS)
        assert result["wcs_source"] == "sidecar"

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        loader = FITSLoader()
        with pytest.raises((FileNotFoundError, ValueError)):
            loader.load(tmp_path / "does_not_exist.fits")

    def test_invalid_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.fits"
        bad.write_bytes(b"not a fits file at all")
        loader = FITSLoader()
        with pytest.raises((ValueError, OSError)):
            loader.load(bad)


class TestFITSLoaderFitsToPng:
    def test_creates_png(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path)
        out_png = tmp_path / "out.png"
        loader = FITSLoader()
        loader.fits_to_png(fits_path, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 0


class TestExtractWcsMetadata:
    def test_returns_list_of_dicts(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path, with_wcs=True)
        loader = FITSLoader()
        result = loader.load(fits_path)
        wcs = result["wcs"]
        assert wcs is not None

        coords = [(10.0, 20.0), (30.0, 40.0)]
        meta = loader.extract_wcs_metadata(wcs, coords)
        assert len(meta) == 2
        for entry in meta:
            assert set(entry.keys()) == {"x_pix", "y_pix", "ra_deg", "dec_deg"}

    def test_pixel_values_preserved(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path, with_wcs=True)
        loader = FITSLoader()
        wcs = loader.load(fits_path)["wcs"]

        coords = [(5.0, 7.0)]
        meta = loader.extract_wcs_metadata(wcs, coords)
        assert meta[0]["x_pix"] == pytest.approx(5.0)
        assert meta[0]["y_pix"] == pytest.approx(7.0)

    def test_ra_dec_are_floats(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path, with_wcs=True)
        loader = FITSLoader()
        wcs = loader.load(fits_path)["wcs"]
        meta = loader.extract_wcs_metadata(wcs, [(32.0, 32.0)])
        assert isinstance(meta[0]["ra_deg"], float)
        assert isinstance(meta[0]["dec_deg"], float)

    def test_empty_coords_returns_empty_list(self, tmp_path: Path) -> None:
        fits_path = _make_fits(tmp_path, with_wcs=True)
        loader = FITSLoader()
        wcs = loader.load(fits_path)["wcs"]
        assert loader.extract_wcs_metadata(wcs, []) == []
