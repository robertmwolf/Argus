"""Tests for src/astrometry/plate_solver.py."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

from src.astrometry.plate_solver import PlateSolver, _celestial_position_angle
from src.detection.streak import StreakDetection
from src.ingest.fits_parser import FITSImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wcs_header(
    crval1: float = 180.0,
    crval2: float = 30.0,
    crpix1: float = 512.0,
    crpix2: float = 512.0,
    cdelt: float = -0.000277778,  # ~1 arcsec/px in degrees
    naxis1: int = 1024,
    naxis2: int = 1024,
) -> fits.Header:
    """Build a minimal TAN-projection FITS header with WCS."""
    header = fits.Header()
    header["DATE-OBS"] = "2024-04-02T02:55:24.38"
    header["NAXIS"]  = 2
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    header["CRPIX1"] = crpix1
    header["CRPIX2"] = crpix2
    header["CDELT1"] = cdelt
    header["CDELT2"] = abs(cdelt)
    header["EQUINOX"] = 2000.0
    return header


def _make_fits_image(
    header: fits.Header,
    exptime_sec: float | None = 10.0,
    pixscale_arcsec: float | None = 1.0,
) -> FITSImage:
    """Wrap a FITS header into a minimal FITSImage."""
    naxis1 = int(header.get("NAXIS1", 1024))
    naxis2 = int(header.get("NAXIS2", 1024))
    crval1 = header.get("CRVAL1", None)
    crval2 = header.get("CRVAL2", None)
    return FITSImage(
        filepath=Path("/fake/test.fits"),
        obs_time=datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc),
        ra_center=float(crval1) if crval1 is not None else None,
        dec_center=float(crval2) if crval2 is not None else None,
        width_px=naxis1,
        height_px=naxis2,
        pixscale_arcsec=pixscale_arcsec,
        exptime_sec=exptime_sec,
        sitelat=None,
        sitelong=None,
        siteelev=None,
        data=np.zeros((naxis2, naxis1), dtype=np.float32),
        header=header,
    )


def _make_streak(
    x_start: float = 400.0,
    y_start: float = 500.0,
    x_end: float = 600.0,
    y_end: float = 500.0,
) -> StreakDetection:
    """Make a synthetic StreakDetection with no sky coords."""
    return StreakDetection(
        x_start=x_start,
        y_start=y_start,
        x_end=x_end,
        y_end=y_end,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCenterPixelMapsToWCS:
    """Center pixel should map to approximately CRVAL when crpix is near center."""

    def test_center_pixel_maps_to_crval(self):
        crval1, crval2 = 180.0, 30.0
        crpix1, crpix2 = 512.0, 512.0
        header = _make_wcs_header(
            crval1=crval1, crval2=crval2,
            crpix1=crpix1, crpix2=crpix2,
        )
        img = _make_fits_image(header)
        # Streak whose center is at CRPIX (0-indexed → CRPIX-1)
        det = _make_streak(
            x_start=crpix1 - 50,
            y_start=crpix2 - 1,
            x_end=crpix1 + 50,
            y_end=crpix2 - 1,
        )
        solver = PlateSolver()
        solver.solve(img, det)

        assert det.ra_center is not None
        assert det.dec_center is not None
        # Center pixel x=crpix1-1 (0-indexed) should map close to CRVAL1
        # ra_center is near CRVAL1 since streak is symmetric around crpix
        assert abs(det.dec_center - crval2) < 0.01

    def test_crpix_pixel_maps_to_crval(self):
        """A streak centered exactly at CRPIX-1 (0-indexed) maps to CRVAL."""
        crval1, crval2 = 45.0, -20.0
        crpix1, crpix2 = 512.0, 512.0
        header = _make_wcs_header(
            crval1=crval1, crval2=crval2,
            crpix1=crpix1, crpix2=crpix2,
        )
        img = _make_fits_image(header)
        # Single-point streak at CRPIX-1 (0-indexed astropy convention)
        px = crpix1 - 1  # CRPIX is 1-indexed in FITS; astropy all_pix2world with origin=0
        py = crpix2 - 1
        det = _make_streak(x_start=px, y_start=py, x_end=px + 100, y_end=py)
        solver = PlateSolver()
        solver.solve(img, det)
        # Start should be very close to CRVAL
        assert abs(det.ra_start - crval1) < 0.01
        assert abs(det.dec_start - crval2) < 0.01


class TestAngularVelocity:
    def test_angular_velocity_positive(self):
        header = _make_wcs_header()
        img = _make_fits_image(header, exptime_sec=10.0)
        det = _make_streak(x_start=400, y_start=511, x_end=624, y_end=511)
        PlateSolver().solve(img, det)
        assert det.angular_velocity_arcsec_s is not None
        assert det.angular_velocity_arcsec_s > 0

    def test_angular_velocity_in_leo_range(self):
        """For a ~200 px streak at 1 arcsec/px and 10s exptime, expect ~20 arcsec/s."""
        # cdelt = 1 arcsec/px = 1/3600 deg/px
        header = _make_wcs_header(cdelt=-1.0 / 3600.0)
        img = _make_fits_image(header, exptime_sec=10.0, pixscale_arcsec=1.0)
        # 200 px streak ≈ 200 arcsec → 20 arcsec/s with 10s exptime
        det = _make_streak(x_start=412, y_start=511, x_end=612, y_end=511)
        PlateSolver().solve(img, det)
        assert det.angular_velocity_arcsec_s is not None
        # Should be ~20 arcsec/s — definitely in the plausible satellite range
        assert 1.0 < det.angular_velocity_arcsec_s < 7200.0  # 0.001–2 deg/s

    def test_angular_velocity_none_when_no_exptime(self):
        header = _make_wcs_header()
        img = _make_fits_image(header, exptime_sec=None)
        det = _make_streak()
        PlateSolver().solve(img, det)
        assert det.angular_velocity_arcsec_s is None

    def test_angular_velocity_none_when_exptime_zero(self):
        header = _make_wcs_header()
        img = _make_fits_image(header, exptime_sec=0.0)
        det = _make_streak()
        PlateSolver().solve(img, det)
        assert det.angular_velocity_arcsec_s is None


class TestNoWCS:
    def test_no_wcs_sky_fields_remain_none(self):
        """A header with no WCS keywords must leave all sky fields as None."""
        header = fits.Header()
        header["DATE-OBS"] = "2024-04-02T02:55:24.38"
        header["NAXIS"]  = 2
        header["NAXIS1"] = 512
        header["NAXIS2"] = 512
        img = _make_fits_image(header)
        det = _make_streak()
        PlateSolver().solve(img, det)
        assert det.ra_start   is None
        assert det.dec_start  is None
        assert det.ra_end     is None
        assert det.dec_end    is None
        assert det.ra_center  is None
        assert det.dec_center is None
        assert det.angular_velocity_arcsec_s is None
        assert det.position_angle_deg is None

    def test_no_wcs_does_not_raise(self):
        """Must not raise any exception — just silently skip."""
        header = fits.Header()
        header["DATE-OBS"] = "2024-04-02T02:55:24.38"
        header["NAXIS"]  = 2
        header["NAXIS1"] = 512
        header["NAXIS2"] = 512
        img = _make_fits_image(header)
        det = _make_streak()
        # Should complete without exception
        result = PlateSolver().solve(img, det)
        assert result is det


class TestPositionAngle:
    def test_position_angle_in_range(self):
        header = _make_wcs_header()
        img = _make_fits_image(header)
        det = _make_streak(x_start=400, y_start=400, x_end=600, y_end=500)
        PlateSolver().solve(img, det)
        if det.position_angle_deg is not None:
            assert 0.0 <= det.position_angle_deg < 360.0

    def test_position_angle_horizontal_streak(self):
        """Horizontal streak in TAN projection should have PA near 90° or 270°."""
        header = _make_wcs_header()
        img = _make_fits_image(header)
        det = _make_streak(x_start=412, y_start=511, x_end=612, y_end=511)
        PlateSolver().solve(img, det)
        assert det.position_angle_deg is not None
        # PA for westward motion in standard TAN ~ 90 or 270 degrees
        pa = det.position_angle_deg
        assert (abs(pa - 90.0) < 10.0 or abs(pa - 270.0) < 10.0)


class TestSkyFieldsPopulated:
    def test_all_sky_fields_set(self):
        header = _make_wcs_header()
        img = _make_fits_image(header)
        det = _make_streak()
        PlateSolver().solve(img, det)
        for field in (
            "ra_start", "dec_start", "ra_end", "dec_end",
            "ra_center", "dec_center", "position_angle_deg",
        ):
            assert getattr(det, field) is not None, f"{field} should be set"

    def test_solve_returns_same_object(self):
        header = _make_wcs_header()
        img = _make_fits_image(header)
        det = _make_streak()
        result = PlateSolver().solve(img, det)
        assert result is det


class TestCelestialPositionAngle:
    def test_due_north_is_zero(self):
        """Going straight north: same RA, higher Dec → PA = 0°."""
        pa = _celestial_position_angle(180.0, 30.0, 180.0, 31.0)
        assert abs(pa) < 0.1 or abs(pa - 360.0) < 0.1

    def test_due_east_is_90(self):
        """Going east (increasing RA at same Dec near equator) → PA ≈ 90°."""
        pa = _celestial_position_angle(180.0, 0.0, 181.0, 0.0)
        assert abs(pa - 90.0) < 1.0

    def test_returns_in_0_360_range(self):
        for ra1, dec1, ra2, dec2 in [
            (10.0, 20.0, 11.0, 21.0),
            (350.0, -10.0, 5.0, -9.0),
            (180.0, 45.0, 179.0, 44.0),
        ]:
            pa = _celestial_position_angle(ra1, dec1, ra2, dec2)
            assert 0.0 <= pa < 360.0
