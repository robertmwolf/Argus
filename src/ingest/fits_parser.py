"""FITS file ingestion and header parsing for the ARGUS pipeline.

Reads a .fits or .fit file, extracts metadata and image data into a
FITSImage dataclass for downstream processing.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


@dataclass
class FITSImage:
    """Parsed representation of a single FITS observation.

    Required fields are always populated. Optional fields are None when
    the corresponding FITS header keyword is absent.

    Attributes:
        filepath: Absolute path to the source .fits file.
        obs_time: Observation start time (UTC, timezone-aware).
        ra_center: Right ascension of FOV centre, degrees. None if absent.
        dec_center: Declination of FOV centre, degrees. None if absent.
        width_px: Image width in pixels (NAXIS1).
        height_px: Image height in pixels (NAXIS2).
        pixscale_arcsec: Pixel scale in arcsec/pixel. None if absent.
        exptime_sec: Exposure duration in seconds. None if absent.
        sitelat: Observer geodetic latitude, degrees. None if absent.
        sitelong: Observer geodetic longitude, degrees. None if absent.
        siteelev: Observer elevation above sea level, metres. None if absent.
        data: Image pixel array, dtype float32.
        header: Full astropy FITS header object.
    """

    filepath: Path
    obs_time: datetime
    ra_center: float | None
    dec_center: float | None
    width_px: int
    height_px: int
    pixscale_arcsec: float | None
    exptime_sec: float | None
    sitelat: float | None
    sitelong: float | None
    siteelev: float | None
    data: np.ndarray
    header: fits.Header


def _parse_obs_time(date_obs: str) -> datetime:
    """Parse DATE-OBS into a UTC-aware datetime.

    Handles both ISO 8601 T-separated and space-separated variants:
      - ``2024-04-02T02:55:24.38``
      - ``2024-04-02 02:55:24``

    Args:
        date_obs: Raw DATE-OBS header string.

    Returns:
        UTC-aware datetime.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    normalised = date_obs.strip().replace(" ", "T")
    try:
        dt = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse DATE-OBS '{date_obs}' as ISO 8601: {exc}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _pixscale_from_header(header: fits.Header) -> float | None:
    """Derive pixel scale in arcsec/pixel from common header keywords.

    Checks PIXSCALE first, then |CDELT1|, then computes from PIXSZ + FOCAL
    (the MILAN/Stellina convention: pixel size in µm and focal length in mm).

    Args:
        header: Full FITS header.

    Returns:
        Pixel scale in arcsec/pixel, or None if unavailable.
    """
    if "PIXSCALE" in header:
        return float(header["PIXSCALE"])
    if "CDELT1" in header:
        return abs(float(header["CDELT1"])) * 3600.0
    # MILAN/Stellina: PIXSZ in µm, FOCAL in mm → arcsec/px = 206265 * µm / (focal_mm * 1000)
    if "PIXSZ" in header and "FOCAL" in header:
        pixsz_um = float(header["PIXSZ"])
        focal_mm = float(header["FOCAL"])
        return 206265.0 * (pixsz_um / 1000.0) / focal_mm
    return None


def _exptime_from_header(header: fits.Header) -> float | None:
    """Extract exposure time in seconds from FITS header.

    Checks EXPTIME (seconds) first, then EXPOSURE (milliseconds, MILAN/Stellina).

    Args:
        header: Full FITS header.

    Returns:
        Exposure time in seconds, or None if unavailable.
    """
    if "EXPTIME" in header:
        return float(header["EXPTIME"])
    if "EXPOSURE" in header:
        return float(header["EXPOSURE"]) / 1000.0
    return None


def parse_fits(path: Path) -> FITSImage:
    """Parse a FITS file into a FITSImage dataclass.

    Opens the primary HDU, validates required fields, extracts all
    metadata, and normalises image data to float32.

    Args:
        path: Path to the .fits or .fit file.

    Returns:
        Populated FITSImage instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If DATE-OBS, NAXIS1, or NAXIS2 are missing or
            unparseable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")

    with fits.open(path) as hdul:
        header = hdul[0].header

        # --- Required fields ---
        if "DATE-OBS" not in header:
            raise ValueError(
                f"Required header keyword DATE-OBS is missing in {path.name}"
            )
        obs_time = _parse_obs_time(str(header["DATE-OBS"]))

        for kw in ("NAXIS1", "NAXIS2"):
            if kw not in header:
                raise ValueError(
                    f"Required header keyword {kw} is missing in {path.name}"
                )
        width_px = int(header["NAXIS1"])
        height_px = int(header["NAXIS2"])

        # --- Optional fields — log warnings when absent ---
        def _get_optional(key: str, cast=float) -> float | None:
            val = header.get(key)
            if val is None:
                logger.warning("Optional header keyword %s absent in %s", key, path.name)
                return None
            return cast(val)

        ra_center = _get_optional("CRVAL1")
        dec_center = _get_optional("CRVAL2")
        pixscale_arcsec = _pixscale_from_header(header)
        if pixscale_arcsec is None:
            logger.warning(
                "Pixel scale (PIXSCALE/CDELT1/PIXSZ+FOCAL) absent in %s", path.name
            )
        exptime_sec = _exptime_from_header(header)
        if exptime_sec is None:
            logger.warning("Exposure time (EXPTIME/EXPOSURE) absent in %s", path.name)
        sitelat = _get_optional("SITELAT")
        sitelong = _get_optional("SITELONG")
        siteelev = _get_optional("SITEELEV")

        # --- Image data ---
        raw = hdul[0].data
        if raw is None:
            raise ValueError(f"Primary HDU in {path.name} contains no image data")
        data = raw.astype(np.float32)

        return FITSImage(
            filepath=path.resolve(),
            obs_time=obs_time,
            ra_center=ra_center,
            dec_center=dec_center,
            width_px=width_px,
            height_px=height_px,
            pixscale_arcsec=pixscale_arcsec,
            exptime_sec=exptime_sec,
            sitelat=sitelat,
            sitelong=sitelong,
            siteelev=siteelev,
            data=data,
            header=header.copy(),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) != 2:
        print("Usage: python fits_parser.py <path/to/image.fits>")
        sys.exit(1)

    fits_path = Path(sys.argv[1])
    image = parse_fits(fits_path)

    print(f"\n=== {image.filepath.name} ===")
    print(f"  obs_time        : {image.obs_time.isoformat()}")
    print(f"  dimensions      : {image.width_px} x {image.height_px} px")
    print(f"  ra_center       : {image.ra_center}")
    print(f"  dec_center      : {image.dec_center}")
    print(f"  pixscale_arcsec : {image.pixscale_arcsec}")
    print(f"  exptime_sec     : {image.exptime_sec}")
    print(f"  sitelat         : {image.sitelat}")
    print(f"  sitelong        : {image.sitelong}")
    print(f"  siteelev        : {image.siteelev}")
    print(f"  data shape      : {image.data.shape}")
    print(f"  data dtype      : {image.data.dtype}")

    optional_fields = {
        "ra_center": image.ra_center,
        "dec_center": image.dec_center,
        "pixscale_arcsec": image.pixscale_arcsec,
        "exptime_sec": image.exptime_sec,
        "sitelat": image.sitelat,
        "sitelong": image.sitelong,
        "siteelev": image.siteelev,
    }
    missing = [k for k, v in optional_fields.items() if v is None]
    if missing:
        print(f"\nWARNING: Missing optional fields: {', '.join(missing)}")
