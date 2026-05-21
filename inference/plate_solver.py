"""ASTAP plate-solver integration for ARGUS.

Invokes the ASTAP binary via subprocess to derive WCS from a FITS image when
the image header lacks native WCS keywords.  A pointing hint (RA, DEC) and
field-of-view estimate derived from the FITS header narrows the search radius,
turning a blind all-sky solve into a sub-second constrained solve.

Configuration
-------------
ASTAP_BIN          Absolute path to the astap executable.  If unset, common
                   installation paths are tried automatically.  Set explicitly
                   when ASTAP is installed in a non-standard location.

    macOS:  export ASTAP_BIN=/Applications/ASTAP.app/Contents/MacOS/astap
    Linux:  export ASTAP_BIN=/usr/local/bin/astap

ASTAP_CATALOG_DIR  Directory containing the H18 or G18 star catalog.  Only
                   needed when the catalog is not co-located with the binary.

ASTAP_TIMEOUT      Subprocess timeout in seconds (default: 60).
ASTAP_DOWNSAMPLE   Downsample factor for speed; 1=full resolution, 2=half
                   (default: 2).  Larger values trade accuracy for speed.

ASTAP and a star catalog (H18 or G18) must be installed separately.
See https://www.hnsky.org/astap.htm
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS

logger = logging.getLogger(__name__)

_DEFAULT_SEARCH_RADIUS_DEG = 5.0
_HINT_SEARCH_RADIUS_DEG = 2.0   # tighter when RA/DEC hint is available
_DEFAULT_DOWNSAMPLE = 2
_DEFAULT_TIMEOUT_S = 60

# Common installation paths checked when ASTAP_BIN is not set
_COMMON_PATHS = [
    "/Applications/ASTAP.app/Contents/MacOS/astap",  # macOS
    "/usr/local/bin/astap",                           # Linux
    "/usr/bin/astap",
]


def _find_astap() -> str | None:
    """Return the ASTAP binary path or None if not found.

    Checks ASTAP_BIN env var first, then common install locations, then PATH.
    """
    env = os.environ.get("ASTAP_BIN", "").strip()
    if env:
        return env if Path(env).exists() else None
    for candidate in _COMMON_PATHS:
        if Path(candidate).exists():
            return candidate
    return shutil.which("astap")


def _fov_from_header(header: Any) -> float | None:
    """Estimate the horizontal field of view in degrees from FITS keywords.

    Uses FOCALLEN (mm), XPIXSZ (µm), NAXIS1 (pixels), and the small-angle
    approximation: plate_scale = pixel_size / focal_length × 206265 arcsec/px.

    Args:
        header: astropy FITS Header or dict-like.

    Returns:
        Horizontal FOV in degrees, or None when required keywords are absent.
    """
    try:
        focallen = float(header.get("FOCALLEN") or 0)
        xpixsz = float(header.get("XPIXSZ") or 0)
        naxis1 = int(header.get("NAXIS1") or 0)
        if focallen > 0 and xpixsz > 0 and naxis1 > 0:
            plate_scale_arcsec = (xpixsz * 1e-3 / focallen) * 206265.0
            return naxis1 * plate_scale_arcsec / 3600.0
    except (TypeError, ValueError):
        pass
    return None


def solve(
    fits_path: Path,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    fov_deg: float | None = None,
    search_radius_deg: float = _DEFAULT_SEARCH_RADIUS_DEG,
    downsample: int = _DEFAULT_DOWNSAMPLE,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> WCS | None:
    """Run ASTAP on *fits_path* and return the resulting WCS.

    ASTAP writes a ``.wcs`` sidecar alongside *fits_path* on a successful
    solve; this file is read and then deleted before returning so the caller's
    working directory stays clean.  A ``.ini`` results file is also removed.

    Args:
        fits_path: Path to the FITS image to solve.
        ra_deg: RA hint in degrees (field centre).  When provided together with
            *dec_deg*, the search radius is tightened to
            ``_HINT_SEARCH_RADIUS_DEG`` unless *search_radius_deg* is given
            explicitly.
        dec_deg: Dec hint in degrees.
        fov_deg: Horizontal field-of-view hint in degrees.  Omit to let ASTAP
            auto-detect from the image dimensions.
        search_radius_deg: Sky search radius passed to ASTAP (``-r``).
        downsample: Image downsample factor (``-z``); 2 is a good default for
            large sensor images.
        timeout: Subprocess wall-clock timeout in seconds.

    Returns:
        astropy WCS on a successful solve, None otherwise.
    """
    bin_path = _find_astap()
    if not bin_path:
        logger.debug("ASTAP not found — skipping plate solve")
        return None

    cmd = [bin_path, "-f", str(fits_path)]

    if ra_deg is not None:
        cmd += ["-ra", f"{ra_deg / 15.0:.6f}"]   # ASTAP uses decimal hours
    if dec_deg is not None:
        cmd += ["-spd", f"{90.0 + dec_deg:.6f}"]  # south polar distance
    if fov_deg is not None:
        cmd += ["-fov", f"{fov_deg:.4f}"]

    # Use tighter radius when we have a pointing hint
    if ra_deg is not None and dec_deg is not None and search_radius_deg == _DEFAULT_SEARCH_RADIUS_DEG:
        search_radius_deg = _HINT_SEARCH_RADIUS_DEG
    cmd += ["-r", f"{search_radius_deg:.2f}"]
    cmd += ["-z", str(downsample)]

    catalog_dir = os.environ.get("ASTAP_CATALOG_DIR", "").strip()
    if catalog_dir:
        cmd += ["-d", catalog_dir]

    logger.debug("ASTAP: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("ASTAP_TIMEOUT", timeout)),
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "ASTAP timed out after %ds on %s", timeout, fits_path.name
        )
        return None
    except Exception as exc:
        logger.warning("ASTAP subprocess failed: %s", exc)
        return None

    wcs_path = fits_path.with_suffix(".wcs")
    ini_path = fits_path.with_suffix(".ini")

    if not wcs_path.exists():
        logger.debug(
            "ASTAP produced no .wcs for %s (exit=%d); solve failed or no stars",
            fits_path.name,
            proc.returncode,
        )
        return None

    try:
        header = fits.Header.fromtextfile(str(wcs_path))
    except Exception as exc:
        logger.warning(
            "Could not read ASTAP .wcs output %s: %s", wcs_path.name, exc
        )
        return None
    finally:
        wcs_path.unlink(missing_ok=True)
        ini_path.unlink(missing_ok=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(header)
        if wcs.naxis == 0 or not wcs.has_celestial:
            logger.debug(
                "ASTAP .wcs for %s contains no celestial axes", fits_path.name
            )
            return None
        logger.info("Plate solve succeeded for %s", fits_path.name)
        return wcs
    except Exception as exc:
        logger.warning(
            "Could not construct WCS from ASTAP output for %s: %s",
            fits_path.name,
            exc,
        )
        return None


def solve_from_header(fits_path: Path, header: Any) -> WCS | None:
    """Plate-solve *fits_path* using pointing hints extracted from its header.

    Reads ``RA``, ``DEC``, ``FOCALLEN``, ``XPIXSZ``, and ``NAXIS1`` from
    *header* and passes them to :func:`solve` as hints.  When ``RA``/``DEC``
    are present the search radius is automatically tightened to
    ``_HINT_SEARCH_RADIUS_DEG`` degrees.

    Args:
        fits_path: Path to the FITS image.
        header: astropy FITS Header (or any dict-like) from that image.

    Returns:
        astropy WCS on success, None if ASTAP is unavailable or solve fails.
    """
    def _flt(key: str) -> float | None:
        val = header.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return solve(
        fits_path,
        ra_deg=_flt("RA"),
        dec_deg=_flt("DEC"),
        fov_deg=_fov_from_header(header),
    )


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m inference.plate_solver <path/to/image.fits>")
        sys.exit(1)

    p = Path(sys.argv[1])
    with fits.open(p) as hdul:
        hdr = hdul[0].header

    result = solve_from_header(p, hdr)
    if result is not None:
        centre = result.all_pix2world(
            [[hdr.get("NAXIS1", 512) / 2, hdr.get("NAXIS2", 512) / 2]], 0
        )
        print(f"Solved: RA={centre[0][0]:.4f}°  Dec={centre[0][1]:.4f}°")
    else:
        print("Solve failed or ASTAP not available.")
    sys.exit(0 if result is not None else 1)
