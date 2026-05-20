"""Generate .strk stub files for a BrentImages observation night.

BrentImages FITS files embed NORAD ID, TLE elements, site info, and observation
time in their headers (written by SkyTrack 1.9.8 during capture).  This script
harvests those headers to produce one .strk file per unique NORAD ID, matching
the format produced by SkyTrack and consumed by convert_gtimages.py.

All observations are written with Reject=2 ("needs annotation") so that
convert_gtimages.py skips them until pixel coordinates are filled in by the
manual annotation workflow.  Once annotated, flip the reject flag to 0 (streak
present) or -1 (no streak found).

Usage::

    python scripts/generate_brentimages_strk.py \\
        --night-dir /Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood

    # Overwrite existing .strk files:
    python scripts/generate_brentimages_strk.py \\
        --night-dir /Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood \\
        --force
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from collections import defaultdict
from datetime import datetime, timezone

from astropy.io import fits
from astropy.time import Time

logger = logging.getLogger(__name__)

# Reject code for observations that have not yet been manually annotated.
_REJECT_PENDING = "2"
_PENDING_COMMENT = "Needs annotation"

# Placeholder TLE fields absent from FITS headers.
_BSTAR_DEFAULT = " 00000-0"
_MM1_DEFAULT = " .00000000"
_MM2_DEFAULT = " 00000-0"
_EPHEM_TYPE = "0"
_ELSET_DEFAULT = "999"
_REV_DEFAULT = "0"
_OBJECT_CLASS = "U"


def _read_header(fits_path: pathlib.Path) -> dict:
    """Return the primary HDU header as a plain dict.

    Args:
        fits_path: Path to a FITS file.

    Returns:
        Dict of header key → value.
    """
    with fits.open(fits_path, memmap=True) as hdul:
        hdul.verify("silentfix")
        return dict(hdul[0].header)


def _jd_from_date_obs(date_obs: str) -> float:
    """Convert a FITS DATE-OBS string to Julian Date.

    Args:
        date_obs: ISO 8601 string, e.g. '2026-05-16T03:25:18.026'.

    Returns:
        Julian Date as float.
    """
    return Time(date_obs, format="isot", scale="utc").jd


def _ecc_to_tle_str(gpecc_int: int) -> str:
    """Format eccentricity header integer as 7-digit TLE string.

    SkyTrack stores eccentricity as a 7-digit integer without the leading '0.'
    For example, 0.0080322 → header value 80322 → TLE string '0080322'.

    Args:
        gpecc_int: Integer value from the GPECC FITS header key.

    Returns:
        Zero-padded 7-character eccentricity string.
    """
    return f"{int(gpecc_int):07d}"


def _build_strk(
    norad_id: int,
    observations: list[dict],
    site_lat: float,
    site_lon: float,
    site_elev: float,
) -> str:
    """Render a .strk file string for one NORAD ID.

    Args:
        norad_id: NORAD catalog number.
        observations: List of per-frame header dicts, sorted by DATE-OBS.
        site_lat: Observatory latitude in decimal degrees.
        site_lon: Observatory longitude in decimal degrees (East positive).
        site_elev: Observatory elevation in metres.

    Returns:
        Complete .strk file content as a string.
    """
    # Pull TLE fields from the first frame's header (same satellite, same TLE).
    h0 = observations[0]["header"]
    sat_name = str(h0.get("OBJECT", f"NORAD {norad_id}")).strip()
    epoch_yr = int(h0.get("GPEPOCYR", 0))
    epoch_da = float(h0.get("GPEPOCDA", 0.0))
    incl = float(h0.get("GPINCL", 0.0))
    raan = float(h0.get("GPRAAN", 0.0))
    ecc_str = _ecc_to_tle_str(h0.get("GPECC", 0))
    argp = float(h0.get("GPAARGP", 0.0))
    ma = float(h0.get("GPMA", 0.0))
    mm = float(h0.get("GPMM", 0.0))

    tle_row = "\t".join([
        str(norad_id),
        str(epoch_yr),
        f"{epoch_da:.8f}",
        f"{incl:8.4f}",
        f"{raan:8.4f}",
        ecc_str,
        f"{argp:8.4f}",
        f"{ma:8.4f}",
        f"{mm:.8f}",
        _BSTAR_DEFAULT,
        _MM1_DEFAULT,
        _MM2_DEFAULT,
        _EPHEM_TYPE,
        f"{_ELSET_DEFAULT:>4}",
        _REV_DEFAULT,
        f"{sat_name:<24}",
        "",          # Object ID — not available in headers
        _OBJECT_CLASS,
    ])

    obs_lines: list[str] = []
    for obs in observations:
        h = obs["header"]
        fname = obs["filename"]
        date_obs = str(h.get("DATE-OBS", ""))
        try:
            jd = _jd_from_date_obs(date_obs)
        except Exception:
            jd = 0.0

        exposure = float(h.get("EXPOSURE", 0.5))
        gain = float(h.get("GAIN", 0))

        obs_lines.append("\t".join([
            fname,
            date_obs,
            f"{jd:.10f}",
            "0", "0",       # Start X, Y
            "0", "0",       # End X, Y
            "0", "0",       # Mid X, Y
            "0",            # Peak SNR
            "0",            # Mean SNR
            "0",            # Elongation
            "0",            # Length
            _REJECT_PENDING,
            "0", "0",       # Mid RA, Dec
            "0", "0",       # Start RA, Dec
            "0", "0",       # End RA, Dec
            "0", "0",       # Expected RA, Dec
            "0",            # Expected Range
            f"{exposure:.4f}",
            f"{gain:.1f}",
            _PENDING_COMMENT,
        ]))

    lines = [
        "[VERSION]",
        "SkyTrack \t1.9.8",
        "[SITE]",
        "\tLatitude(deg)\tLongitude(deg, East=+)\tElevation(m)",
        f"{site_lat}\t{site_lon}\t{site_elev}",
        "[TLE]",
        "NORAD\tEpochYear\tEpochDay\tIncl\tRAAN\tECC\tARGP\tMA\tMM\t"
        "BSTAR\tMM1\tMM2\tEphemType\tElset\tRev\tName\tObject ID\tClass",
        tle_row,
        "[OBS]",
        "Image\tDate Time(UTC)\tJD Midpoint\tStart X Pixel\tStart Y Pixel\t"
        "End X Pixel\tEnd Y Pixel\tMid X Pixel\tMid Y Pixel\tPeak SNR\t"
        "Mean SNR\tElongation\tLength\tReject\tMid RA\tMid Dec\t"
        "Start RA\tStart Dec\tEnd RA\tEnd Dec\t"
        "Expected RA\tExpected Dec\tExpected Range\tExposure\tGain\tComment",
    ] + obs_lines

    return "\n".join(lines) + "\n"


def generate(night_dir: pathlib.Path, force: bool = False) -> None:
    """Generate .strk stub files for all NORAD IDs in a BrentImages night.

    Args:
        night_dir: Directory containing Streak_NORADID_HHMMSS.fits files.
        force: If True, overwrite existing .strk files.
    """
    fits_files = sorted(night_dir.glob("Streak_*.fits"))
    if not fits_files:
        raise FileNotFoundError(f"No Streak_*.fits files found in {night_dir}")

    logger.info("Scanning %d FITS files in %s", len(fits_files), night_dir)

    # Group files by NORAD ID extracted from header (authoritative over filename).
    by_norad: dict[int, list[dict]] = defaultdict(list)
    site_info: tuple[float, float, float] | None = None

    for fpath in fits_files:
        try:
            h = _read_header(fpath)
        except Exception as exc:
            logger.warning("Cannot read %s: %s", fpath.name, exc)
            continue

        norad_id = h.get("NORADID")
        if norad_id is None:
            logger.warning("No NORADID in header of %s — skipping", fpath.name)
            continue

        norad_id = int(norad_id)
        by_norad[norad_id].append({"filename": fpath.name, "header": h})

        if site_info is None:
            lat = float(h.get("SITELAT", 0.0))
            lon = float(h.get("SITELONG", 0.0))
            elev = float(h.get("SITEELEV", 0.0))
            site_info = (lat, lon, elev)

    if site_info is None:
        raise ValueError("Could not determine site info from any FITS header")

    lat, lon, elev = site_info
    written = skipped = 0

    for norad_id, obs_list in sorted(by_norad.items()):
        # Sort observations chronologically.
        obs_list.sort(key=lambda o: o["header"].get("DATE-OBS", ""))

        out_path = night_dir / f"{norad_id}.strk"
        if out_path.exists() and not force:
            logger.info("Skipping %s (already exists; use --force to overwrite)", out_path.name)
            skipped += 1
            continue

        content = _build_strk(norad_id, obs_list, lat, lon, elev)
        out_path.write_text(content, encoding="utf-8")
        logger.info(
            "Written %s (%d frames, satellite: %s)",
            out_path.name,
            len(obs_list),
            obs_list[0]["header"].get("OBJECT", "?"),
        )
        written += 1

    logger.info(
        "Done. Written: %d .strk files, skipped: %d. "
        "All observations marked Reject=%s — annotate pixel coords before training.",
        written,
        skipped,
        _REJECT_PENDING,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate .strk stubs for a BrentImages observation night"
    )
    parser.add_argument(
        "--night-dir",
        required=True,
        type=pathlib.Path,
        help="Directory containing Streak_NORADID_HHMMSS.fits files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .strk files",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    generate(args.night_dir, force=args.force)


if __name__ == "__main__":
    main()
