"""Spatial pre-screen: filter TLEs whose predicted position falls inside the FOV.

This is a lightweight pass that reduces 5–15k TLE candidates down to
5–50 before running full propagation in the matcher.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from src.matching.propagator import propagate

logger = logging.getLogger(__name__)


def _angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle angular separation in degrees between two sky positions.

    Args:
        ra1, dec1: First position in degrees.
        ra2, dec2: Second position in degrees.

    Returns:
        Separation in degrees.
    """
    r1 = math.radians(ra1)
    d1 = math.radians(dec1)
    r2 = math.radians(ra2)
    d2 = math.radians(dec2)
    cos_sep = (
        math.sin(d1) * math.sin(d2)
        + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    )
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def filter_by_fov(
    tle_dicts: list[dict],
    obs_time: datetime,
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
    ra_center: float,
    dec_center: float,
    fov_radius_deg: float,
) -> list[dict]:
    """Filter TLEs to those whose predicted position falls within the FOV.

    # Source: ARGUS architecture — spatial cone pre-screen
    # Ref: agent_docs/architecture.md

    Runs lightweight SGP4 propagation on each TLE and discards objects
    whose predicted topocentric position is outside the specified FOV cone.
    TLEs with missing or unparseable lines are silently skipped.

    Args:
        tle_dicts: List of Space-Track GP_History dicts.
        obs_time: UTC observation time.
        observer_lat: Observer geodetic latitude in degrees.
        observer_lon: Observer geodetic longitude in degrees.
        observer_alt_m: Observer elevation in metres.
        ra_center: FOV centre right ascension in degrees.
        dec_center: FOV centre declination in degrees.
        fov_radius_deg: Half-cone radius in degrees.

    Returns:
        Subset of tle_dicts whose predicted position is within fov_radius_deg
        of (ra_center, dec_center).  Each surviving dict has an extra key
        ``_propagated`` containing the result dict from :func:`propagate`
        so the matcher can reuse it without re-propagating.
    """
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    survivors: list[dict] = []
    skipped = 0

    for tle in tle_dicts:
        result = propagate(tle, obs_time, observer_lat, observer_lon, observer_alt_m)
        if result is None:
            skipped += 1
            continue

        sep = _angular_separation(
            result["predicted_ra"],
            result["predicted_dec"],
            ra_center,
            dec_center,
        )
        if sep <= fov_radius_deg:
            survivors.append({**tle, "_propagated": result})

    logger.info(
        "Spatial filter: %d/%d TLEs in FOV (%.1f° radius), %d skipped",
        len(survivors),
        len(tle_dicts),
        fov_radius_deg,
        skipped,
    )
    return survivors


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    print("spatial_filter.py: no standalone demo without Space-Track data.")
    print("Import filter_by_fov and pass TLE dicts from spacetrack_query.")
