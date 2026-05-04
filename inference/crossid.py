"""Satellite ephemeris cross-identification for ARGUS inference detections.

Cross-matches DINO OBB detections against Space-Track GP_History TLEs using
SGP4 propagation and Gaussian confidence scoring.  TLE data is fetched
directly from Space-Track for the observation time window and cached to disk
via src.matching.spacetrack_query.

Requires environment variables:
    SPACETRACK_USER — your Space-Track account email
    SPACETRACK_PASS — your Space-Track account password

# Source: Danarianto et al. — Gaussian confidence scoring for satellite crossID
# Ref: cite per published paper
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from src.matching.spacetrack_query import query_gp_history

logger = logging.getLogger(__name__)

# Gaussian sigma for position score: 0.25° = 900 arcsec
# Source: ARGUS architecture — scoring formulas
# Ref: agent_docs/architecture.md
_POSITION_SIGMA_ARCSEC = 900.0


# ---------------------------------------------------------------------------
# TLE loading from Space-Track
# ---------------------------------------------------------------------------

def _fetch_tle_catalog(
    obs_time: datetime,
    epoch_window_days: int,
) -> list[tuple[str, str, str]]:
    """Fetch TLEs from Space-Track GP_History for the observation window.

    Delegates to src.matching.spacetrack_query.query_gp_history which handles
    authentication, rate limiting, and disk caching.

    Args:
        obs_time: UTC observation time.
        epoch_window_days: Days before obs_time to include in the epoch search.

    Returns:
        List of (name, line1, line2) tuples ready for SGP4 propagation.
    """
    records = query_gp_history(obs_time, epoch_window_days=epoch_window_days)
    catalog: list[tuple[str, str, str]] = []
    for rec in records:
        name  = rec.get("OBJECT_NAME", "UNKNOWN")
        line1 = rec.get("TLE_LINE1", "")
        line2 = rec.get("TLE_LINE2", "")
        if line1 and line2:
            catalog.append((name, line1, line2))
    logger.debug(
        "Space-Track returned %d records; %d have TLE lines",
        len(records), len(catalog),
    )
    return catalog


# ---------------------------------------------------------------------------
# SGP4 propagation (reuses Phase 0 approach via skyfield)
# ---------------------------------------------------------------------------

_ts = None  # lazy-loaded skyfield Timescale


def _timescale():
    """Return a cached skyfield Timescale (expensive to create repeatedly)."""
    global _ts
    if _ts is None:
        from skyfield.api import load  # type: ignore[import]
        _ts = load.timescale()
    return _ts


def _propagate_to_radec(
    name: str,
    line1: str,
    line2: str,
    obs_time: datetime,
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
) -> dict[str, Any] | None:
    """Propagate one TLE to obs_time and return topocentric RA/Dec.

    Reuses the same skyfield EarthSatellite approach as src/matching/propagator.py.

    # Source: ARGUS architecture — SGP4 propagation + TEME→topocentric
    # Ref: agent_docs/architecture.md

    Args:
        name: Satellite name (for logging).
        line1: TLE line 1.
        line2: TLE line 2.
        obs_time: UTC observation datetime.
        observer_lat: Observer geodetic latitude in degrees.
        observer_lon: Observer geodetic longitude in degrees.
        observer_alt_m: Observer elevation above WGS84 in metres.

    Returns:
        Dict with keys: object_name, norad_id (int), predicted_ra (deg),
        predicted_dec (deg), tle_age_hours.  None on failure.
    """
    try:
        from skyfield.api import EarthSatellite, wgs84  # type: ignore[import]
    except ImportError:  # pragma: no cover
        logger.error("skyfield not installed; cannot propagate TLEs")
        return None

    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    try:
        sat = EarthSatellite(line1, line2, name, _timescale())
    except Exception as exc:
        logger.debug("TLE parse failed for %s: %s", name, exc)
        return None

    # Extract NORAD ID from line 2, column 2-6
    try:
        norad_id = int(line2[2:7].strip())
    except ValueError:
        norad_id = 0

    # TLE epoch → age
    try:
        # line1 columns 18-32 encode epoch as YYDDD.fraction
        epoch_str = line1[18:32].strip()
        yr2 = int(epoch_str[:2])
        year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
        day_frac = float(epoch_str[2:])
        tle_epoch = datetime(year, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta
        tle_epoch += timedelta(days=day_frac - 1)
        tle_age_hours = (obs_time - tle_epoch).total_seconds() / 3600.0
    except Exception:
        tle_age_hours = 0.0

    try:
        ts = _timescale()
        observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_alt_m)
        t = ts.from_datetime(obs_time)
        topocentric = (sat - observer).at(t)
        ra_angle, dec_angle, _ = topocentric.radec()
        predicted_ra = float(ra_angle._degrees) % 360.0
        predicted_dec = float(dec_angle._degrees)
    except Exception as exc:
        logger.debug("Propagation failed for %s: %s", name, exc)
        return None

    return {
        "object_name": name,
        "norad_id": norad_id,
        "predicted_ra": predicted_ra,
        "predicted_dec": predicted_dec,
        "tle_age_hours": tle_age_hours,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Compute the angular separation between two sky positions in arcseconds.

    Uses the haversine formula for numerical stability at small angles.

    Args:
        ra1, dec1: First position in degrees.
        ra2, dec2: Second position in degrees.

    Returns:
        Angular separation in arcseconds.
    """
    ra1_r  = math.radians(ra1)
    dec1_r = math.radians(dec1)
    ra2_r  = math.radians(ra2)
    dec2_r = math.radians(dec2)

    dra  = ra2_r - ra1_r
    ddec = dec2_r - dec1_r

    a = (math.sin(ddec / 2) ** 2
         + math.cos(dec1_r) * math.cos(dec2_r) * math.sin(dra / 2) ** 2)
    sep_rad = 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
    return math.degrees(sep_rad) * 3600.0


def _gaussian_score(delta: float, sigma: float) -> float:
    """Score from 0–1 using Gaussian falloff from zero error.

    # Source: ARGUS architecture — Gaussian scoring formulas
    # Ref: agent_docs/architecture.md

    Args:
        delta: Absolute error value.
        sigma: Characteristic scale (score ≈ 0.61 at delta == sigma).

    Returns:
        Score in [0.0, 1.0].
    """
    return math.exp(-0.5 * (delta / sigma) ** 2)


# ---------------------------------------------------------------------------
# Main cross-identification entry point
# ---------------------------------------------------------------------------

def cross_identify(
    detections: list[dict],
    obs_time: datetime,
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
    epoch_window_days: int = 3,
) -> list[dict]:
    """Cross-match detections against Space-Track GP_History TLEs.

    Fetches TLEs for the epoch window ending at obs_time from Space-Track
    (cached to disk), propagates each to obs_time via SGP4, and scores
    candidates using Gaussian position confidence.

    Each detection dict is mutated in-place: an 'identifications' key is
    added containing up to 3 ranked candidate dicts.

    # Source: Danarianto et al. — Gaussian confidence scoring for satellite crossID
    # Ref: cite per published paper

    Args:
        detections: List of detection dicts from the inference pipeline.
            Each dict should have 'ra_deg' and 'dec_deg' keys (may be None).
        obs_time: UTC observation time from the FITS header.
        observer_lat: Observer geodetic latitude in degrees.
        observer_lon: Observer geodetic longitude in degrees.
        observer_alt_m: Observer elevation above WGS84 in metres.
        epoch_window_days: Days before obs_time to search for TLE epochs.
            Passed to Space-Track GP_History; default 3 days.

    Returns:
        The mutated *detections* list (same objects, with 'identifications' added).
    """
    catalog = _fetch_tle_catalog(obs_time, epoch_window_days)
    if not catalog:
        logger.warning("Empty TLE catalog — all identifications will be empty")
        for det in detections:
            det.setdefault("identifications", [])
        return detections

    logger.debug("Cross-identifying %d detections against %d TLEs", len(detections), len(catalog))

    for det in detections:
        obs_ra  = det.get("ra_deg")
        obs_dec = det.get("dec_deg")

        if obs_ra is None or obs_dec is None:
            logger.debug("Detection missing sky coords — skipping cross-ID")
            det["identifications"] = []
            continue

        candidates: list[dict] = []
        for name, line1, line2 in catalog:
            result = _propagate_to_radec(
                name, line1, line2, obs_time,
                observer_lat, observer_lon, observer_alt_m,
            )
            if result is None:
                continue

            sep_arcsec = _angular_separation_arcsec(
                obs_ra, obs_dec,
                result["predicted_ra"], result["predicted_dec"],
            )
            score = _gaussian_score(sep_arcsec, sigma=_POSITION_SIGMA_ARCSEC)

            candidates.append({
                "satellite_name": result["object_name"],
                "norad_id":       result["norad_id"],
                "confidence":     score,
                "separation_arcsec": sep_arcsec,
                "rank":           0,  # filled below
            })

        # Sort descending by confidence, assign ranks 1–3
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        top3 = candidates[:3]
        for rank, cand in enumerate(top3, start=1):
            cand["rank"] = rank

        det["identifications"] = top3
        if top3:
            logger.debug(
                "Best match: %s (NORAD %d) sep=%.1f\" conf=%.3f",
                top3[0]["satellite_name"], top3[0]["norad_id"],
                top3[0]["separation_arcsec"], top3[0]["confidence"],
            )

    return detections


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    # Smoke-test: cross-ID a single detection at a known position
    # Requires SPACETRACK_USER and SPACETRACK_PASS to be set in the environment.
    obs = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)
    dets = [
        {"ra_deg": 83.82, "dec_deg": -5.39, "confidence": 0.9},
        {"ra_deg": None,  "dec_deg": None,   "confidence": 0.5},  # no sky coords
    ]

    result = cross_identify(
        dets, obs,
        observer_lat=49.61, observer_lon=6.13, observer_alt_m=280.0,
        epoch_window_days=3,
    )

    for i, d in enumerate(result):
        ids = d.get("identifications", [])
        print(f"\nDetection {i}: ra={d.get('ra_deg')} dec={d.get('dec_deg')}")
        if ids:
            for c in ids:
                print(f"  rank {c['rank']}: {c['satellite_name']} "
                      f"(NORAD {c['norad_id']}) sep={c['separation_arcsec']:.1f}\" "
                      f"conf={c['confidence']:.3f}")
        else:
            print("  (no identifications)")
    sys.exit(0)
