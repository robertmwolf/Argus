"""SGP4 propagation and TEME→topocentric coordinate conversion.

Propagates a single TLE to an observation epoch and returns the
predicted topocentric RA/Dec, angular velocity, and direction for
an observer at a given geodetic position.

Uses skyfield's EarthSatellite (which wraps sgp4 internally) for the
TEME→topocentric coordinate transformation in skyfield 1.49.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from skyfield.api import EarthSatellite, load, wgs84

logger = logging.getLogger(__name__)

_ts = None  # lazy-loaded timescale


def _timescale():
    """Return a cached skyfield Timescale."""
    global _ts
    if _ts is None:
        _ts = load.timescale()
    return _ts


def propagate(
    tle_dict: dict,
    obs_time: datetime,
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
) -> dict | None:
    """Propagate a single TLE to obs_time and compute topocentric position.

    # Source: ARGUS architecture — SGP4 propagation + TEME→topocentric
    # Ref: agent_docs/architecture.md

    Uses skyfield EarthSatellite for propagation and coordinate transformation.
    EarthSatellite wraps sgp4 internally and handles the TEME→topocentric
    conversion automatically.

    Args:
        tle_dict: Space-Track GP_History record containing TLE_LINE1, TLE_LINE2,
            OBJECT_NAME, NORAD_CAT_ID, and EPOCH fields.
        obs_time: UTC datetime to propagate to.
        observer_lat: Observer geodetic latitude in degrees.
        observer_lon: Observer geodetic longitude in degrees.
        observer_alt_m: Observer elevation above WGS84 ellipsoid in metres.

    Returns:
        Dict with keys:
            norad_id (int), object_name (str), tle_epoch (datetime),
            tle_age_hours (float), predicted_ra (float, deg),
            predicted_dec (float, deg),
            predicted_velocity_arcsec_s (float),
            predicted_direction_deg (float),
            predicted_magnitude (None — not yet implemented).
        Returns None if propagation fails or required TLE fields are missing.
    """
    line1 = tle_dict.get("TLE_LINE1")
    line2 = tle_dict.get("TLE_LINE2")
    name = tle_dict.get("OBJECT_NAME", "UNKNOWN")

    if not line1 or not line2:
        logger.debug("Skipping TLE with missing lines: %s", name)
        return None

    try:
        sat = EarthSatellite(line1, line2, name, _timescale())
    except Exception as exc:
        logger.debug("Failed to parse TLE for %s: %s", name, exc)
        return None

    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    ts = _timescale()
    observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_alt_m)

    try:
        t = ts.from_datetime(obs_time)
        topocentric = (sat - observer).at(t)
        ra_angle, dec_angle, _ = topocentric.radec()
        predicted_ra = ra_angle._degrees % 360.0
        predicted_dec = dec_angle._degrees
    except Exception as exc:
        logger.debug("Propagation failed for %s: %s", name, exc)
        return None

    # Estimate velocity and direction by propagating 1 second forward
    try:
        t2 = ts.from_datetime(obs_time + timedelta(seconds=1))
        topo2 = (sat - observer).at(t2)
        ra2_angle, dec2_angle, _ = topo2.radec()
        ra2 = ra2_angle._degrees % 360.0
        dec2 = dec2_angle._degrees

        dra = (ra2 - predicted_ra) * math.cos(math.radians(predicted_dec))
        ddec = dec2 - predicted_dec
        velocity_arcsec_s = math.hypot(dra, ddec) * 3600.0
        direction_deg = math.degrees(math.atan2(dra, ddec)) % 360.0
    except Exception:
        velocity_arcsec_s = 0.0
        direction_deg = 0.0

    # TLE epoch and age
    tle_epoch_str = tle_dict.get("EPOCH", "")
    try:
        tle_epoch = datetime.fromisoformat(tle_epoch_str.replace("Z", "+00:00"))
        if tle_epoch.tzinfo is None:
            tle_epoch = tle_epoch.replace(tzinfo=timezone.utc)
        tle_age_hours = (obs_time - tle_epoch).total_seconds() / 3600.0
    except (ValueError, AttributeError):
        tle_epoch = obs_time
        tle_age_hours = 0.0

    return {
        "norad_id": int(tle_dict.get("NORAD_CAT_ID", 0)),
        "object_name": name,
        "tle_epoch": tle_epoch,
        "tle_age_hours": tle_age_hours,
        "predicted_ra": predicted_ra,
        "predicted_dec": predicted_dec,
        "predicted_velocity_arcsec_s": velocity_arcsec_s,
        "predicted_direction_deg": direction_deg,
        "predicted_magnitude": None,
    }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Starlink-1007 TLE (example — not current)
    sample_tle = {
        "OBJECT_NAME": "STARLINK-1007",
        "NORAD_CAT_ID": "44713",
        "TLE_LINE1": "1 44713U 19074A   24093.50000000  .00001764  00000-0  13679-3 0  9998",
        "TLE_LINE2": "2 44713  53.0536 100.4783 0001199  86.5965 273.5324 15.06396608241234",
        "EPOCH": "2024-04-02T12:00:00",
    }
    obs = datetime(2024, 4, 2, 12, 0, 0, tzinfo=timezone.utc)
    result = propagate(sample_tle, obs, 45.0, 9.0, 200.0)
    if result:
        print(f"RA:  {result['predicted_ra']:.4f}°")
        print(f"Dec: {result['predicted_dec']:.4f}°")
        print(f"Vel: {result['predicted_velocity_arcsec_s']:.2f} arcsec/s")
        print(f"Dir: {result['predicted_direction_deg']:.2f}°")
        print(f"TLE age: {result['tle_age_hours']:.1f}h")
    else:
        print("Propagation failed.")
