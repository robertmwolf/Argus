"""SatChecker candidate provider for ARGUS cross-identification.

This module is an optional prototype path for obtaining candidate TLEs from
the IAU CPS SatChecker FOV API instead of ARGUS's local ``tle_catalog``.  It
returns catalog entries in the same shape consumed by ``inference.crossid`` so
the existing SGP4 propagation and scoring logic remains the source of truth.

SatChecker API docs:
https://satchecker.readthedocs.io/en/latest/fov.html
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://satchecker.cps.iau.org"
_FOV_ENDPOINT = "/fov/satellite-passes/"
_DEFAULT_TIMEOUT_S = 8.0
_DEFAULT_MIN_RADIUS_DEG = 0.5
_DEFAULT_MARGIN_DEG = 0.25
_DEFAULT_MAX_RADIUS_DEG = 8.0
_DEFAULT_DURATION_S = 2.0


def _env_float(name: str, default: float) -> float:
    """Return an environment variable parsed as float, or *default*."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _to_julian_date(dt: datetime) -> float:
    """Convert a UTC datetime to Julian Date."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp() / 86400.0 + 2440587.5


def _valid_number(value: Any) -> bool:
    """Return True when *value* is a finite numeric value."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _angular_separation_deg(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
) -> float:
    """Return angular separation between two sky positions in degrees."""
    ra1_r = math.radians(ra1)
    dec1_r = math.radians(dec1)
    ra2_r = math.radians(ra2)
    dec2_r = math.radians(dec2)
    dra = ra2_r - ra1_r
    ddec = dec2_r - dec1_r
    a = (
        math.sin(ddec / 2.0) ** 2
        + math.cos(dec1_r) * math.cos(dec2_r) * math.sin(dra / 2.0) ** 2
    )
    return math.degrees(2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a)))))


def _detection_points(detections: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Extract all finite RA/Dec points from detection tips and midpoints."""
    points: list[tuple[float, float]] = []
    for det in detections:
        for ra_key, dec_key in (
            ("ra_tip1_deg", "dec_tip1_deg"),
            ("ra_tip2_deg", "dec_tip2_deg"),
            ("ra_center_deg", "dec_center_deg"),
        ):
            ra = det.get(ra_key)
            dec = det.get(dec_key)
            if _valid_number(ra) and _valid_number(dec):
                points.append((float(ra) % 360.0, float(dec)))
    return points


def derive_fov_from_detections(
    detections: list[dict[str, Any]],
    *,
    min_radius_deg: float | None = None,
    margin_deg: float | None = None,
    max_radius_deg: float | None = None,
) -> tuple[float, float, float] | None:
    """Derive a circular FOV around the detections for SatChecker.

    Args:
        detections: ARGUS detection dicts with RA/Dec tip coordinates.
        min_radius_deg: Minimum query radius. Defaults to
            ``ARGUS_SATCHECKER_MIN_RADIUS_DEG`` or 0.5 deg.
        margin_deg: Extra angular padding beyond the detected endpoints.
            Defaults to ``ARGUS_SATCHECKER_MARGIN_DEG`` or 0.25 deg.
        max_radius_deg: Hard cap to avoid accidental expensive sky-wide FOV
            queries. Defaults to ``ARGUS_SATCHECKER_MAX_RADIUS_DEG`` or 8 deg.

    Returns:
        ``(ra_deg, dec_deg, radius_deg)`` or None when detections have no sky
        coordinates.
    """
    points = _detection_points(detections)
    if not points:
        return None

    min_radius = (
        min_radius_deg
        if min_radius_deg is not None
        else _env_float("ARGUS_SATCHECKER_MIN_RADIUS_DEG", _DEFAULT_MIN_RADIUS_DEG)
    )
    margin = (
        margin_deg
        if margin_deg is not None
        else _env_float("ARGUS_SATCHECKER_MARGIN_DEG", _DEFAULT_MARGIN_DEG)
    )
    max_radius = (
        max_radius_deg
        if max_radius_deg is not None
        else _env_float("ARGUS_SATCHECKER_MAX_RADIUS_DEG", _DEFAULT_MAX_RADIUS_DEG)
    )

    x = sum(math.cos(math.radians(ra)) for ra, _ in points)
    y = sum(math.sin(math.radians(ra)) for ra, _ in points)
    center_ra = math.degrees(math.atan2(y, x)) % 360.0 if x or y else points[0][0]
    if math.isclose(center_ra, 360.0, abs_tol=1e-10):
        center_ra = 0.0
    center_dec = sum(dec for _, dec in points) / len(points)
    radius = max(
        _angular_separation_deg(center_ra, center_dec, ra, dec)
        for ra, dec in points
    )
    radius = min(max(radius + margin, min_radius), max_radius)
    return center_ra, center_dec, radius


def _first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value found in *record* for *keys*."""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _catalog_entry_from_tle(record: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any] | None:
    """Map a SatChecker TLE-like record into a cross-ID catalog entry."""
    line1 = _first_value(record, ("tle_line1", "TLE_LINE1", "line1"))
    line2 = _first_value(record, ("tle_line2", "TLE_LINE2", "line2"))
    if not line1 or not line2:
        return None

    name = _first_value(record, ("satellite_name", "name", "OBJECT_NAME"))
    norad_id = _first_value(record, ("satellite_id", "norad_id", "NORAD_CAT_ID"))
    epoch = _first_value(record, ("epoch", "tle_epoch", "EPOCH"))
    source = _first_value(record, ("data_source", "source")) or "satchecker"

    return {
        "name": name or fallback.get("name") or f"NORAD-{norad_id}",
        "line1": str(line1).strip(),
        "line2": str(line2).strip(),
        "tle_epoch": epoch,
        "source": f"satchecker:{source}",
        "tle_search_mode": "satchecker_fov",
        "tle_data_fresh_at": None,
    }


def _iter_tle_records(satellite: dict[str, Any]) -> list[dict[str, Any]]:
    """Return likely TLE payloads from a SatChecker satellite response."""
    for key in ("tle_data", "tles", "tle"):
        value = satellite.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        if isinstance(value, dict):
            return [value]
    return []


def catalog_from_satchecker_response(payload: Any) -> list[dict[str, Any]]:
    """Parse SatChecker FOV response JSON into cross-ID catalog entries.

    The public docs show response examples, but included TLE placement may vary
    between synchronous and task-status responses.  This parser accepts the
    documented ``data.satellites`` shape and tolerates TLE dictionaries either
    on the satellite object or beside individual position rows.
    """
    responses = payload if isinstance(payload, list) else [payload]
    catalog: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for response in responses:
        if not isinstance(response, dict):
            continue
        data = response.get("data")
        if not isinstance(data, dict):
            continue
        satellites = data.get("satellites")
        if not isinstance(satellites, dict):
            continue

        for label, satellite in satellites.items():
            if not isinstance(satellite, dict):
                continue
            fallback = {
                "name": satellite.get("name") or str(label).split(" (")[0],
                "norad_id": satellite.get("norad_id"),
            }
            tle_records = _iter_tle_records(satellite)
            for position in satellite.get("positions", []):
                if isinstance(position, dict):
                    tle_records.extend(_iter_tle_records(position))
                    if position.get("tle_line1") or position.get("TLE_LINE1"):
                        tle_records.append(position)

            for record in tle_records:
                entry = _catalog_entry_from_tle(record, fallback)
                if entry is None:
                    continue
                key = (entry["line1"], entry["line2"])
                if key in seen:
                    continue
                seen.add(key)
                catalog.append(entry)

    return catalog


class SatCheckerCandidateProvider:
    """Candidate provider backed by the SatChecker FOV API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SATCHECKER_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = (
            timeout_s
            if timeout_s is not None
            else _env_float("ARGUS_SATCHECKER_TIMEOUT_S", _DEFAULT_TIMEOUT_S)
        )

    def get_catalog(
        self,
        detections: list[dict[str, Any]],
        obs_time: datetime,
        observer_lat: float,
        observer_lon: float,
        observer_alt_m: float,
        *,
        exposure_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch candidate TLEs for detections from SatChecker.

        Args:
            detections: ARGUS detections with sky-coordinate tips.
            obs_time: Exposure start time in UTC.
            observer_lat: Observer latitude in degrees.
            observer_lon: Observer longitude in degrees east.
            observer_alt_m: Observer elevation in meters.
            exposure_time: Exposure duration in seconds. Defaults to
                ``ARGUS_SATCHECKER_DEFAULT_DURATION_S`` or 2 seconds.

        Returns:
            Catalog entries consumable by ``inference.crossid``.
        """
        fov = derive_fov_from_detections(detections)
        if fov is None:
            logger.debug("SatChecker provider skipped: detections have no RA/Dec")
            return []

        ra, dec, fov_radius = fov
        duration = max(
            float(exposure_time) if exposure_time is not None else _env_float(
                "ARGUS_SATCHECKER_DEFAULT_DURATION_S",
                _DEFAULT_DURATION_S,
            ),
            1.0,
        )
        params = {
            "latitude": observer_lat,
            "longitude": observer_lon,
            "elevation": observer_alt_m,
            "start_time_jd": f"{_to_julian_date(obs_time):.8f}",
            "duration": f"{duration:.3f}",
            "ra": f"{ra:.8f}",
            "dec": f"{dec:.8f}",
            "fov_radius": f"{fov_radius:.6f}",
            "group_by": "satellite",
            "include_tles": "true",
            "async": "false",
            "data_source": os.environ.get("ARGUS_SATCHECKER_DATA_SOURCE", "any"),
        }
        if os.environ.get("ARGUS_SATCHECKER_ILLUMINATED_ONLY", "").lower() in {"1", "true", "yes"}:
            params["illuminated_only"] = "true"
        constellation = os.environ.get("ARGUS_SATCHECKER_CONSTELLATION")
        if constellation:
            params["constellation"] = constellation

        url = f"{self.base_url}{_FOV_ENDPOINT}"
        logger.info(
            "Fetching SatChecker candidates: ra=%.4f dec=%.4f radius=%.3f duration=%.1fs",
            ra,
            dec,
            fov_radius,
            duration,
        )
        response = requests.get(url, params=params, timeout=self.timeout_s)
        response.raise_for_status()
        catalog = catalog_from_satchecker_response(response.json())
        logger.info("SatChecker returned %d TLE candidate records", len(catalog))
        return catalog
