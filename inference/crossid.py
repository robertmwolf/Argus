"""Satellite ephemeris cross-identification for ARGUS inference detections.

Cross-matches DINO OBB detections against Space-Track TLEs using SGP4
propagation and Gaussian confidence scoring.

Enhancements adapted from SkyTrack (colleague's pipeline):
  - Scores against streak midpoint RA/Dec propagated at mid-exposure time,
    which is physically correct (the satellite is at the midpoint at the
    middle of the exposure, not at either tip).
  - For border-clipped streaks, scores the propagated midpoint against the
    visible RA/Dec segment and skips endpoint-dependent length/direction
    refinements because the measured tips are image-boundary intersections.
  - Computes along-track (Atrk) and cross-track (Xtrk) residuals per
    candidate, decomposing positional error into timing vs. orbital-plane
    components.
  - Disambiguates streak direction (which tip is start vs. end) by
    propagating the top candidate at exposure-start time and assigning the
    tip closer to that position as tip1 (start).
  - Computes expected streak length in pixels from SGP4 and plate scale
    derived from the detection geometry, used as an additional confidence
    factor.

TLE source routing:
  - inference reads the local tle_catalog first
  - on a local miss, development can refresh current GP data through
    Space-Track's test API via TLECatalogManager
  - gp_history is never called from inference

# Source: Danarianto et al. — Gaussian confidence scoring for satellite crossID
# Ref: cite per published paper

# Source: SkyTrack (colleague) — along-track/cross-track residuals,
#         direction disambiguation, expected streak length
# Ref: examples/streak_live.inc
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from src.matching.scorer import tle_age_penalty
from src.matching.tle_manager import TLECatalogManager

logger = logging.getLogger(__name__)

# Gaussian sigma for position score: 0.25° = 900 arcsec
_POSITION_SIGMA_ARCSEC = 900.0

# Gaussian sigma for length score: 40% relative error → score ≈ 0.61
_LENGTH_SIGMA_RELATIVE = 0.4

# A broad epoch hit still counts as "no match" if no propagated candidate lands
# within roughly a degree of the detection midpoint.
_MIN_VIABLE_POSITION_SCORE = 0.01

# Mirrors inference.postprocess.QUALITY_EDGE without importing the postprocess
# module into the cross-ID path.
_QUALITY_EDGE = 1


# ---------------------------------------------------------------------------
# TLE loading from Space-Track
# ---------------------------------------------------------------------------

_tle_manager = TLECatalogManager()


def _format_utc(dt: datetime | None) -> str | None:
    """Format a datetime as compact ISO8601 UTC for API/UI metadata."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _catalog_entry(raw: Any) -> dict[str, Any]:
    """Normalize legacy tuple catalogs and metadata-rich catalog dicts."""
    if isinstance(raw, tuple):
        name, line1, line2 = raw
        return {"name": name, "line1": line1, "line2": line2}
    return raw


def _fetch_tle_catalog(
    obs_time: datetime,
    epoch_window_days: int,
    min_mean_motion: float = 0,
) -> list[dict[str, Any]]:
    """Return TLEs for the observation window via the two-track TLE manager.

    Delegates to :class:`~src.matching.tle_manager.TLECatalogManager`, which
    implements the live/historical branching policy:

    - Recent observations (< 72 h): local DB first; CelesTrak refresh on miss.
    - Historical observations (≥ 72 h): local DB only; miss → unknown.

    Space-Track ``gp_history`` is never called here.

    Args:
        obs_time: UTC observation time.
        epoch_window_days: Days before obs_time to search.
        min_mean_motion: Minimum mean_motion in rev/day (0 = all orbit classes
            including GEO; 11.25 = LEO only).

    Returns:
        List of catalog dicts ready for SGP4 propagation and scoring.
    """
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    db_rows = _tle_manager.get_tles(obs_time, epoch_window_days, min_mean_motion)
    if not db_rows:
        age_hours = (datetime.now(tz=timezone.utc) - obs_time).total_seconds() / 3600
        logger.warning(
            "No TLE coverage for obs_time=%s, window=%dd, age=%.1fh; "
            "leaving detections unknown",
            obs_time.isoformat(),
            epoch_window_days,
            age_hours,
        )
        return []

    catalog = _catalog_from_rows(db_rows)
    logger.debug(
        "TLE manager returned %d records; %d have TLE lines",
        len(db_rows),
        len(catalog),
    )
    return catalog


def _catalog_from_rows(db_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map TLE manager rows into cross-ID catalog entries."""
    return [
        {
            "name": r["object_name"],
            "line1": r["tle_line1"],
            "line2": r["tle_line2"],
            "tle_epoch": r.get("epoch"),
            "source": r.get("source"),
            "epoch_drift_hours": r.get("epoch_drift_hours"),
            "epoch_search_window_days": r.get("epoch_search_window_days"),
            "tle_search_mode": r.get("tle_search_mode"),
            "tle_data_fresh_at": r.get("tle_data_fresh_at"),
        }
        for r in db_rows
        if r.get("tle_line1") and r.get("tle_line2")
    ]


def _fetch_current_tle_catalog(obs_time: datetime) -> list[dict[str, Any]]:
    """Return current fallback TLEs, refreshing stale data through the manager."""
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)
    return _catalog_from_rows(_tle_manager.get_current_fallback_tles(obs_time))


def _catalog_has_mode(catalog: list[dict[str, Any]], mode: str) -> bool:
    """Return True when any catalog entry came from a given search mode."""
    return any(_catalog_entry(entry).get("tle_search_mode") == mode for entry in catalog)


def _best_position_score(detections: list[dict]) -> float:
    """Return the best raw position score currently attached to detections."""
    best = 0.0
    for det in detections:
        for ident in det.get("identifications", []):
            score = ident.get("position_score")
            if score is None:
                continue
            best = max(best, float(score))
    return best


# ---------------------------------------------------------------------------
# SGP4 propagation
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

    try:
        norad_id = int(line2[2:7].strip())
    except ValueError:
        norad_id = 0

    try:
        epoch_str = line1[18:32].strip()
        yr2 = int(epoch_str[:2])
        year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
        day_frac = float(epoch_str[2:])
        tle_epoch = datetime(year, 1, 1, tzinfo=timezone.utc)
        tle_epoch += timedelta(days=day_frac - 1)
        tle_age_hours = (obs_time - tle_epoch).total_seconds() / 3600.0
    except Exception:
        tle_age_hours = 0.0
        tle_epoch = None

    try:
        ts = _timescale()
        observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_alt_m)
        t = ts.from_datetime(obs_time)
        topocentric = (sat - observer).at(t)
        ra_angle, dec_angle, _ = topocentric.radec()
        predicted_ra  = float(ra_angle._degrees) % 360.0
        predicted_dec = float(dec_angle._degrees)
        if not (math.isfinite(predicted_ra) and math.isfinite(predicted_dec)):
            logger.debug("Propagation returned NaN RA/Dec for %s — skipping", name)
            return None
    except Exception as exc:
        logger.debug("Propagation failed for %s: %s", name, exc)
        return None

    return {
        "object_name":   name,
        "norad_id":      norad_id,
        "predicted_ra":  predicted_ra,
        "predicted_dec": predicted_dec,
        "tle_age_hours": tle_age_hours,
        "tle_epoch":     tle_epoch,
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angular_separation_arcsec(
    ra1: float, dec1: float,
    ra2: float, dec2: float,
) -> float:
    """Angular separation between two sky positions in arcseconds (haversine).

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


def _is_edge_clipped(det: dict) -> bool:
    """Return True when detection endpoints are likely image-border clips.

    Border-clipped streak tips are not physical exposure start/end points.
    Treating them as full-track endpoints biases midpoint, direction, and
    length-based catalogue scores.

    Args:
        det: Detection dict, optionally including quality metadata from
            :func:`inference.postprocess.classify_detection_quality`.

    Returns:
        True when the detection touches/crosses an image border.
    """
    return bool(
        det.get("edge_clipped")
        or det.get("edge_contacts")
        or det.get("quality_flag") == _QUALITY_EDGE
    )


def _segment_separation_arcsec(
    point_ra: float,
    point_dec: float,
    start_ra: float,
    start_dec: float,
    end_ra: float,
    end_dec: float,
) -> float:
    """Shortest angular distance from a sky point to a small RA/Dec segment.

    The local tangent-plane approximation is appropriate for the short streak
    segments used here and handles RA wrap before projecting.

    Args:
        point_ra, point_dec: Sky point in degrees.
        start_ra, start_dec: First visible streak tip in degrees.
        end_ra, end_dec: Second visible streak tip in degrees.

    Returns:
        Closest distance to the finite visible segment in arcseconds.
    """
    cos_dec = math.cos(math.radians(point_dec))

    def project(ra: float, dec: float) -> tuple[float, float]:
        dra = ra - point_ra
        dra -= 360.0 * math.floor((dra + 180.0) / 360.0)
        return (dra * cos_dec * 3600.0, (dec - point_dec) * 3600.0)

    sx, sy = project(start_ra, start_dec)
    ex, ey = project(end_ra, end_dec)
    vx = ex - sx
    vy = ey - sy
    mag2 = vx * vx + vy * vy
    if mag2 < 1.0:
        return min(
            _angular_separation_arcsec(point_ra, point_dec, start_ra, start_dec),
            _angular_separation_arcsec(point_ra, point_dec, end_ra, end_dec),
        )

    # Point is the tangent-plane origin, so closest parameter is -S dot V / |V|².
    u = max(0.0, min(1.0, -(sx * vx + sy * vy) / mag2))
    cx = sx + u * vx
    cy = sy + u * vy
    return math.sqrt(cx * cx + cy * cy)


def _gaussian_score(delta: float, sigma: float) -> float:
    """Score in [0, 1] using Gaussian falloff from zero error.

    # Source: ARGUS architecture — Gaussian scoring formulas
    # Ref: agent_docs/architecture.md

    Args:
        delta: Absolute error value.
        sigma: Characteristic scale (score ≈ 0.61 at delta == sigma).

    Returns:
        Score in [0.0, 1.0].
    """
    return math.exp(-0.5 * (delta / sigma) ** 2)


def _confidence_with_epoch_penalty(position_score: float, tle_age_hours: float) -> tuple[float, float]:
    """Apply the TLE date-drift penalty to positional match confidence.

    The position-only cross-ID score can look deceptively strong when ARGUS is
    forced to use a broad epoch window or current catalog against an old photo.
    Reusing the established LEO TLE age penalty makes that date drift explicit:
    24 h keeps about 61% of the positional confidence, 48 h about 14%, and
    72 h about 1%.

    Args:
        position_score: Raw angular-position score in [0, 1].
        tle_age_hours: Hours between TLE epoch and photo time.

    Returns:
        ``(penalized_confidence, epoch_penalty)``.
    """
    penalty = tle_age_penalty(abs(tle_age_hours))
    return position_score * penalty, penalty


# ---------------------------------------------------------------------------
# SkyTrack-derived helpers
# ---------------------------------------------------------------------------

def _streak_mid_radec(det: dict) -> tuple[float, float] | None:
    """Return the midpoint RA/Dec of a streak from its two tips.

    Falls back to whichever tip is available if only one has sky coords.

    Args:
        det: Detection dict with ra_tip1_deg, dec_tip1_deg, ra_tip2_deg,
             dec_tip2_deg keys (any may be None).

    Returns:
        (mid_ra_deg, mid_dec_deg), or None if no sky coords at all.
    """
    ra1  = det.get("ra_tip1_deg")
    dec1 = det.get("dec_tip1_deg")
    ra2  = det.get("ra_tip2_deg")
    dec2 = det.get("dec_tip2_deg")

    def _valid(v: float | None) -> bool:
        return v is not None and math.isfinite(v)

    if _valid(ra1) and _valid(dec1) and _valid(ra2) and _valid(dec2):
        dra = ra2 - ra1  # type: ignore[operator]
        dra -= 360.0 * math.floor((dra + 180.0) / 360.0)
        return ((ra1 + dra / 2.0) % 360.0, (dec1 + dec2) / 2.0)  # type: ignore[operator]
    if _valid(ra1) and _valid(dec1):
        return (ra1, dec1)  # type: ignore[return-value]
    if _valid(ra2) and _valid(dec2):
        return (ra2, dec2)  # type: ignore[return-value]
    return None


def _plate_scale_from_det(det: dict) -> float | None:
    """Derive plate scale (arcsec/pixel) from a detection's geometry.

    Computes the angular separation between the two tips and divides by the
    OBB long axis (streak length in pixels).  Only valid when both tips have
    sky coordinates and the streak is long enough to give a reliable estimate.

    Args:
        det: Detection dict.

    Returns:
        Plate scale in arcsec/pixel, or None if it cannot be derived.
    """
    ra1  = det.get("ra_tip1_deg")
    dec1 = det.get("dec_tip1_deg")
    ra2  = det.get("ra_tip2_deg")
    dec2 = det.get("dec_tip2_deg")
    obb  = det.get("obb")

    if None in (ra1, dec1, ra2, dec2) or obb is None:
        return None

    length_px = float(obb.get("w", 0))
    if length_px < 10.0:
        return None

    sep_arcsec = _angular_separation_arcsec(ra1, dec1, ra2, dec2)
    if sep_arcsec < 1.0:
        return None

    return sep_arcsec / length_px


def _atrk_xtrk(
    obs_ra: float, obs_dec: float,
    pred_ra: float, pred_dec: float,
    start_ra: float, start_dec: float,
    end_ra: float, end_dec: float,
) -> tuple[float, float]:
    """Decompose a positional residual into along-track and cross-track components.

    Along-track (Atrk): error component in the satellite's direction of motion.
    A large Atrk usually indicates a TLE epoch timing offset.

    Cross-track (Xtrk): error component perpendicular to motion.
    A large Xtrk usually indicates the wrong satellite or bad orbit plane.

    # Source: SkyTrack (colleague) — ComputeOneResidual
    # Ref: examples/streak_live.inc, line 2083

    Args:
        obs_ra, obs_dec:     Observed mid-streak position (degrees).
        pred_ra, pred_dec:   Predicted position from SGP4 (degrees).
        start_ra, start_dec: Observed streak start tip (degrees).
        end_ra, end_dec:     Observed streak end tip (degrees).

    Returns:
        (atrk_arcsec, xtrk_arcsec) — signed residuals in arcseconds.
    """
    cos_dec_pred = math.cos(math.radians(pred_dec))
    cos_dec_obs  = math.cos(math.radians(obs_dec))

    # Positional residual in arcseconds
    dra = obs_ra - pred_ra
    dra -= 360.0 * math.floor((dra + 180.0) / 360.0)
    dra_arcsec  = dra * cos_dec_pred * 3600.0
    ddec_arcsec = (obs_dec - pred_dec) * 3600.0

    # Track direction vector (start → end) in arcseconds
    mra = end_ra - start_ra
    mra -= 360.0 * math.floor((mra + 180.0) / 360.0)
    mra_arcsec  = mra * cos_dec_obs * 3600.0
    mdec_arcsec = (end_dec - start_dec) * 3600.0

    mag = math.sqrt(mra_arcsec ** 2 + mdec_arcsec ** 2)
    if mag < 1.0:
        return 0.0, 0.0

    ura  = mra_arcsec / mag
    udec = mdec_arcsec / mag

    atrk =  dra_arcsec * ura  + ddec_arcsec * udec
    xtrk = -dra_arcsec * udec + ddec_arcsec * ura
    return atrk, xtrk


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
    exposure_time: float | None = None,
    min_mean_motion: float = 0,
    _catalog_override: list[dict[str, Any]] | None = None,
    _allow_current_retry: bool = True,
) -> list[dict]:
    """Cross-match detections against Space-Track TLEs.

    For each detection:
      1. Scores TLE candidates against the streak midpoint propagated at
         mid-exposure time (more accurate than scoring against either tip).
      2. Computes along-track / cross-track residuals for the top-3 candidates.
      3. When exposure_time is provided, disambiguates which tip is the start
         of the pass by propagating the top candidate at exposure-start time.
      4. Computes expected streak length from SGP4 + plate scale and applies
         a length-match confidence penalty to the top candidate.

    Each detection dict is mutated in-place: an 'identifications' key is
    added containing up to 3 ranked candidate dicts.

    # Source: Danarianto et al. — Gaussian confidence scoring for satellite crossID
    # Ref: cite per published paper

    # Source: SkyTrack (colleague) — midpoint scoring, Atrk/Xtrk, direction
    #         disambiguation, expected streak length
    # Ref: examples/streak_live.inc

    Args:
        detections: List of detection dicts from the inference pipeline.
        obs_time: UTC DATE-OBS from FITS header (= exposure start time).
        observer_lat: Observer geodetic latitude in degrees.
        observer_lon: Observer geodetic longitude in degrees.
        observer_alt_m: Observer elevation above WGS84 in metres.
        epoch_window_days: TLE epoch search window in days.
        exposure_time: Exposure duration in seconds.  When provided, scoring
            uses the true mid-exposure time and direction disambiguation is
            performed.
        min_mean_motion: Minimum mean_motion in rev/day (0 = all orbit classes
            including GEO/MEO; 11.25 = LEO only).  Default 0 so GEO targets
            are not silently excluded.

    Returns:
        The mutated *detections* list (same objects, with 'identifications'
        added).
    """
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    # obs_time is DATE-OBS = exposure start; scoring uses mid-exposure
    if exposure_time is not None:
        mid_time   = obs_time + timedelta(seconds=exposure_time / 2.0)
        start_time = obs_time
        end_time   = obs_time + timedelta(seconds=exposure_time)
    else:
        mid_time   = obs_time
        start_time = obs_time
        end_time   = None

    if _catalog_override is not None:
        catalog = _catalog_override
    else:
        try:
            catalog = _fetch_tle_catalog(obs_time, epoch_window_days, min_mean_motion)
        except Exception:
            logger.warning(
                "TLE catalog fetch failed — skipping cross-identification",
                exc_info=True,
            )
            catalog = []

    if not catalog:
        logger.warning("Empty TLE catalog — all identifications will be empty")
        for det in detections:
            det.setdefault("identifications", [])
        return detections

    logger.debug(
        "Cross-identifying %d detections against %d TLEs",
        len(detections), len(catalog),
    )

    for det in detections:
        # Compute midpoint sky coords for scoring
        mid = _streak_mid_radec(det)
        if mid is None:
            logger.debug("Detection has no sky coords — skipping cross-ID")
            det["identifications"] = []
            continue

        mid_ra, mid_dec = mid

        has_both_tips = all(
            det.get(k) is not None and math.isfinite(det[k])
            for k in ("ra_tip1_deg", "dec_tip1_deg", "ra_tip2_deg", "dec_tip2_deg")
        )
        edge_clipped = _is_edge_clipped(det)

        plate_scale = _plate_scale_from_det(det)

        # --- Score every TLE against the midpoint at mid-exposure time -------
        candidates: list[dict] = []
        for raw_entry in catalog:
            entry = _catalog_entry(raw_entry)
            name = entry["name"]
            line1 = entry["line1"]
            line2 = entry["line2"]
            result = _propagate_to_radec(
                name, line1, line2, mid_time,
                observer_lat, observer_lon, observer_alt_m,
            )
            if result is None:
                continue

            midpoint_sep_arcsec = _angular_separation_arcsec(
                mid_ra, mid_dec,
                result["predicted_ra"], result["predicted_dec"],
            )
            sep_arcsec = midpoint_sep_arcsec
            position_score_mode = "midpoint"
            if edge_clipped and has_both_tips:
                sep_arcsec = _segment_separation_arcsec(
                    result["predicted_ra"],
                    result["predicted_dec"],
                    det["ra_tip1_deg"],
                    det["dec_tip1_deg"],
                    det["ra_tip2_deg"],
                    det["dec_tip2_deg"],
                )
                position_score_mode = "edge_visible_segment"

            pos_score = _gaussian_score(sep_arcsec, sigma=_POSITION_SIGMA_ARCSEC)
            confidence, epoch_penalty = _confidence_with_epoch_penalty(
                pos_score,
                result["tle_age_hours"],
            )

            candidates.append({
                "satellite_name":    result["object_name"],
                "norad_id":          result["norad_id"],
                "confidence":        confidence,
                "position_score":    round(pos_score, 4),
                "position_score_mode": position_score_mode,
                "epoch_penalty":     round(epoch_penalty, 4),
                "separation_arcsec": sep_arcsec,
                "midpoint_separation_arcsec": round(midpoint_sep_arcsec, 2),
                "tle_age_hours":     result["tle_age_hours"],
                "tle_epoch":         entry.get("tle_epoch") or _format_utc(result.get("tle_epoch")),
                "photo_taken_at":    _format_utc(obs_time),
                "tle_data_fresh_at": entry.get("tle_data_fresh_at"),
                "tle_source":        entry.get("source"),
                "tle_search_mode":   entry.get("tle_search_mode"),
                "epoch_search_window_days": entry.get("epoch_search_window_days"),
                "epoch_drift_hours": entry.get("epoch_drift_hours")
                    if entry.get("epoch_drift_hours") is not None
                    else abs(result["tle_age_hours"]),
                "rank":              0,
                "_line1":            line1,
                "_line2":            line2,
                "_pred_ra":          result["predicted_ra"],
                "_pred_dec":         result["predicted_dec"],
            })

        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        top3 = candidates[:3]

        # --- Atrk / Xtrk for all top-3 (no extra propagations needed) -------
        # Source: SkyTrack (colleague) — ComputeOneResidual
        if has_both_tips and not edge_clipped:
            for cand in top3:
                atrk, xtrk = _atrk_xtrk(
                    mid_ra, mid_dec,
                    cand["_pred_ra"], cand["_pred_dec"],
                    det["ra_tip1_deg"], det["dec_tip1_deg"],
                    det["ra_tip2_deg"], det["dec_tip2_deg"],
                )
                cand["atrk_arcsec"] = round(atrk, 2)
                cand["xtrk_arcsec"] = round(xtrk, 2)

        # --- Top-1 only: direction disambiguation + expected length ----------
        # Source: SkyTrack (colleague) — StreakDirection, StreakExpectedPos
        if (
            top3
            and has_both_tips
            and not edge_clipped
            and exposure_time is not None
            and top3[0].get("position_score", 0.0) >= _MIN_VIABLE_POSITION_SCORE
        ):
            best = top3[0]

            # Propagate at exposure start to find which tip is the start
            start_result = _propagate_to_radec(
                best["satellite_name"], best["_line1"], best["_line2"],
                start_time, observer_lat, observer_lon, observer_alt_m,
            )
            if start_result is not None:
                d1 = _angular_separation_arcsec(
                    det["ra_tip1_deg"], det["dec_tip1_deg"],
                    start_result["predicted_ra"], start_result["predicted_dec"],
                )
                d2 = _angular_separation_arcsec(
                    det["ra_tip2_deg"], det["dec_tip2_deg"],
                    start_result["predicted_ra"], start_result["predicted_dec"],
                )
                if d1 > d2:
                    # tip1 is farther from the expected start — swap
                    det["ra_tip1_deg"],  det["ra_tip2_deg"]  = det["ra_tip2_deg"],  det["ra_tip1_deg"]
                    det["dec_tip1_deg"], det["dec_tip2_deg"] = det["dec_tip2_deg"], det["dec_tip1_deg"]
                    det["streak_direction_swapped"] = True
                    logger.debug("Direction: swapped streak endpoints for NORAD %d", best["norad_id"])
                else:
                    det["streak_direction_swapped"] = False

                # Expected streak length via start + end propagation
                if end_time is not None and plate_scale is not None:
                    end_result = _propagate_to_radec(
                        best["satellite_name"], best["_line1"], best["_line2"],
                        end_time, observer_lat, observer_lon, observer_alt_m,
                    )
                    if end_result is not None:
                        exp_sep_arcsec = _angular_separation_arcsec(
                            start_result["predicted_ra"], start_result["predicted_dec"],
                            end_result["predicted_ra"],   end_result["predicted_dec"],
                        )
                        exp_length_px = exp_sep_arcsec / plate_scale
                        obs_length_px = det.get("streak_length_px") or 0.0

                        if exp_length_px > 0:
                            rel_err = abs(obs_length_px - exp_length_px) / exp_length_px
                            length_score = _gaussian_score(rel_err, sigma=_LENGTH_SIGMA_RELATIVE)
                            best["expected_length_px"] = round(exp_length_px, 1)
                            best["length_score"]       = round(length_score, 3)
                            # Composite confidence: date-penalized position × length match
                            best["confidence"] = round(best["confidence"] * length_score, 4)
                            logger.debug(
                                "Expected length: %.0fpx  observed: %.0fpx  "
                                "rel_err=%.2f  length_score=%.3f",
                                exp_length_px, obs_length_px, rel_err, length_score,
                            )
        elif top3 and edge_clipped:
            for cand in top3:
                cand["edge_clipped"] = True
                cand["length_score_skipped"] = True
                cand["direction_disambiguation_skipped"] = True

        # --- Clean up temp fields and assign ranks ---------------------------
        for rank, cand in enumerate(top3, start=1):
            cand.pop("_line1", None)
            cand.pop("_line2", None)
            cand.pop("_pred_ra", None)
            cand.pop("_pred_dec", None)
            cand["rank"] = rank

        det["identifications"] = top3

        if top3:
            logger.debug(
                "Best match: %s (NORAD %d) sep=%.1f\" conf=%.3f",
                top3[0]["satellite_name"], top3[0]["norad_id"],
                top3[0]["separation_arcsec"], top3[0]["confidence"],
            )

    if (
        _allow_current_retry
        and _catalog_has_mode(catalog, "broad_epoch")
        and _best_position_score(detections) < _MIN_VIABLE_POSITION_SCORE
    ):
        logger.info(
            "Broad epoch TLE search produced no viable propagated match "
            "(best position score %.4f); trying current TLE fallback",
            _best_position_score(detections),
        )
        try:
            current_catalog = _fetch_current_tle_catalog(obs_time)
        except Exception:
            logger.warning(
                "Current TLE fallback fetch failed — keeping broad epoch results",
                exc_info=True,
            )
            current_catalog = []

        if current_catalog:
            return cross_identify(
                detections,
                obs_time,
                observer_lat,
                observer_lon,
                observer_alt_m,
                epoch_window_days=epoch_window_days,
                exposure_time=exposure_time,
                min_mean_motion=min_mean_motion,
                _catalog_override=current_catalog,
                _allow_current_retry=False,
            )

    return detections


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    obs = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)
    dets = [
        {
            "ra_tip1_deg": 83.82, "dec_tip1_deg": -5.39,
            "ra_tip2_deg": 83.85, "dec_tip2_deg": -5.41,
            "obb": {"cx": 400, "cy": 300, "w": 120, "h": 4, "angle_deg": 30.0},
            "streak_length_px": 120,
            "confidence": 0.9,
        },
        {
            "ra_tip1_deg": None, "dec_tip1_deg": None,
            "ra_tip2_deg": None, "dec_tip2_deg": None,
            "obb": None, "streak_length_px": 0, "confidence": 0.5,
        },
    ]

    result = cross_identify(
        dets, obs,
        observer_lat=49.61, observer_lon=6.13, observer_alt_m=280.0,
        epoch_window_days=3,
        exposure_time=2.0,
    )

    for i, d in enumerate(result):
        ids = d.get("identifications", [])
        print(
            f"\nDetection {i}: mid=({_streak_mid_radec(d)})  "
            f"dir_swapped={d.get('streak_direction_swapped')}"
        )
        if ids:
            for c in ids:
                atrk = c.get("atrk_arcsec", "—")
                xtrk = c.get("xtrk_arcsec", "—")
                exp_l = c.get("expected_length_px", "—")
                print(
                    f"  rank {c['rank']}: {c['satellite_name']} "
                    f"(NORAD {c['norad_id']}) sep={c['separation_arcsec']:.1f}\" "
                    f"conf={c['confidence']:.3f}  "
                    f"Atrk={atrk}\"  Xtrk={xtrk}\"  exp_len={exp_l}px"
                )
        else:
            print("  (no identifications)")
    sys.exit(0)
