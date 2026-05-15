"""Satellite ephemeris cross-identification for ARGUS inference detections.

Cross-matches DINO OBB detections against Space-Track TLEs using SGP4
propagation and Gaussian confidence scoring.

Enhancements adapted from SkyTrack (colleague's pipeline):
  - Scores against streak midpoint RA/Dec propagated at mid-exposure time,
    which is physically correct (the satellite is at the midpoint at the
    middle of the exposure, not at either tip).
  - Computes along-track (Atrk) and cross-track (Xtrk) residuals per
    candidate, decomposing positional error into timing vs. orbital-plane
    components.
  - Disambiguates streak direction (which tip is start vs. end) by
    propagating the top candidate at exposure-start time and assigning the
    tip closer to that position as tip1 (start).
  - Computes expected streak length in pixels from SGP4 and plate scale
    derived from the detection geometry, used as an additional confidence
    factor.

TLE source routing (Space-Track API policy compliance):
  - inference reads only from the local tle_catalog table
  - if local coverage is missing, objects are left unidentified/unknown
  - Space-Track current/history calls are explicit maintenance tasks, never
    automatic inference fallbacks

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

from src.matching.tle_manager import TLECatalogManager

logger = logging.getLogger(__name__)

# Gaussian sigma for position score: 0.25° = 900 arcsec
_POSITION_SIGMA_ARCSEC = 900.0

# Gaussian sigma for length score: 40% relative error → score ≈ 0.61
_LENGTH_SIGMA_RELATIVE = 0.4


# ---------------------------------------------------------------------------
# TLE loading from Space-Track
# ---------------------------------------------------------------------------

_tle_manager = TLECatalogManager()


def _fetch_tle_catalog(
    obs_time: datetime,
    epoch_window_days: int,
) -> list[tuple[str, str, str]]:
    """Return TLEs for the observation window via the two-track TLE manager.

    Delegates to :class:`~src.matching.tle_manager.TLECatalogManager`, which
    implements the live/historical branching policy:

    - Recent observations (< 72 h): local DB first; CelesTrak refresh on miss.
    - Historical observations (≥ 72 h): local DB only; miss → unknown.

    Space-Track ``gp_history`` is never called here.

    Args:
        obs_time: UTC observation time.
        epoch_window_days: Days before obs_time to search.

    Returns:
        List of (name, line1, line2) tuples ready for SGP4 propagation.
    """
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    db_rows = _tle_manager.get_tles(obs_time, epoch_window_days)
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

    catalog = [
        (r["object_name"], r["tle_line1"], r["tle_line2"])
        for r in db_rows
        if r.get("tle_line1") and r.get("tle_line2")
    ]
    logger.debug(
        "TLE manager returned %d records; %d have TLE lines",
        len(db_rows),
        len(catalog),
    )
    return catalog


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

    try:
        ts = _timescale()
        observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_alt_m)
        t = ts.from_datetime(obs_time)
        topocentric = (sat - observer).at(t)
        ra_angle, dec_angle, _ = topocentric.radec()
        predicted_ra  = float(ra_angle._degrees) % 360.0
        predicted_dec = float(dec_angle._degrees)
    except Exception as exc:
        logger.debug("Propagation failed for %s: %s", name, exc)
        return None

    return {
        "object_name":   name,
        "norad_id":      norad_id,
        "predicted_ra":  predicted_ra,
        "predicted_dec": predicted_dec,
        "tle_age_hours": tle_age_hours,
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

    if ra1 is not None and dec1 is not None and ra2 is not None and dec2 is not None:
        dra = ra2 - ra1
        dra -= 360.0 * math.floor((dra + 180.0) / 360.0)
        return ((ra1 + dra / 2.0) % 360.0, (dec1 + dec2) / 2.0)
    if ra1 is not None and dec1 is not None:
        return (ra1, dec1)
    if ra2 is not None and dec2 is not None:
        return (ra2, dec2)
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

    try:
        catalog = _fetch_tle_catalog(obs_time, epoch_window_days)
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
            det.get(k) is not None
            for k in ("ra_tip1_deg", "dec_tip1_deg", "ra_tip2_deg", "dec_tip2_deg")
        )

        plate_scale = _plate_scale_from_det(det)

        # --- Score every TLE against the midpoint at mid-exposure time -------
        candidates: list[dict] = []
        for name, line1, line2 in catalog:
            result = _propagate_to_radec(
                name, line1, line2, mid_time,
                observer_lat, observer_lon, observer_alt_m,
            )
            if result is None:
                continue

            sep_arcsec = _angular_separation_arcsec(
                mid_ra, mid_dec,
                result["predicted_ra"], result["predicted_dec"],
            )
            pos_score = _gaussian_score(sep_arcsec, sigma=_POSITION_SIGMA_ARCSEC)

            candidates.append({
                "satellite_name":    result["object_name"],
                "norad_id":          result["norad_id"],
                "confidence":        pos_score,
                "separation_arcsec": sep_arcsec,
                "tle_age_hours":     result["tle_age_hours"],
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
        if has_both_tips:
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
        if top3 and has_both_tips and exposure_time is not None:
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
                            # Composite confidence: position × length match
                            best["confidence"] = round(best["confidence"] * length_score, 4)
                            logger.debug(
                                "Expected length: %.0fpx  observed: %.0fpx  "
                                "rel_err=%.2f  length_score=%.3f",
                                exp_length_px, obs_length_px, rel_err, length_score,
                            )

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
