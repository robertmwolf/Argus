"""Multi-factor candidate matching engine for the ARGUS pipeline.

Scores each propagated TLE candidate against an observed StreakDetection
using angular position, velocity, direction, and (optionally) magnitude,
then returns a ranked list of CandidateMatch objects.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

from src.detection.classical_detector import StreakDetection
from src.ingest.fits_parser import FITSImage
from src.matching.scorer import aggregate_score, gaussian_score

logger = logging.getLogger(__name__)

# Per-factor Gaussian sigma values (from architecture.md)
_SIGMA_POSITION_DEG = 0.25
_SIGMA_VELOCITY_PCT = 5.0
_SIGMA_DIRECTION_DEG = 2.5
_MAGNITUDE_NEUTRAL = 0.5   # neutral score when magnitude unavailable
_AMBIGUITY_THRESHOLD = 0.05


@dataclass
class CandidateMatch:
    """A single TLE candidate scored against an observed streak.

    Attributes:
        norad_id: NORAD catalog ID of the satellite.
        object_name: Common name of the satellite.
        tle_epoch: Epoch datetime of the TLE used.
        tle_age_hours: Hours from tle_epoch to obs_time.
        predicted_ra: Predicted right ascension at obs_time (degrees).
        predicted_dec: Predicted declination at obs_time (degrees).
        predicted_velocity_arcsec_s: Predicted angular velocity (arcsec/s).
        predicted_direction_deg: Predicted celestial position angle (degrees).
        predicted_magnitude: Predicted visual magnitude (or None).
        angular_sep_arcsec: Separation between observed and predicted position.
        velocity_delta_pct: Percent difference in angular velocity.
        direction_delta_deg: Absolute PA difference after 180° ambiguity resolution.
        magnitude_delta: Magnitude difference (or None).
        position_score: Gaussian score for position match (0–1).
        velocity_score: Gaussian score for velocity match (0–1).
        direction_score: Gaussian score for direction match (0–1).
        magnitude_score: Gaussian score for magnitude match (0–1).
        weighted_score: Final aggregated score (0–1).
        ambiguous: True if the next-ranked candidate is within 0.05 of this score.
    """

    norad_id: int
    object_name: str
    tle_epoch: datetime
    tle_age_hours: float
    predicted_ra: float
    predicted_dec: float
    predicted_velocity_arcsec_s: float
    predicted_direction_deg: float
    predicted_magnitude: float | None
    angular_sep_arcsec: float
    velocity_delta_pct: float
    direction_delta_deg: float
    magnitude_delta: float | None
    position_score: float
    velocity_score: float
    direction_score: float
    magnitude_score: float
    weighted_score: float
    ambiguous: bool = False


def _angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle angular separation in degrees."""
    r1 = math.radians(ra1)
    d1 = math.radians(dec1)
    r2 = math.radians(ra2)
    d2 = math.radians(dec2)
    cos_sep = (
        math.sin(d1) * math.sin(d2)
        + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def _resolve_direction_delta(observed_pa: float, predicted_pa: float) -> float:
    """Return direction delta resolving the 180° streak direction ambiguity.

    # Source: ARGUS architecture — streak direction ambiguity resolution
    # Ref: agent_docs/architecture.md

    ASTRiDE assigns start/end arbitrarily, so the observed PA carries an
    implicit 180° ambiguity.  We test both orientations against the
    SGP4-predicted direction and take the smaller delta.

    Args:
        observed_pa: Observed position angle in degrees (may be off by 180°).
        predicted_pa: SGP4-predicted celestial position angle in degrees.

    Returns:
        Minimum of |observed - predicted| and |observed + 180 - predicted|,
        both wrapped to [0, 180].
    """
    def _wrap(delta: float) -> float:
        delta = abs(delta) % 360.0
        return min(delta, 360.0 - delta)

    delta0 = _wrap(observed_pa - predicted_pa)
    delta180 = _wrap((observed_pa + 180.0) - predicted_pa)
    return min(delta0, delta180)


def match(
    detection: StreakDetection,
    candidates: list[dict],
    fits_image: FITSImage,
) -> list[CandidateMatch]:
    """Score each candidate against the detection and return ranked matches.

    # Source: ARGUS architecture — multi-factor weighted scoring
    # Ref: agent_docs/architecture.md

    Args:
        detection: Observed streak with sky coordinates populated.
        candidates: List of propagated candidate dicts (from propagator.propagate()
            or spatial_filter results with ``_propagated`` key).
        fits_image: The source FITS image (used for context; not mutated).

    Returns:
        List of CandidateMatch sorted by weighted_score descending.
        The top candidate has ``ambiguous=True`` when the second candidate's
        weighted_score is within 0.05 of the top score.
        Returns empty list if detection has no sky coordinates.
    """
    if detection.ra_center is None or detection.dec_center is None:
        logger.debug("Detection has no sky coords; skipping match.")
        return []

    scored: list[CandidateMatch] = []

    for cand in candidates:
        # Support both raw propagated dicts and spatial_filter-wrapped dicts
        prop = cand.get("_propagated", cand)

        angular_sep_deg = _angular_separation_deg(
            detection.ra_center,
            detection.dec_center,
            prop["predicted_ra"],
            prop["predicted_dec"],
        )
        angular_sep_arcsec = angular_sep_deg * 3600.0

        # Velocity delta (percent)
        obs_vel = detection.angular_velocity_arcsec_s or 0.0
        pred_vel = prop["predicted_velocity_arcsec_s"]
        if pred_vel > 0:
            velocity_delta_pct = abs(obs_vel - pred_vel) / pred_vel * 100.0
        else:
            velocity_delta_pct = 0.0

        # Direction delta — resolve 180° ambiguity
        obs_pa = detection.position_angle_deg or 0.0
        direction_delta_deg = _resolve_direction_delta(obs_pa, prop["predicted_direction_deg"])

        # Per-factor scores
        position_score = gaussian_score(angular_sep_deg, _SIGMA_POSITION_DEG)
        velocity_score = gaussian_score(velocity_delta_pct, _SIGMA_VELOCITY_PCT)
        direction_score = gaussian_score(direction_delta_deg, _SIGMA_DIRECTION_DEG)
        magnitude_score = _MAGNITUDE_NEUTRAL  # not yet implemented

        weighted = aggregate_score(
            position_score,
            velocity_score,
            direction_score,
            magnitude_score,
            prop["tle_age_hours"],
        )

        scored.append(
            CandidateMatch(
                norad_id=prop["norad_id"],
                object_name=prop["object_name"],
                tle_epoch=prop["tle_epoch"],
                tle_age_hours=prop["tle_age_hours"],
                predicted_ra=prop["predicted_ra"],
                predicted_dec=prop["predicted_dec"],
                predicted_velocity_arcsec_s=prop["predicted_velocity_arcsec_s"],
                predicted_direction_deg=prop["predicted_direction_deg"],
                predicted_magnitude=prop["predicted_magnitude"],
                angular_sep_arcsec=angular_sep_arcsec,
                velocity_delta_pct=velocity_delta_pct,
                direction_delta_deg=direction_delta_deg,
                magnitude_delta=None,
                position_score=position_score,
                velocity_score=velocity_score,
                direction_score=direction_score,
                magnitude_score=magnitude_score,
                weighted_score=weighted,
            )
        )

    scored.sort(key=lambda c: c.weighted_score, reverse=True)

    # Flag ambiguity
    if len(scored) >= 2:
        if scored[0].weighted_score - scored[1].weighted_score < _AMBIGUITY_THRESHOLD:
            scored[0].ambiguous = True

    return scored


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("matcher.py: import match() and pass a StreakDetection + candidate list.")
    print("See tests/test_matcher.py for usage examples.")
