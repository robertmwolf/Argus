"""Scoring utilities for the ARGUS satellite matching pipeline.

Provides per-factor Gaussian scoring, TLE age penalty, and weighted
score aggregation as specified in architecture.md.
"""

from __future__ import annotations

import math
# Weights for weighted_score aggregation
_WEIGHT_POSITION = 0.35
_WEIGHT_VELOCITY = 0.30
_WEIGHT_DIRECTION = 0.25
_WEIGHT_MAGNITUDE = 0.10


def gaussian_score(delta: float, sigma: float) -> float:
    """Score from 0–1 using Gaussian falloff from zero error.

    Returns 1.0 when delta is 0, decays toward 0 as delta grows.

    Args:
        delta: The absolute error (e.g. angular separation, velocity delta).
        sigma: The characteristic scale; score is ~0.61 at delta=sigma.

    Returns:
        Score in [0.0, 1.0].
    """
    return math.exp(-0.5 * (delta / sigma) ** 2)


def tle_age_penalty(
    age_hours: float,
    sigma_hours: float = 24.0,
) -> float:
    """Gaussian decay penalty for stale TLEs.

    # Source: ARGUS architecture — TLE age penalty for position score
    # Ref: agent_docs/architecture.md

    Default sigma=24h (normal fresh TLE):
      < 6h  → > 0.96 (essentially full score)
      24h   → 0.61
      48h   → 0.14
      72h   → 0.01

    For broad-epoch fallback TLEs pass sigma_hours=168 (7 days):
      24h   → 1.00
      72h   → 0.96
      168h  → 0.61
      360h  → 0.05

    Args:
        age_hours: Hours elapsed since TLE epoch to observation time.
        sigma_hours: Gaussian sigma controlling how fast the penalty decays.
            Use the default (24h) for normal-mode TLEs.  Pass a wider value
            (e.g. 168h) for broad-epoch fallback TLEs where a stale epoch is
            expected and positional geometry should still contribute confidence.

    Returns:
        Penalty multiplier in (0, 1].
    """
    return math.exp(-0.5 * (age_hours / sigma_hours) ** 2)


def aggregate_score(
    position_score: float,
    velocity_score: float,
    direction_score: float,
    magnitude_score: float,
    tle_age_hours: float,
) -> float:
    """Compute weighted aggregated confidence score.

    # Source: ARGUS architecture — weighted scoring formula
    # Ref: agent_docs/architecture.md

    Formula:
        weighted = 0.35 * position * age_penalty
                 + 0.30 * velocity
                 + 0.25 * direction
                 + 0.10 * magnitude

    Args:
        position_score: Gaussian score for angular separation (0–1).
        velocity_score: Gaussian score for velocity match (0–1).
        direction_score: Gaussian score for position angle match (0–1).
        magnitude_score: Gaussian score for brightness match (0–1).
            Pass 0.5 when magnitude is unavailable (neutral).
        tle_age_hours: Hours between TLE epoch and observation time.

    Returns:
        Weighted score in [0.0, 1.0].
    """
    penalty = tle_age_penalty(tle_age_hours)
    return (
        _WEIGHT_POSITION * position_score * penalty
        + _WEIGHT_VELOCITY * velocity_score
        + _WEIGHT_DIRECTION * direction_score
        + _WEIGHT_MAGNITUDE * magnitude_score
    )


if __name__ == "__main__":
    print("Gaussian score at 0:       ", gaussian_score(0.0, sigma=1.0))
    print("Gaussian score at 1*sigma: ", gaussian_score(1.0, sigma=1.0))
    print("Gaussian score at 2*sigma: ", gaussian_score(2.0, sigma=1.0))
    print()
    print("TLE age penalty at  0h:  ", tle_age_penalty(0.0))
    print("TLE age penalty at  6h:  ", tle_age_penalty(6.0))
    print("TLE age penalty at 24h:  ", tle_age_penalty(24.0))
    print("TLE age penalty at 72h:  ", tle_age_penalty(72.0))
    print()
    score = aggregate_score(0.9, 0.8, 0.85, 0.5, tle_age_hours=12.0)
    print(f"Example aggregate score: {score:.4f}")
