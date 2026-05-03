"""Space-Track GP_History query with disk caching.

Queries the Space-Track API for TLE data near a given observation time,
caches results to disk, and enforces rate limiting.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import diskcache as dc
import spacetrack.operators as op
from spacetrack import SpaceTrackClient

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")
_RATE_LIMIT_SECONDS = 3.0
_last_request_time: float = 0.0


def _get_client() -> SpaceTrackClient:
    """Build authenticated SpaceTrackClient from environment variables.

    Raises:
        ValueError: If SPACETRACK_USER or SPACETRACK_PASS is not set.
    """
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")
    if not user:
        raise ValueError(
            "SPACETRACK_USER environment variable is not set. "
            "Export your Space-Track email before running."
        )
    if not password:
        raise ValueError(
            "SPACETRACK_PASS environment variable is not set. "
            "Export your Space-Track password before running."
        )
    return SpaceTrackClient(identity=user, password=password)


def _cache_ttl(obs_time: datetime) -> int:
    """Return cache TTL in seconds based on how old the observation is.

    Args:
        obs_time: UTC observation time.

    Returns:
        TTL in seconds: 48h for historical (>7 days old), 2h for recent.
    """
    age_days = (datetime.now(tz=timezone.utc) - obs_time).total_seconds() / 86400
    if age_days > 7:
        return 48 * 3600
    return 2 * 3600


def _cache_key(obs_time: datetime, epoch_window_days: int) -> str:
    """Build deterministic cache key from obs_time and window.

    Args:
        obs_time: UTC observation time.
        epoch_window_days: Size of epoch search window in days.

    Returns:
        Cache key string.
    """
    return f"{obs_time.strftime('%Y%m%d%H')}_{epoch_window_days}d"


def _rate_limit() -> None:
    """Sleep if needed to stay under 1 request per _RATE_LIMIT_SECONDS."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def query_gp_history(
    obs_time: datetime,
    epoch_window_days: int = 3,
) -> list[dict]:
    """Query Space-Track GP_History for TLEs near obs_time.

    Searches for TLE epochs in the window (obs_time - epoch_window_days, obs_time).
    Results are sorted by epoch descending (most recent TLE per object first).
    Results are cached to disk using diskcache.

    Args:
        obs_time: UTC datetime of the observation.
        epoch_window_days: Days before obs_time to include in epoch search.

    Returns:
        List of raw TLE dicts from Space-Track JSON response.
        Each dict has keys including: OBJECT_NAME, NORAD_CAT_ID,
        TLE_LINE1, TLE_LINE2, EPOCH.

    Raises:
        ValueError: If SPACETRACK_USER or SPACETRACK_PASS are not set.
    """
    # Ensure obs_time is timezone-aware
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    key = _cache_key(obs_time, epoch_window_days)
    ttl = _cache_ttl(obs_time)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = dc.Cache(str(_CACHE_DIR))

    cached = cache.get(key)
    if cached is not None:
        logger.debug("Cache hit for key %s (%d TLEs)", key, len(cached))
        return cached

    epoch_start = obs_time - timedelta(days=epoch_window_days)
    epoch_end = obs_time

    epoch_range = op.inclusive_range(
        epoch_start.strftime("%Y-%m-%d"),
        epoch_end.strftime("%Y-%m-%d"),
    )

    logger.info(
        "Querying GP_History: epoch %s to %s",
        epoch_start.date(),
        epoch_end.date(),
    )

    _rate_limit()
    client = _get_client()
    results = list(
        client.gp_history(
            epoch=epoch_range,
            orderby="epoch desc",
            format="json",
        )
    )

    logger.info("Got %d TLEs from Space-Track", len(results))
    cache.set(key, results, expire=ttl)
    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m src.matching.spacetrack_query <ISO-datetime>")
        print("  e.g. 2024-04-02T02:55:24")
        sys.exit(1)

    obs = datetime.fromisoformat(sys.argv[1])
    if obs.tzinfo is None:
        obs = obs.replace(tzinfo=timezone.utc)

    window = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    tles = query_gp_history(obs, epoch_window_days=window)
    print(f"Total TLEs returned: {len(tles)}")
    print("First 5 objects:")
    for t in tles[:5]:
        print(f"  {t.get('OBJECT_NAME', 'UNKNOWN')}  NORAD={t.get('NORAD_CAT_ID')}")
