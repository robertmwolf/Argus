"""Space-Track API queries with disk caching.

API policy compliance (per Space-Track operator notice):

  GP class (query_gp_current):
    - Use for the most recent TLE for each active object.
    - Call at most ONCE PER HOUR.  Results are cached for 55 minutes.
    - Time calls 10-20 minutes off the top/bottom of the hour
      (e.g. HH:12 or HH:48, never HH:00 or HH:30).
    - Recommended query:
      class/gp/decay_date/null-val/CREATION_DATE/>now-0.042/format/json

  GP_History class (query_gp_history):
    - For ONE-TIME, ad-hoc historical queries only.
    - gp_history has 220 M+ rows; do not poll it repeatedly.
    - Once downloaded, cache PERMANENTLY — historical TLEs are immutable.
    - For full-catalog date ranges use the annual zip bundles instead:
      https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu

Rate limits: 30 req/min, 300 req/hr.  Always sleep ≥3 s between requests.
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

# GP current: minimum interval between live queries (seconds).
# Space-Track policy: at most once per hour.
_GP_CURRENT_MIN_INTERVAL_S = 3600
# Cache key used to persist the GP-current result and enforce the hour lock.
_GP_CURRENT_CACHE_KEY = "gp_current_active_v1"
# Cache TTL for GP-current results (55 min — slightly under the 1-hour floor so
# that the next call always finds a slightly-stale entry and can re-fetch).
_GP_CURRENT_TTL_S = 55 * 60


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


def _cache_ttl(obs_time: datetime) -> int | None:
    """Return cache TTL in seconds, or None (permanent) for historical data.

    Historical TLE epochs are immutable — once fetched from GP_History they
    never need to be re-fetched.  Only observations within the last 48 hours
    get a finite TTL, because their TLE window might still be updated.

    Args:
        obs_time: UTC observation time.

    Returns:
        None for historical data (permanent), or seconds for recent data.
    """
    age_days = (datetime.now(tz=timezone.utc) - obs_time).total_seconds() / 86400
    if age_days > 2:
        return None  # permanent — historical TLEs are immutable
    return 2 * 3600  # 2 hours for very recent obs where the window may still change


def _cache_key(obs_time: datetime, epoch_window_days: int, min_mean_motion: float = 11.25) -> str:
    """Build deterministic cache key from obs_time, window, and motion filter.

    Args:
        obs_time: UTC observation time.
        epoch_window_days: Size of epoch search window in days.
        min_mean_motion: Minimum mean motion filter applied to the query.

    Returns:
        Cache key string.
    """
    return f"gp_history_{obs_time.strftime('%Y%m%d%H')}_{epoch_window_days}d_mm{min_mean_motion:.2f}"


def _rate_limit() -> None:
    """Sleep if needed to stay under 1 request per _RATE_LIMIT_SECONDS."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# GP class — current active TLEs (≤ once per hour)
# ---------------------------------------------------------------------------

def query_gp_current() -> list[dict]:
    """Fetch current TLEs for all active objects using the optimised GP class.

    Uses Space-Track's recommended query for the most recent element sets::

        class/gp/decay_date/null-val/CREATION_DATE/>now-0.042/format/json

    Results are cached for 55 minutes.  If this function was called less than
    one hour ago the cached result is returned without any network request,
    enforcing Space-Track's one-request-per-hour guideline.

    Best practice: schedule callers 10–20 minutes off the hour
    (e.g. HH:12 or HH:48) to avoid peak load at HH:00 and HH:30.

    Returns:
        List of TLE dicts from Space-Track.  Each dict has keys including
        OBJECT_NAME, NORAD_CAT_ID, TLE_LINE1, TLE_LINE2, EPOCH, MEAN_MOTION.

    Raises:
        ValueError: If SPACETRACK_USER or SPACETRACK_PASS are not set.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = dc.Cache(str(_CACHE_DIR))

    cached = cache.get(_GP_CURRENT_CACHE_KEY)
    if cached is not None:
        logger.debug("GP current cache hit (%d TLEs)", len(cached))
        return cached

    # Compute the creation-date cutoff: TLEs created in the last ~1 hour.
    # Matches Space-Track's recommended now-0.042 (≈ 1 hr) threshold.
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info("Querying GP class for active TLEs created since %s", cutoff_str)
    _rate_limit()
    client = _get_client()

    results = list(
        client.gp(
            decay_date="null-val",
            creation_date=op.greater_than(cutoff_str),
            orderby="norad_cat_id asc",
        )
    )

    logger.info("GP current: %d active TLEs returned", len(results))
    cache.set(_GP_CURRENT_CACHE_KEY, results, expire=_GP_CURRENT_TTL_S)
    return results


# ---------------------------------------------------------------------------
# GP_History class — one-time ad-hoc historical queries
# ---------------------------------------------------------------------------

def query_gp_history(
    obs_time: datetime,
    epoch_window_days: int = 1,
    min_mean_motion: float = 11.25,
) -> list[dict]:
    """Query Space-Track GP_History for TLEs near a historical obs_time.

    **Use this ONLY for ad-hoc historical queries.**  GP_History contains
    220 M+ rows and must not be polled repeatedly.  Results are cached
    permanently on disk (historical TLEs are immutable and never change).

    For large date ranges or full-catalog dumps, download the annual TLE zip
    bundles instead of calling this function:
    https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu

    For observations taken within the last 2 hours use :func:`query_gp_current`
    instead; it uses the optimised GP class and respects the one-hour limit.

    Searches for TLE epochs in the window ``(obs_time - epoch_window_days,
    obs_time)``.  Results are filtered to objects with
    ``mean_motion >= min_mean_motion`` (default 11.25 rev/day = LEO) to keep
    result sizes manageable.  Pass ``min_mean_motion=0`` to include all orbits
    (warning: this can return millions of records and may be rejected).

    Args:
        obs_time: UTC datetime of the observation.
        epoch_window_days: Days before obs_time to include in epoch search.
        min_mean_motion: Minimum mean motion in rev/day (11.25 = LEO only).

    Returns:
        List of TLE dicts from Space-Track.

    Raises:
        ValueError: If SPACETRACK_USER or SPACETRACK_PASS are not set.
        ValueError: If obs_time is within the last 2 hours (use query_gp_current).
    """
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    age_hours = (datetime.now(tz=timezone.utc) - obs_time).total_seconds() / 3600
    if age_hours < 2:
        raise ValueError(
            f"obs_time {obs_time.isoformat()} is only {age_hours:.1f} hours old. "
            "Use query_gp_current() for recent observations — it uses the optimised "
            "GP class and is cached for 55 minutes."
        )

    key = _cache_key(obs_time, epoch_window_days, min_mean_motion)
    ttl = _cache_ttl(obs_time)  # None = permanent for historical data

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = dc.Cache(str(_CACHE_DIR))

    cached = cache.get(key)
    if cached is not None:
        logger.debug("GP_History cache hit for key %s (%d TLEs)", key, len(cached))
        return cached

    epoch_start = obs_time - timedelta(days=epoch_window_days)
    epoch_end = obs_time

    epoch_range = op.inclusive_range(
        epoch_start.strftime("%Y-%m-%d"),
        epoch_end.strftime("%Y-%m-%d"),
    )

    logger.info(
        "Querying GP_History: epoch %s to %s (mean_motion >= %.2f) — will cache permanently",
        epoch_start.date(),
        epoch_end.date(),
        min_mean_motion,
    )

    _rate_limit()
    client = _get_client()

    query_kwargs: dict = dict(epoch=epoch_range, orderby="epoch desc")
    if min_mean_motion > 0:
        query_kwargs["mean_motion"] = op.greater_than(min_mean_motion)

    results = list(client.gp_history(**query_kwargs))

    logger.info("GP_History returned %d TLEs; caching permanently", len(results))
    cache.set(key, results, expire=ttl)  # ttl=None → permanent
    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "current":
        tles = query_gp_current()
        print(f"Active TLEs from GP class: {len(tles)}")
        for t in tles[:5]:
            print(f"  {t.get('OBJECT_NAME', 'UNKNOWN')}  NORAD={t.get('NORAD_CAT_ID')}")

    elif cmd == "history":
        if len(sys.argv) < 3:
            print("Usage: python -m src.matching.spacetrack_query history <ISO-datetime> [window_days]")
            sys.exit(1)
        obs = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)
        window = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        tles = query_gp_history(obs, epoch_window_days=window)
        print(f"GP_History TLEs returned: {len(tles)}")
        for t in tles[:5]:
            print(f"  {t.get('OBJECT_NAME', 'UNKNOWN')}  NORAD={t.get('NORAD_CAT_ID')}")

    else:
        print("Usage:")
        print("  python -m src.matching.spacetrack_query current")
        print("  python -m src.matching.spacetrack_query history <ISO-datetime> [window_days]")
        sys.exit(1)
