"""TLE catalog manager for ARGUS inference.

Provides :class:`TLECatalogManager`, the single entry point for retrieving TLE
records during cross-identification.  It implements the two-track lookup policy:

Live track (obs_time within the last 72 hours)
    1. Check the local ``tle_catalog`` table.
    2. On a miss, trigger a CelesTrak refresh if the 2-hour cooldown has elapsed,
       then re-check.

Historical track (obs_time older than 72 hours)
    1. Check the local ``tle_catalog`` table only.
    2. On a miss, log a diagnostic and return empty — operator must have
       bootstrapped the relevant year's zip bundle via
       ``scripts/bootstrap_tle_catalog.py``.

Space-Track ``gp_history`` is **never** called at runtime.  It is a one-time
download resource (per Space-Track API policy: 1 query per lifetime per object)
and must only be used via explicit operator/admin tools.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.matching.tle_store import get_last_coverage_time, query_tles_for_window

logger = logging.getLogger(__name__)

# Observations newer than this are treated as "live" and eligible for a
# CelesTrak refresh on a cache miss.  Beyond this window, current TLEs have
# drifted too far for LEO objects (72 h ≈ MAX_TLE_AGE_HOURS for LEO).
_LIVE_THRESHOLD_HOURS = 72.0

_CELESTRAK_COVERAGE_TAG = "celestrak_refresh"


class TLECatalogManager:
    """Two-track TLE catalog lookup with automatic live-edge refresh.

    Attributes:
        live_threshold_hours: Observations newer than this are eligible for a
            CelesTrak refresh on a cache miss.
    """

    def __init__(self, live_threshold_hours: float = _LIVE_THRESHOLD_HOURS) -> None:
        self.live_threshold_hours = live_threshold_hours

    def get_tles(
        self,
        obs_time: datetime,
        epoch_window_days: int = 3,
        min_mean_motion: float = 11.25,
    ) -> list[dict[str, Any]]:
        """Return TLE records for *obs_time*, refreshing CelesTrak if needed.

        Args:
            obs_time: UTC observation time from the FITS header.
            epoch_window_days: How many days before obs_time to search.
            min_mean_motion: Minimum mean_motion in rev/day (11.25 = LEO only).
                Pass 0 to include all orbit classes.

        Returns:
            List of TLE row dicts with keys: norad_id, epoch, object_name,
            object_type, mean_motion, tle_line1, tle_line2, source.
        """
        if obs_time.tzinfo is None:
            obs_time = obs_time.replace(tzinfo=timezone.utc)

        rows = query_tles_for_window(obs_time, epoch_window_days, min_mean_motion)
        if rows:
            logger.debug(
                "TLE catalog hit: %d records for obs_time=%s",
                len(rows), obs_time.isoformat(),
            )
            return rows

        age_hours = (datetime.now(tz=timezone.utc) - obs_time).total_seconds() / 3600

        if age_hours < self.live_threshold_hours:
            rows = self._try_celestrak_refresh(obs_time, epoch_window_days, min_mean_motion)
        else:
            logger.warning(
                "No local TLE coverage for historical obs_time=%s (age=%.1fh). "
                "Run: python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ "
                "--years %d",
                obs_time.isoformat(),
                age_hours,
                obs_time.year,
            )

        return rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_celestrak_refresh(
        self,
        obs_time: datetime,
        epoch_window_days: int,
        min_mean_motion: float,
    ) -> list[dict[str, Any]]:
        """Attempt a CelesTrak refresh and re-query the local DB.

        Respects the 2-hour cooldown encoded in :mod:`scripts.celestrak_client`.
        On any network failure, logs a warning and returns empty rather than
        propagating the exception into the inference path.

        Args:
            obs_time: UTC observation time (for the re-query after refresh).
            epoch_window_days: Passed through to :func:`query_tles_for_window`.
            min_mean_motion: Passed through to :func:`query_tles_for_window`.

        Returns:
            TLE rows after the refresh attempt (may still be empty).
        """
        last = get_last_coverage_time(_CELESTRAK_COVERAGE_TAG)
        if last is not None:
            cooldown_h = (datetime.now(tz=timezone.utc) - last).total_seconds() / 3600
            if cooldown_h < 2.0:
                logger.debug(
                    "TLE miss (live) — CelesTrak cooldown active (%.1fh remaining); "
                    "returning empty for obs_time=%s",
                    2.0 - cooldown_h,
                    obs_time.isoformat(),
                )
                return []

        logger.info(
            "TLE miss (live) — triggering CelesTrak refresh for obs_time=%s",
            obs_time.isoformat(),
        )
        try:
            from scripts.celestrak_client import fetch_and_upsert
            fetch_and_upsert(force=False)
        except Exception as exc:
            logger.warning("CelesTrak refresh failed: %s", exc)
            return []

        rows = query_tles_for_window(obs_time, epoch_window_days, min_mean_motion)
        logger.debug(
            "Post-refresh query: %d records for obs_time=%s",
            len(rows), obs_time.isoformat(),
        )
        return rows


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import timedelta

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    manager = TLECatalogManager()

    # Historical query (no network call expected)
    hist_time = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)
    hist_rows = manager.get_tles(hist_time, epoch_window_days=3)
    print(f"Historical query (2024-04-02): {len(hist_rows)} records")

    # Live query (may trigger CelesTrak refresh)
    live_time = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    live_rows = manager.get_tles(live_time, epoch_window_days=1)
    print(f"Live query (now-1h): {len(live_rows)} records")

    sys.exit(0)
