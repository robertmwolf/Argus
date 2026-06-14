"""TLE catalog manager for ARGUS inference.

Provides :class:`TLECatalogManager`, the single entry point for retrieving TLE
records during cross-identification.

Inference path (get_tles) — DB only, no network calls ever:
    1. Normal epoch window query (epoch_window_days around obs_time).
    2. On insufficient coverage, widen to the broad epoch window (default ±60 d).
    3. On a miss, log a diagnostic and return empty.

Catalog refresh is handled entirely by scheduled tasks outside the inference
path (e.g. scripts/celestrak_client.py run by a cron/launchd job).  The
helper methods get_current_fallback_tles, _refresh_current_catalog, and
_try_celestrak_refresh exist for use by those tasks only and are never called
from get_tles.

Space-Track ``gp_history`` is **never** called at runtime.  It is a one-time
download resource (per Space-Track API policy) and must only be used via
explicit operator/admin tools such as scripts/bootstrap_tle_catalog.py.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.matching.tle_store import (
    get_last_coverage_time,
    get_latest_coverage_time,
    query_latest_tles,
    query_tles_for_epoch_drift,
    query_tles_for_window,
    record_coverage,
    upsert_tles,
)

logger = logging.getLogger(__name__)

# Observations newer than this are treated as "live" and eligible for a
# CelesTrak refresh on a cache miss.  Beyond this window, current TLEs have
# drifted too far for LEO objects (72 h ≈ MAX_TLE_AGE_HOURS for LEO).
_LIVE_THRESHOLD_HOURS = 72.0

_CELESTRAK_COVERAGE_TAG = "celestrak_refresh"
_SPACETRACK_GP_COVERAGE_TAG = "spacetrack_gp_current"
_CURRENT_REFRESH_TAGS = [_SPACETRACK_GP_COVERAGE_TAG, _CELESTRAK_COVERAGE_TAG, "gp_current"]
_BROAD_EPOCH_WINDOW_DAYS = 60
_CURRENT_REFRESH_MAX_AGE_MINUTES = 45.0
_MIN_NORMAL_CANDIDATES = 100
_PRODUCTION_ENV_NAMES = {"prod", "production"}


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
        min_mean_motion: float = 0,
    ) -> list[dict[str, Any]]:
        """Return TLE records for *obs_time*, refreshing CelesTrak if needed.

        Args:
            obs_time: UTC observation time from the FITS header.
            epoch_window_days: How many days before obs_time to search.
            min_mean_motion: Minimum mean_motion in rev/day (0 = all orbit
                classes including GEO; 11.25 = LEO only).

        Returns:
            List of TLE row dicts with keys: norad_id, epoch, object_name,
            object_type, mean_motion, tle_line1, tle_line2, source.
        """
        if obs_time.tzinfo is None:
            obs_time = obs_time.replace(tzinfo=timezone.utc)

        rows = query_tles_for_window(obs_time, epoch_window_days, min_mean_motion)
        min_normal_candidates = int(
            os.environ.get("ARGUS_MIN_NORMAL_TLE_CANDIDATES", _MIN_NORMAL_CANDIDATES)
        )
        if len(rows) >= min_normal_candidates:
            logger.debug(
                "TLE catalog hit: %d records for obs_time=%s",
                len(rows), obs_time.isoformat(),
            )
            return self._annotate_rows(rows, obs_time, epoch_window_days, "normal")
        if rows:
            logger.info(
                "TLE catalog normal-window hit only %d records for obs_time=%s; "
                "treating as insufficient coverage and trying broad epoch window",
                len(rows),
                obs_time.isoformat(),
            )

        broad_window_days = max(
            epoch_window_days,
            int(os.environ.get("ARGUS_BROAD_TLE_WINDOW_DAYS", _BROAD_EPOCH_WINDOW_DAYS)),
        )
        if broad_window_days > epoch_window_days:
            broad_rows = query_tles_for_epoch_drift(
                obs_time,
                broad_window_days,
                min_mean_motion,
            )
            if broad_rows:
                logger.info(
                    "TLE catalog broad-window hit: %d records for obs_time=%s "
                    "(±%dd)",
                    len(broad_rows),
                    obs_time.isoformat(),
                    broad_window_days,
                )
                return self._annotate_rows(
                    broad_rows,
                    obs_time,
                    broad_window_days,
                    "broad_epoch",
                )

        age_hours = (datetime.now(tz=timezone.utc) - obs_time).total_seconds() / 3600
        logger.warning(
            "No local TLE coverage for obs_time=%s (age=%.1fh, broad_window=±%dd). "
            "Run: python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ "
            "--years %d  (or ensure the scheduled catalog refresh has run recently)",
            obs_time.isoformat(),
            age_hours,
            broad_window_days,
            obs_time.year,
        )
        return rows

    def get_current_fallback_tles(
        self,
        obs_time: datetime,
        min_mean_motion: float = 0,
    ) -> list[dict[str, Any]]:
        """Return latest current TLEs, refreshing if the catalog is stale.

        This is used after the broad historical window produces no viable
        propagated match.  In development the refresh source is Space-Track's
        test GP endpoint; production uses the configured current-catalog path.

        Args:
            obs_time: UTC observation time from the FITS header.
            min_mean_motion: Minimum mean_motion in rev/day.

        Returns:
            Current fallback TLE rows annotated with search metadata.
        """
        if obs_time.tzinfo is None:
            obs_time = obs_time.replace(tzinfo=timezone.utc)

        fresh_at = self._fresh_data_timestamp()
        freshness_minutes = (
            (datetime.now(tz=timezone.utc) - fresh_at).total_seconds() / 60.0
            if fresh_at is not None else float("inf")
        )

        if freshness_minutes > _CURRENT_REFRESH_MAX_AGE_MINUTES:
            self._refresh_current_catalog(obs_time)
            fresh_at = self._fresh_data_timestamp()
            freshness_minutes = (
                (datetime.now(tz=timezone.utc) - fresh_at).total_seconds() / 60.0
                if fresh_at is not None else float("inf")
            )

        current_rows = query_latest_tles(min_mean_motion)
        if not current_rows:
            return []

        logger.info(
            "Using latest current TLE catalog as fallback for obs_time=%s "
            "(%d records; freshness=%.1f min)",
            obs_time.isoformat(),
            len(current_rows),
            freshness_minutes,
        )
        rows = self._annotate_rows(current_rows, obs_time, 0, "current_fallback")
        fresh_iso = fresh_at.isoformat().replace("+00:00", "Z") if fresh_at else None
        for row in rows:
            row["tle_data_fresh_at"] = fresh_iso
        return rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_development_env() -> bool:
        """Return True when current-data refreshes should use Space-Track test."""
        env_name = (
            os.environ.get("ARGUS_ENV")
            or os.environ.get("APP_ENV")
            or os.environ.get("ENVIRONMENT")
            or "development"
        ).strip().lower()
        return env_name not in _PRODUCTION_ENV_NAMES

    @staticmethod
    def _parse_epoch(value: Any) -> datetime | None:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def _annotate_rows(
        self,
        rows: list[dict[str, Any]],
        obs_time: datetime,
        search_window_days: int,
        mode: str,
    ) -> list[dict[str, Any]]:
        """Attach search-mode metadata used by cross-ID scoring and UI.

        TLEs whose epoch is after obs_time are tagged "forward_epoch" regardless
        of the caller-supplied mode.  The crossid epoch penalty uses a wider sigma
        for forward_epoch so that back-propagating from a recently-published TLE
        (e.g., epoch 4 days after the shot) is not crushed to near-zero.
        """
        fresh_at = self._fresh_data_timestamp()
        fresh_iso = fresh_at.isoformat().replace("+00:00", "Z") if fresh_at else None
        annotated: list[dict[str, Any]] = []
        for row in rows:
            epoch_dt = self._parse_epoch(row.get("epoch"))
            drift_h = (
                abs((obs_time - epoch_dt).total_seconds()) / 3600.0
                if epoch_dt is not None else None
            )
            row_mode = mode
            if epoch_dt is not None and epoch_dt > obs_time:
                row_mode = "forward_epoch"
            annotated.append({
                **row,
                "epoch_drift_hours": drift_h,
                "epoch_search_window_days": search_window_days,
                "tle_search_mode": row_mode,
                "tle_data_fresh_at": fresh_iso,
            })
        return annotated

    def _fresh_data_timestamp(self) -> datetime | None:
        """Return the newest timestamp for current-catalog refresh sources."""
        return (
            get_latest_coverage_time(_CURRENT_REFRESH_TAGS, min_record_count=1)
            or get_latest_coverage_time(min_record_count=1)
        )

    def _refresh_current_catalog(self, obs_time: datetime) -> None:
        """Refresh current TLE data, using Space-Track test in development."""
        if self._is_development_env():
            logger.info(
                "Current TLE data is stale or absent — refreshing via Space-Track GP "
                "test API for obs_time=%s",
                obs_time.isoformat(),
            )
            try:
                from src.matching.spacetrack_query import query_gp_current
                records = query_gp_current()
                inserted = upsert_tles(records, source="spacetrack_gp")
                if records:
                    record_coverage(
                        _SPACETRACK_GP_COVERAGE_TAG,
                        "Space-Track GP current refresh (development/test API)",
                        len(records),
                    )
                logger.info(
                    "Space-Track GP current refresh returned %d records (%d inserted)",
                    len(records),
                    inserted,
                )
            except Exception as exc:
                logger.warning("Space-Track GP current refresh failed: %s", exc)
            return

        logger.info(
            "Current TLE data is stale or absent — refreshing via production current catalog"
        )
        try:
            from scripts.celestrak_client import fetch_and_upsert
            fetch_and_upsert(force=False)
        except Exception as exc:
            logger.warning("Current catalog refresh failed: %s", exc)

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
