"""TLE catalog manager for ARGUS inference.

Provides :class:`TLECatalogManager`, the single entry point for retrieving TLE
records during cross-identification.

Inference path (get_tles) — DB only, no network calls ever:
    1. Normal epoch window query (epoch_window_days around obs_time).
    2. On insufficient coverage, widen to the broad epoch window (default ±60 d).
    3. On a miss, log a diagnostic and return empty.

Catalog refresh is handled entirely by explicit operator tasks outside the
inference path (for example, ``scripts/update_tle_catalog.py``).

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
    get_latest_coverage_time,
    query_tles_for_epoch_drift,
    query_tles_for_window,
)

logger = logging.getLogger(__name__)

_SPACETRACK_GP_COVERAGE_TAG = "spacetrack_gp_current"
_CURRENT_REFRESH_TAGS = [_SPACETRACK_GP_COVERAGE_TAG, "celestrak_refresh", "gp_current"]
_BROAD_EPOCH_WINDOW_DAYS = 60
_MIN_NORMAL_CANDIDATES = 100


class TLECatalogManager:
    """Local-only TLE catalog lookup for inference."""

    def get_tles(
        self,
        obs_time: datetime,
        epoch_window_days: int = 3,
        min_mean_motion: float = 0,
    ) -> list[dict[str, Any]]:
        """Return locally stored TLE records for *obs_time*.

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
