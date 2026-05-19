"""Tests for src/matching/tle_manager.py — TLECatalogManager.

All tests run offline using mocks.  No live CelesTrak or Space-Track API
calls are made.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

_OBS_TIME_HIST = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)

_FAKE_ROWS = [
    {
        "norad_id": 44713,
        "epoch": "2024-04-01T12:00:00Z",
        "object_name": "STARLINK-1007",
        "object_type": "PAYLOAD",
        "mean_motion": 15.06,
        "tle_line1": "1 44713U 19074A   24093.50000000  .00001764  00000-0  13679-3 0  9998",
        "tle_line2": "2 44713  53.0536 100.4783 0001199  86.5965 273.5324 15.06396608241234",
        "source": "celestrak",
    }
]


# ---------------------------------------------------------------------------
# Historical track — local DB only
# ---------------------------------------------------------------------------

class TestHistoricalTrack:
    def test_historical_hit_returns_rows_immediately(self):
        """DB hit on a historical obs_time returns annotated rows immediately."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=_FAKE_ROWS * 120,
        ) as mock_query:
            rows = manager.get_tles(_OBS_TIME_HIST, epoch_window_days=3)

        assert rows[0]["norad_id"] == _FAKE_ROWS[0]["norad_id"]
        assert rows[0]["tle_search_mode"] == "normal"
        assert rows[0]["epoch_search_window_days"] == 3
        mock_query.assert_called_once()

    def test_small_normal_hit_uses_broad_epoch_window(self):
        """A tiny normal-window catalog is treated as insufficient coverage."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=_FAKE_ROWS,
        ):
            with patch(
                "src.matching.tle_manager.query_tles_for_epoch_drift",
                return_value=_FAKE_ROWS * 120,
            ):
                rows = manager.get_tles(_OBS_TIME_HIST, epoch_window_days=3)

        assert len(rows) == 120
        assert rows[0]["tle_search_mode"] == "broad_epoch"

    def test_historical_miss_uses_broad_epoch_window(self):
        """DB miss falls back to a broad epoch search before current data."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=[],
        ):
            with patch(
                "src.matching.tle_manager.query_tles_for_epoch_drift",
                return_value=_FAKE_ROWS,
            ) as mock_broad:
                with patch("src.matching.tle_manager.query_latest_tles") as mock_latest:
                    rows = manager.get_tles(_OBS_TIME_HIST, epoch_window_days=3)

        assert rows[0]["norad_id"] == _FAKE_ROWS[0]["norad_id"]
        assert rows[0]["tle_search_mode"] == "broad_epoch"
        assert rows[0]["epoch_search_window_days"] == 30
        mock_broad.assert_called_once()
        mock_latest.assert_not_called()

    def test_historical_miss_logs_bootstrap_instruction(self, caplog):
        """A miss after broad/current fallbacks logs a clear operator diagnostic."""
        import logging
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch("src.matching.tle_manager.query_tles_for_epoch_drift", return_value=[]):
                with patch("src.matching.tle_manager.query_latest_tles", return_value=[]):
                    with patch.object(manager, "_refresh_current_catalog"):
                        with patch("src.matching.tle_manager.get_latest_coverage_time", return_value=datetime.now(tz=timezone.utc)):
                            with caplog.at_level(logging.WARNING, logger="src.matching.tle_manager"):
                                manager.get_tles(_OBS_TIME_HIST, epoch_window_days=3)

        assert any("bootstrap" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Live track — broad/current fallbacks on miss
# ---------------------------------------------------------------------------

class TestLiveTrack:
    @staticmethod
    def _recent_time(hours_ago: float = 1.0) -> datetime:
        return datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)

    def test_live_hit_returns_rows_immediately(self):
        """DB hit on a recent obs_time returns annotated rows immediately."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=_FAKE_ROWS * 120,
        ):
            rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        assert rows[0]["norad_id"] == _FAKE_ROWS[0]["norad_id"]
        assert rows[0]["tle_search_mode"] == "normal"

    def test_miss_with_stale_current_data_triggers_refresh_and_uses_latest(self):
        """If broad search misses and fresh data is stale, current catalog refresh runs."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=[],
        ):
            with patch("src.matching.tle_manager.query_tles_for_epoch_drift", return_value=[]):
                with patch("src.matching.tle_manager.get_latest_coverage_time", return_value=None):
                    with patch.object(manager, "_refresh_current_catalog") as mock_refresh:
                        with patch("src.matching.tle_manager.query_latest_tles", return_value=_FAKE_ROWS):
                            rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        mock_refresh.assert_called_once()
        assert rows[0]["tle_search_mode"] == "current_fallback"

    def test_miss_with_fresh_current_data_skips_refresh_but_uses_latest(self):
        """Fresh current data is reused without a network refresh."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        recent_refresh = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch("src.matching.tle_manager.query_tles_for_epoch_drift", return_value=[]):
                with patch("src.matching.tle_manager.get_latest_coverage_time", return_value=recent_refresh):
                    with patch.object(manager, "_refresh_current_catalog") as mock_refresh:
                        with patch("src.matching.tle_manager.query_latest_tles", return_value=_FAKE_ROWS):
                            rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        mock_refresh.assert_not_called()
        assert rows[0]["tle_search_mode"] == "current_fallback"

    def test_miss_still_empty_after_refresh_returns_empty(self):
        """If all fallback sources are empty, get_tles returns []."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch("src.matching.tle_manager.query_tles_for_epoch_drift", return_value=[]):
                with patch("src.matching.tle_manager.get_latest_coverage_time", return_value=None):
                    with patch.object(manager, "_refresh_current_catalog"):
                        with patch("src.matching.tle_manager.query_latest_tles", return_value=[]):
                            rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        assert rows == []


# ---------------------------------------------------------------------------
# Threshold boundary
# ---------------------------------------------------------------------------

class TestLiveThresholdBoundary:
    def test_exactly_at_threshold_is_historical(self):
        """obs_time exactly at live threshold still follows safe fallback path."""
        from src.matching.tle_manager import TLECatalogManager, _LIVE_THRESHOLD_HOURS

        manager = TLECatalogManager()
        exactly_at_threshold = (
            datetime.now(tz=timezone.utc) - timedelta(hours=_LIVE_THRESHOLD_HOURS)
        )
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch("src.matching.tle_manager.query_tles_for_epoch_drift", return_value=[]):
                with patch("src.matching.tle_manager.get_latest_coverage_time", return_value=None):
                    with patch("src.matching.tle_manager.query_latest_tles", return_value=[]):
                        with patch.object(manager, "_refresh_current_catalog") as mock_refresh:
                            manager.get_tles(exactly_at_threshold, epoch_window_days=1)

        mock_refresh.assert_called_once()

    def test_naive_obs_time_is_made_utc_aware(self):
        """A naive datetime is treated as UTC without raising an exception."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        naive = datetime(2024, 4, 2, 2, 55, 24)  # no tzinfo
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch("src.matching.tle_manager.query_tles_for_epoch_drift", return_value=[]):
                with patch("src.matching.tle_manager.query_latest_tles", return_value=[]):
                    with patch.object(manager, "_refresh_current_catalog"):
                        rows = manager.get_tles(naive, epoch_window_days=3)

        assert rows == []
