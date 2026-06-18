"""Tests for src/matching/tle_manager.py — TLECatalogManager.

All tests run offline using mocks.  No live CelesTrak or Space-Track API
calls are made.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch


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
                rows = manager.get_tles(_OBS_TIME_HIST, epoch_window_days=3)

        assert rows[0]["norad_id"] == _FAKE_ROWS[0]["norad_id"]
        assert rows[0]["tle_search_mode"] == "broad_epoch"
        assert rows[0]["epoch_search_window_days"] == 60
        mock_broad.assert_called_once()

    def test_historical_miss_logs_bootstrap_instruction(self, caplog):
        """A miss after broad/current fallbacks logs a clear operator diagnostic."""
        import logging
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch("src.matching.tle_manager.query_tles_for_epoch_drift", return_value=[]):
                with caplog.at_level(logging.WARNING, logger="src.matching.tle_manager"):
                    manager.get_tles(_OBS_TIME_HIST, epoch_window_days=3)

        assert any("bootstrap" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Recent observations use the same local-only lookup
# ---------------------------------------------------------------------------

class TestRecentObservations:
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

    def test_miss_returns_empty_without_network_fallback(self):
        """A recent miss remains local-only and returns an empty catalog."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch("src.matching.tle_manager.query_tles_for_epoch_drift", return_value=[]):
                rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        assert rows == []


# ---------------------------------------------------------------------------
# Datetime normalization
# ---------------------------------------------------------------------------

class TestDatetimeNormalization:
    def test_naive_obs_time_is_made_utc_aware(self):
        """A naive datetime is treated as UTC without raising an exception."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        naive = datetime(2024, 4, 2, 2, 55, 24)  # no tzinfo
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch("src.matching.tle_manager.query_tles_for_epoch_drift", return_value=[]):
                rows = manager.get_tles(naive, epoch_window_days=3)

        assert rows == []
