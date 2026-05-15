"""Tests for src/matching/tle_manager.py — TLECatalogManager.

All tests run offline using mocks.  No live CelesTrak or Space-Track API
calls are made.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

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
        """DB hit on a historical obs_time returns rows without any network call."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=_FAKE_ROWS,
        ) as mock_query:
            rows = manager.get_tles(_OBS_TIME_HIST, epoch_window_days=3)

        assert rows == _FAKE_ROWS
        mock_query.assert_called_once()

    def test_historical_miss_returns_empty_no_network(self):
        """DB miss on historical obs_time returns [] without calling CelesTrak."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=[],
        ):
            with patch("scripts.celestrak_client.fetch_and_upsert") as mock_fetch:
                rows = manager.get_tles(_OBS_TIME_HIST, epoch_window_days=3)

        assert rows == []
        mock_fetch.assert_not_called()

    def test_historical_miss_logs_bootstrap_instruction(self, caplog):
        """A miss on old data should log a clear diagnostic for the operator."""
        import logging
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with caplog.at_level(logging.WARNING, logger="src.matching.tle_manager"):
                manager.get_tles(_OBS_TIME_HIST, epoch_window_days=3)

        assert any("bootstrap" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Live track — CelesTrak refresh on miss
# ---------------------------------------------------------------------------

class TestLiveTrack:
    @staticmethod
    def _recent_time(hours_ago: float = 1.0) -> datetime:
        return datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)

    def test_live_hit_returns_rows_immediately(self):
        """DB hit on a recent obs_time returns rows without a CelesTrak call."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=_FAKE_ROWS,
        ):
            with patch("scripts.celestrak_client.fetch_and_upsert") as mock_fetch:
                rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        assert rows == _FAKE_ROWS
        mock_fetch.assert_not_called()

    def test_live_miss_cooldown_elapsed_triggers_refresh(self):
        """DB miss on a recent obs_time triggers CelesTrak if cooldown has elapsed."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        # Simulate: DB miss → CelesTrak fetch → DB now has rows
        query_side_effects = [[], _FAKE_ROWS]
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            side_effect=query_side_effects,
        ):
            with patch(
                "src.matching.tle_manager.get_last_coverage_time",
                return_value=None,  # never refreshed → cooldown elapsed
            ):
                with patch(
                    "scripts.celestrak_client.fetch_and_upsert"
                ) as mock_fetch:
                    rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        mock_fetch.assert_called_once()
        assert rows == _FAKE_ROWS

    def test_live_miss_cooldown_active_skips_refresh(self):
        """DB miss on a recent obs_time does NOT trigger CelesTrak within 2h cooldown."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        # Last refresh was 30 minutes ago — still in cooldown
        recent_refresh = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=[],
        ):
            with patch(
                "src.matching.tle_manager.get_last_coverage_time",
                return_value=recent_refresh,
            ):
                with patch(
                    "scripts.celestrak_client.fetch_and_upsert"
                ) as mock_fetch:
                    rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        mock_fetch.assert_not_called()
        assert rows == []

    def test_live_miss_celestrak_failure_returns_empty(self):
        """If CelesTrak raises an exception, get_tles returns [] without propagating."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=[],
        ):
            with patch(
                "src.matching.tle_manager.get_last_coverage_time",
                return_value=None,
            ):
                with patch(
                    "scripts.celestrak_client.fetch_and_upsert",
                    side_effect=ConnectionError("network down"),
                ):
                    rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        assert rows == []

    def test_live_miss_still_empty_after_refresh_returns_empty(self):
        """CelesTrak refresh succeeds but the object isn't in the catalog yet → []."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        with patch(
            "src.matching.tle_manager.query_tles_for_window",
            return_value=[],  # both before and after refresh
        ):
            with patch(
                "src.matching.tle_manager.get_last_coverage_time",
                return_value=None,
            ):
                with patch("scripts.celestrak_client.fetch_and_upsert"):
                    rows = manager.get_tles(self._recent_time(), epoch_window_days=1)

        assert rows == []


# ---------------------------------------------------------------------------
# Threshold boundary
# ---------------------------------------------------------------------------

class TestLiveThresholdBoundary:
    def test_exactly_at_threshold_is_historical(self):
        """obs_time exactly at live_threshold_hours old → historical path (no CelesTrak)."""
        from src.matching.tle_manager import TLECatalogManager, _LIVE_THRESHOLD_HOURS

        manager = TLECatalogManager()
        exactly_at_threshold = (
            datetime.now(tz=timezone.utc) - timedelta(hours=_LIVE_THRESHOLD_HOURS)
        )
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch("scripts.celestrak_client.fetch_and_upsert") as mock_fetch:
                manager.get_tles(exactly_at_threshold, epoch_window_days=1)

        mock_fetch.assert_not_called()

    def test_one_second_inside_live_window_triggers_refresh(self):
        """obs_time 1 s inside the live window → live path with CelesTrak eligible."""
        from src.matching.tle_manager import TLECatalogManager, _LIVE_THRESHOLD_HOURS

        manager = TLECatalogManager()
        just_inside = (
            datetime.now(tz=timezone.utc)
            - timedelta(hours=_LIVE_THRESHOLD_HOURS)
            + timedelta(seconds=1)
        )
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            with patch(
                "src.matching.tle_manager.get_last_coverage_time",
                return_value=None,
            ):
                with patch(
                    "scripts.celestrak_client.fetch_and_upsert"
                ) as mock_fetch:
                    manager.get_tles(just_inside, epoch_window_days=1)

        mock_fetch.assert_called_once()

    def test_naive_obs_time_is_made_utc_aware(self):
        """A naive datetime is treated as UTC without raising an exception."""
        from src.matching.tle_manager import TLECatalogManager

        manager = TLECatalogManager()
        naive = datetime(2024, 4, 2, 2, 55, 24)  # no tzinfo
        with patch("src.matching.tle_manager.query_tles_for_window", return_value=[]):
            rows = manager.get_tles(naive, epoch_window_days=3)

        assert rows == []
