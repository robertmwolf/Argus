"""Tests for src/matching/spacetrack_query.py.

Unit tests run offline using mocks.
Integration tests (marked @pytest.mark.integration) make live Space-Track API
calls and require SPACETRACK_USER and SPACETRACK_PASS in the environment.
Run them with: pytest -m integration
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.matching.spacetrack_query import (
    _GP_CURRENT_CACHE_KEY,
    _GP_CURRENT_TTL_S,
    _cache_key,
    _cache_ttl,
    query_gp_current,
    query_gp_history,
)

# ISS NORAD ID — always in the Space-Track catalog, good integration anchor
_ISS_NORAD = "25544"
# Historical observation window used throughout the test suite
_OBS_TIME = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)


class TestCacheKey:
    def test_key_includes_hour(self):
        obs = datetime(2024, 4, 2, 14, 30, 0, tzinfo=timezone.utc)
        key = _cache_key(obs, epoch_window_days=3)
        assert "2024040214" in key

    def test_key_includes_window(self):
        obs = datetime(2024, 4, 2, 14, 0, 0, tzinfo=timezone.utc)
        key = _cache_key(obs, epoch_window_days=7)
        assert "7d" in key

    def test_key_has_gp_history_prefix(self):
        obs = datetime(2024, 4, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert _cache_key(obs, 3).startswith("gp_history_")

    def test_different_windows_give_different_keys(self):
        obs = datetime(2024, 4, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert _cache_key(obs, 3) != _cache_key(obs, 7)

    def test_different_mean_motion_filters_give_different_keys(self):
        obs = datetime(2024, 4, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert _cache_key(obs, 3, 11.25) != _cache_key(obs, 3, 0.0)


class TestCacheTtl:
    def test_old_observation_gets_permanent_cache(self):
        """Historical TLEs are immutable — TTL must be None (permanent)."""
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert _cache_ttl(old) is None

    def test_observation_older_than_2_days_is_permanent(self):
        obs = datetime.now(tz=timezone.utc) - timedelta(days=3)
        assert _cache_ttl(obs) is None

    def test_recent_observation_gets_finite_ttl(self):
        recent = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        assert _cache_ttl(recent) == 2 * 3600

    def test_just_over_2_days_is_permanent(self):
        obs = datetime.now(tz=timezone.utc) - timedelta(hours=49)
        assert _cache_ttl(obs) is None


class TestQueryGpHistory:
    def test_missing_user_env_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("SPACETRACK_USER", raising=False)
        monkeypatch.delenv("SPACETRACK_PASS", raising=False)
        with pytest.raises(ValueError, match="SPACETRACK_USER"):
            query_gp_history(_OBS_TIME)

    def test_missing_pass_env_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("SPACETRACK_USER", "user@example.com")
        monkeypatch.delenv("SPACETRACK_PASS", raising=False)
        with pytest.raises(ValueError, match="SPACETRACK_PASS"):
            query_gp_history(_OBS_TIME)

    def test_recent_obs_time_raises_value_error(self, monkeypatch, tmp_path):
        """query_gp_history must reject obs_time within the last 2 hours."""
        monkeypatch.setenv("SPACETRACK_USER", "u@example.com")
        monkeypatch.setenv("SPACETRACK_PASS", "secret")
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        recent = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        with pytest.raises(ValueError, match="query_gp_current"):
            query_gp_history(recent)

    def test_cache_hit_does_not_call_api(self, monkeypatch, tmp_path):
        """Second call with same key must not hit Space-Track."""
        monkeypatch.setenv("SPACETRACK_USER", "u@example.com")
        monkeypatch.setenv("SPACETRACK_PASS", "secret")
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")

        fake_results = [{"OBJECT_NAME": "SAT-1", "NORAD_CAT_ID": "12345"}]
        call_count = {"n": 0}

        def fake_gp_history(**kwargs):
            call_count["n"] += 1
            return iter(fake_results)

        mock_client = MagicMock()
        mock_client.gp_history.side_effect = fake_gp_history

        with patch("src.matching.spacetrack_query.SpaceTrackClient", return_value=mock_client):
            r1 = query_gp_history(_OBS_TIME, epoch_window_days=3)
            r2 = query_gp_history(_OBS_TIME, epoch_window_days=3)

        assert call_count["n"] == 1, "API should be called only once; second call served from cache"
        assert r1 == r2 == fake_results

    def test_historical_result_cached_permanently(self, monkeypatch, tmp_path):
        """Results for historical obs_time must be stored with no expiry."""
        import diskcache as dc

        monkeypatch.setenv("SPACETRACK_USER", "u@example.com")
        monkeypatch.setenv("SPACETRACK_PASS", "secret")
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", cache_dir)

        fake_results = [{"OBJECT_NAME": "OLD-SAT", "NORAD_CAT_ID": "99999"}]
        mock_client = MagicMock()
        mock_client.gp_history.return_value = iter(fake_results)

        with patch("src.matching.spacetrack_query.SpaceTrackClient", return_value=mock_client):
            query_gp_history(_OBS_TIME, epoch_window_days=1)

        cache = dc.Cache(str(cache_dir))
        key = _cache_key(_OBS_TIME, 1, 11.25)
        # diskcache returns (value, expire_time); expire_time is None when permanent
        _, expire = cache.get(key, expire_time=True)
        assert expire is None, "Historical TLE cache entry must never expire"


class TestQueryGpCurrent:
    def test_missing_user_env_raises_value_error(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SPACETRACK_USER", raising=False)
        monkeypatch.delenv("SPACETRACK_PASS", raising=False)
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        with pytest.raises(ValueError, match="SPACETRACK_USER"):
            query_gp_current()

    def test_cache_hit_does_not_call_api(self, monkeypatch, tmp_path):
        """Second call within 55 minutes must not hit the GP class API."""
        monkeypatch.setenv("SPACETRACK_USER", "u@example.com")
        monkeypatch.setenv("SPACETRACK_PASS", "secret")
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")

        fake_results = [{"OBJECT_NAME": "ISS", "NORAD_CAT_ID": "25544"}]
        call_count = {"n": 0}

        def fake_gp(**kwargs):
            call_count["n"] += 1
            return iter(fake_results)

        mock_client = MagicMock()
        mock_client.gp.side_effect = fake_gp

        with patch("src.matching.spacetrack_query.SpaceTrackClient", return_value=mock_client):
            r1 = query_gp_current()
            r2 = query_gp_current()

        assert call_count["n"] == 1, "GP class must not be called more than once per cache window"
        assert r1 == r2 == fake_results

    def test_gp_current_result_cached_with_correct_ttl(self, monkeypatch, tmp_path):
        """GP-current result must be cached with the 55-minute TTL."""
        import diskcache as dc

        monkeypatch.setenv("SPACETRACK_USER", "u@example.com")
        monkeypatch.setenv("SPACETRACK_PASS", "secret")
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", cache_dir)

        fake_results = [{"OBJECT_NAME": "ISS", "NORAD_CAT_ID": "25544"}]
        mock_client = MagicMock()
        mock_client.gp.return_value = iter(fake_results)

        with patch("src.matching.spacetrack_query.SpaceTrackClient", return_value=mock_client):
            query_gp_current()

        cache = dc.Cache(str(cache_dir))
        _, expire = cache.get(_GP_CURRENT_CACHE_KEY, expire_time=True)
        assert expire is not None, "GP-current cache entry must have a finite TTL"
        # Expiry should be within a minute of now + 55 min
        import time
        expected = time.time() + _GP_CURRENT_TTL_S
        assert abs(expire - expected) < 60, f"Unexpected TTL expire time: {expire}"


# ---------------------------------------------------------------------------
# Integration tests — live Space-Track API
# Require: SPACETRACK_USER and SPACETRACK_PASS set in the environment.
# Run with: pytest -m integration
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "OBJECT_NAME", "NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2",
    "EPOCH", "MEAN_MOTION", "OBJECT_TYPE",
}


@pytest.mark.integration
class TestSpaceTrackGpCurrentIntegration:
    """Live GP class tests (≤ once per hour — results cached 55 min)."""

    def test_returns_non_empty_result_set(self, spacetrack_creds, tmp_path, monkeypatch):
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_current()
        assert len(results) > 0, "GP class returned no active TLEs"

    def test_records_have_required_fields(self, spacetrack_creds, tmp_path, monkeypatch):
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_current()
        assert results, "No records returned"
        for rec in results[:20]:
            missing = _REQUIRED_FIELDS - rec.keys()
            assert not missing, f"Record missing fields: {missing}"

    def test_second_call_served_from_cache(self, spacetrack_creds, tmp_path, monkeypatch):
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        r1 = query_gp_current()
        assert r1, "First call returned no results"

        api_call_count = {"n": 0}
        original_cls = __import__(
            "src.matching.spacetrack_query", fromlist=["SpaceTrackClient"]
        ).SpaceTrackClient

        class CountingClient(original_cls):
            def gp(self, **kwargs):
                api_call_count["n"] += 1
                return super().gp(**kwargs)

        with patch("src.matching.spacetrack_query.SpaceTrackClient", CountingClient):
            r2 = query_gp_current()

        assert api_call_count["n"] == 0, (
            f"Expected 0 API calls on second request; got {api_call_count['n']}."
        )
        assert r1 == r2


@pytest.mark.integration
class TestSpaceTrackGpHistoryIntegration:
    """Live GP_History tests — one-time historical queries, cached permanently."""

    def test_authentication_succeeds(self, spacetrack_creds, tmp_path, monkeypatch):
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert isinstance(results, list)

    def test_returns_non_empty_result_set(self, spacetrack_creds, tmp_path, monkeypatch):
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert len(results) > 0

    def test_records_have_required_fields(self, spacetrack_creds, tmp_path, monkeypatch):
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert results
        for rec in results[:20]:
            missing = _REQUIRED_FIELDS - rec.keys()
            assert not missing, f"Record missing fields: {missing}\nRecord: {rec}"

    def test_tle_lines_are_non_empty_strings(self, spacetrack_creds, tmp_path, monkeypatch):
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert results
        for rec in results[:20]:
            l1 = rec.get("TLE_LINE1", "")
            l2 = rec.get("TLE_LINE2", "")
            assert l1.startswith("1 "), f"TLE_LINE1 malformed: {l1!r}"
            assert l2.startswith("2 "), f"TLE_LINE2 malformed: {l2!r}"

    def test_iss_present_in_one_day_window(self, spacetrack_creds, tmp_path, monkeypatch):
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        norad_ids = {str(r.get("NORAD_CAT_ID", "")).strip() for r in results}
        assert _ISS_NORAD in norad_ids, (
            f"ISS (NORAD {_ISS_NORAD}) not found in {len(results)} records."
        )

    def test_result_cached_permanently(self, spacetrack_creds, tmp_path, monkeypatch):
        """Verify the integration result is stored with no expiry (TTL=None)."""
        import diskcache as dc

        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        query_gp_history(_OBS_TIME, epoch_window_days=1)

        cache = dc.Cache(str(tmp_path / "cache"))
        key = _cache_key(_OBS_TIME, 1, 11.25)
        _, expire = cache.get(key, expire_time=True)
        assert expire is None, "Historical TLE cache must be permanent (TTL=None)"

    def test_second_call_served_from_cache(self, spacetrack_creds, tmp_path, monkeypatch):
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        r1 = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert r1

        api_call_count = {"n": 0}
        original_cls = __import__(
            "src.matching.spacetrack_query", fromlist=["SpaceTrackClient"]
        ).SpaceTrackClient

        class CountingClient(original_cls):
            def gp_history(self, **kwargs):
                api_call_count["n"] += 1
                return super().gp_history(**kwargs)

        with patch("src.matching.spacetrack_query.SpaceTrackClient", CountingClient):
            r2 = query_gp_history(_OBS_TIME, epoch_window_days=1)

        assert api_call_count["n"] == 0, (
            f"Expected 0 API calls on second request; got {api_call_count['n']}."
        )
        assert r1 == r2
