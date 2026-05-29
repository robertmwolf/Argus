"""Tests for src/matching/spacetrack_query.py.

All tests run offline using mocks.  No live Space-Track API calls are made.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.matching.spacetrack_query import (
    _GP_CURRENT_CACHE_KEY,
    _GP_CURRENT_TTL_S,
    _cache_key,
    _cache_path,
    _cache_ttl,
    _gp_current_cache_key,
    _space_track_base_url,
    _space_track_cache_namespace,
    query_gp_current,
    query_gp_history,
)

# ISS NORAD ID — always in the Space-Track catalog, good integration anchor
_ISS_NORAD = "25544"
# Historical observation window used throughout the test suite
_OBS_TIME = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)


class TestCacheKey:
    def test_development_default_uses_test_site(self, monkeypatch):
        monkeypatch.delenv("SPACETRACK_BASE_URL", raising=False)
        monkeypatch.delenv("ARGUS_ENV", raising=False)
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        assert _space_track_base_url() == "https://for-testing-only.space-track.org/"

    def test_production_env_uses_official_site(self, monkeypatch):
        monkeypatch.delenv("SPACETRACK_BASE_URL", raising=False)
        monkeypatch.setenv("ARGUS_ENV", "production")
        assert _space_track_base_url() == "https://www.space-track.org/"

    def test_base_url_override_wins(self, monkeypatch):
        monkeypatch.setenv("ARGUS_ENV", "production")
        monkeypatch.setenv("SPACETRACK_BASE_URL", "https://for-testing-only.space-track.org")
        assert _space_track_base_url() == "https://for-testing-only.space-track.org/"

    def test_base_url_must_be_https(self, monkeypatch):
        monkeypatch.setenv("SPACETRACK_BASE_URL", "http://for-testing-only.space-track.org")
        with pytest.raises(ValueError, match="HTTPS"):
            _space_track_base_url()

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

    def test_history_cache_key_includes_space_track_namespace(self, monkeypatch):
        obs = datetime(2024, 4, 2, 0, 0, 0, tzinfo=timezone.utc)

        monkeypatch.setenv("SPACETRACK_BASE_URL", "https://for-testing-only.space-track.org/")
        test_key = _cache_key(obs, 3)

        monkeypatch.setenv("SPACETRACK_BASE_URL", "https://www.space-track.org/")
        prod_key = _cache_key(obs, 3)

        assert test_key != prod_key
        assert "gp_history_test_" in test_key
        assert "gp_history_prod_" in prod_key

    def test_gp_current_cache_key_includes_space_track_namespace(self, monkeypatch):
        monkeypatch.setenv("SPACETRACK_BASE_URL", "https://for-testing-only.space-track.org/")
        assert _space_track_cache_namespace() == "test"
        assert _gp_current_cache_key() == f"{_GP_CURRENT_CACHE_KEY}_test"

        monkeypatch.setenv("SPACETRACK_BASE_URL", "https://www.space-track.org/")
        assert _space_track_cache_namespace() == "prod"
        assert _gp_current_cache_key() == f"{_GP_CURRENT_CACHE_KEY}_prod"

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
        monkeypatch.setenv("SPACETRACK_USER", "u@example.com")
        monkeypatch.setenv("SPACETRACK_PASS", "secret")
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", cache_dir)

        fake_results = [{"OBJECT_NAME": "OLD-SAT", "NORAD_CAT_ID": "99999"}]
        mock_client = MagicMock()
        mock_client.gp_history.return_value = iter(fake_results)

        with patch("src.matching.spacetrack_query.SpaceTrackClient", return_value=mock_client):
            query_gp_history(_OBS_TIME, epoch_window_days=1)

        key = _cache_key(_OBS_TIME, 1, 11.25)
        with _cache_path(key).open("r", encoding="utf-8") as handle:
            entry = json.load(handle)
        assert entry["expire_at"] is None, "Historical TLE cache entry must never expire"


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
        monkeypatch.setenv("SPACETRACK_USER", "u@example.com")
        monkeypatch.setenv("SPACETRACK_PASS", "secret")
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", cache_dir)

        fake_results = [{"OBJECT_NAME": "ISS", "NORAD_CAT_ID": "25544"}]
        mock_client = MagicMock()
        mock_client.gp.return_value = iter(fake_results)

        with patch("src.matching.spacetrack_query.SpaceTrackClient", return_value=mock_client):
            query_gp_current()

        with _cache_path(_gp_current_cache_key()).open("r", encoding="utf-8") as handle:
            entry = json.load(handle)
        expire = entry["expire_at"]
        assert expire is not None, "GP-current cache entry must have a finite TTL"
        # Expiry should be within a minute of now + 55 min
        expected = time.time() + _GP_CURRENT_TTL_S
        assert abs(expire - expected) < 60, f"Unexpected TTL expire time: {expire}"
