"""Tests for src/matching/spacetrack_query.py."""

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.matching.spacetrack_query import (
    _cache_key,
    _cache_ttl,
    query_gp_history,
)


class TestCacheKey:
    def test_key_includes_hour(self):
        obs = datetime(2024, 4, 2, 14, 30, 0, tzinfo=timezone.utc)
        key = _cache_key(obs, epoch_window_days=3)
        assert "2024040214" in key

    def test_key_includes_window(self):
        obs = datetime(2024, 4, 2, 14, 0, 0, tzinfo=timezone.utc)
        key = _cache_key(obs, epoch_window_days=7)
        assert "7d" in key

    def test_different_windows_give_different_keys(self):
        obs = datetime(2024, 4, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert _cache_key(obs, 3) != _cache_key(obs, 7)


class TestCacheTtl:
    def test_old_observation_gets_long_ttl(self):
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert _cache_ttl(old) == 48 * 3600

    def test_recent_observation_gets_short_ttl(self):
        from datetime import timedelta
        recent = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        assert _cache_ttl(recent) == 2 * 3600


class TestQueryGpHistory:
    def test_missing_user_env_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("SPACETRACK_USER", raising=False)
        monkeypatch.delenv("SPACETRACK_PASS", raising=False)
        with pytest.raises(ValueError, match="SPACETRACK_USER"):
            query_gp_history(datetime(2024, 4, 2, tzinfo=timezone.utc))

    def test_missing_pass_env_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("SPACETRACK_USER", "user@example.com")
        monkeypatch.delenv("SPACETRACK_PASS", raising=False)
        with pytest.raises(ValueError, match="SPACETRACK_PASS"):
            query_gp_history(datetime(2024, 4, 2, tzinfo=timezone.utc))

    def test_cache_hit_does_not_call_api(self, monkeypatch, tmp_path):
        """Second call with same key must not hit Space-Track."""
        monkeypatch.setenv("SPACETRACK_USER", "u@example.com")
        monkeypatch.setenv("SPACETRACK_PASS", "secret")

        fake_results = [{"OBJECT_NAME": "SAT-1", "NORAD_CAT_ID": "12345"}]

        # Patch the cache dir and the SpaceTrackClient
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")

        call_count = {"n": 0}

        def fake_gp_history(**kwargs):
            call_count["n"] += 1
            return iter(fake_results)

        mock_client = MagicMock()
        mock_client.gp_history.side_effect = fake_gp_history

        with patch("src.matching.spacetrack_query.SpaceTrackClient", return_value=mock_client):
            obs = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            r1 = query_gp_history(obs, epoch_window_days=3)
            r2 = query_gp_history(obs, epoch_window_days=3)

        assert call_count["n"] == 1, "API should be called only once; second call served from cache"
        assert r1 == r2 == fake_results
