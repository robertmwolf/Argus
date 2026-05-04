"""Tests for src/matching/spacetrack_query.py.

Unit tests run offline using mocks.
Integration tests (marked @pytest.mark.integration) make live Space-Track API
calls and require SPACETRACK_USER and SPACETRACK_PASS in the environment.
Run them with: pytest -m integration
"""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.matching.spacetrack_query import (
    _cache_key,
    _cache_ttl,
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

    def test_different_windows_give_different_keys(self):
        obs = datetime(2024, 4, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert _cache_key(obs, 3) != _cache_key(obs, 7)

    def test_different_mean_motion_filters_give_different_keys(self):
        obs = datetime(2024, 4, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert _cache_key(obs, 3, 11.25) != _cache_key(obs, 3, 0.0)


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
class TestSpaceTrackIntegration:
    """Live API tests against Space-Track GP_History.

    Each test uses an isolated tmp_path cache to avoid masking connectivity
    failures with a warm cache from a previous run.
    """

    def test_authentication_succeeds(self, spacetrack_creds, tmp_path, monkeypatch):
        """Credentials in the environment must authenticate without error."""
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        # A 1-day window around a well-known historical date returns quickly
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert isinstance(results, list)

    def test_returns_non_empty_result_set(self, spacetrack_creds, tmp_path, monkeypatch):
        """GP_History must return at least one record for the 1-day window."""
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert len(results) > 0, "Expected GP_History records; got an empty response"

    def test_records_have_required_fields(self, spacetrack_creds, tmp_path, monkeypatch):
        """Every record must carry the fields the pipeline depends on."""
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert results, "No records returned — cannot check fields"
        for rec in results[:20]:   # spot-check first 20
            missing = _REQUIRED_FIELDS - rec.keys()
            assert not missing, f"Record missing fields: {missing}\nRecord: {rec}"

    def test_tle_lines_are_non_empty_strings(self, spacetrack_creds, tmp_path, monkeypatch):
        """TLE_LINE1 and TLE_LINE2 must be non-empty strings starting with '1 ' / '2 '."""
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert results, "No records returned"
        for rec in results[:20]:
            l1 = rec.get("TLE_LINE1", "")
            l2 = rec.get("TLE_LINE2", "")
            assert l1.startswith("1 "), f"TLE_LINE1 malformed: {l1!r}"
            assert l2.startswith("2 "), f"TLE_LINE2 malformed: {l2!r}"

    def test_mean_motion_values_are_positive(self, spacetrack_creds, tmp_path, monkeypatch):
        """MEAN_MOTION (rev/day) must be a positive numeric string."""
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert results, "No records returned"
        for rec in results[:20]:
            mm = float(rec["MEAN_MOTION"])
            assert mm > 0, f"Expected positive mean motion, got {mm} for {rec['OBJECT_NAME']}"

    def test_iss_present_in_one_day_window(self, spacetrack_creds, tmp_path, monkeypatch):
        """ISS (NORAD 25544) must appear in a 1-day GP_History window."""
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        norad_ids = {str(r.get("NORAD_CAT_ID", "")).strip() for r in results}
        assert _ISS_NORAD in norad_ids, (
            f"ISS (NORAD {_ISS_NORAD}) not found in {len(results)} records. "
            "The catalog may be empty or the window too narrow."
        )

    def test_iss_tle_lines_parse_with_skyfield(self, spacetrack_creds, tmp_path, monkeypatch):
        """ISS TLE lines returned by the API must be parseable by skyfield."""
        from skyfield.api import EarthSatellite, load  # type: ignore[import]

        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        results = query_gp_history(_OBS_TIME, epoch_window_days=1)
        iss_records = [r for r in results if str(r.get("NORAD_CAT_ID", "")).strip() == _ISS_NORAD]
        assert iss_records, f"ISS not found in results (got {len(results)} records)"

        rec = iss_records[0]
        ts = load.timescale()
        sat = EarthSatellite(rec["TLE_LINE1"], rec["TLE_LINE2"], rec["OBJECT_NAME"], ts)
        assert sat is not None

    def test_second_call_served_from_cache(self, spacetrack_creds, tmp_path, monkeypatch):
        """Second call with the same obs_time must be served from disk cache,
        not from a second HTTP request to Space-Track."""
        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")

        # Prime the cache with a real API call, then verify the second call is cached
        r1 = query_gp_history(_OBS_TIME, epoch_window_days=1)
        assert r1, "First call returned no results — cannot test cache"

        api_call_count = {"n": 0}
        original_client_cls = __import__(
            "src.matching.spacetrack_query", fromlist=["SpaceTrackClient"]
        ).SpaceTrackClient

        class CountingClient(original_client_cls):
            def gp_history(self, **kwargs):
                api_call_count["n"] += 1
                return super().gp_history(**kwargs)

        with patch("src.matching.spacetrack_query.SpaceTrackClient", CountingClient):
            r2 = query_gp_history(_OBS_TIME, epoch_window_days=1)

        assert api_call_count["n"] == 0, (
            f"Expected 0 API calls on second request; got {api_call_count['n']}. "
            "Cache may not be working."
        )
        assert r1 == r2
