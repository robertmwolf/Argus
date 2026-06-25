"""Tests for src/matching/propagator.py.

All tests run offline — skyfield loads its bundled timescale data with no
network access when called with load.timescale(builtin=True) is not required
here because _timescale() calls load.timescale() which falls back to bundled
data when no Skyfield data directory is configured.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.matching.propagator import propagate


# Starlink-1007 TLE — epoch 2024-04-02 (well within SGP4 validity range).
_VALID_TLE = {
    "OBJECT_NAME": "STARLINK-1007",
    "NORAD_CAT_ID": "44713",
    "TLE_LINE1": "1 44713U 19074A   24093.50000000  .00001764  00000-0  13679-3 0  9998",
    "TLE_LINE2": "2 44713  53.0536 100.4783 0001199  86.5965 273.5324 15.06396608241234",
    "EPOCH": "2024-04-02T12:00:00",
}
_OBS_TIME = datetime(2024, 4, 2, 12, 0, 0, tzinfo=timezone.utc)
_LAT, _LON, _ALT = 43.67, -81.02, 365.0


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_returns_dict_for_valid_tle() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert isinstance(result, dict)


def test_result_contains_required_keys() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    for key in (
        "norad_id", "object_name", "tle_epoch", "tle_age_hours",
        "predicted_ra", "predicted_dec",
        "predicted_velocity_arcsec_s", "predicted_direction_deg",
        "predicted_magnitude",
    ):
        assert key in result, f"missing key: {key}"


def test_predicted_ra_in_range() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert 0.0 <= result["predicted_ra"] < 360.0


def test_predicted_dec_in_range() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert -90.0 <= result["predicted_dec"] <= 90.0


def test_norad_id_matches_input() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert result["norad_id"] == 44713


def test_object_name_matches_input() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert result["object_name"] == "STARLINK-1007"


def test_velocity_is_nonnegative() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert result["predicted_velocity_arcsec_s"] >= 0.0


def test_direction_in_range() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert 0.0 <= result["predicted_direction_deg"] < 360.0


def test_magnitude_is_none() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert result["predicted_magnitude"] is None


def test_tle_age_hours_correct() -> None:
    result = propagate(_VALID_TLE, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert result["tle_age_hours"] == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# Naive datetime (no tzinfo)
# ---------------------------------------------------------------------------


def test_naive_datetime_is_accepted() -> None:
    naive_time = datetime(2024, 4, 2, 12, 0, 0)
    result = propagate(_VALID_TLE, naive_time, _LAT, _LON, _ALT)
    assert result is not None


def test_naive_and_aware_give_same_result() -> None:
    naive = datetime(2024, 4, 2, 12, 0, 0)
    aware = datetime(2024, 4, 2, 12, 0, 0, tzinfo=timezone.utc)
    r_naive = propagate(_VALID_TLE, naive, _LAT, _LON, _ALT)
    r_aware = propagate(_VALID_TLE, aware, _LAT, _LON, _ALT)
    assert r_naive is not None and r_aware is not None
    assert r_naive["predicted_ra"] == pytest.approx(r_aware["predicted_ra"], abs=1e-6)


# ---------------------------------------------------------------------------
# Missing / malformed TLE lines
# ---------------------------------------------------------------------------


def test_missing_line1_returns_none() -> None:
    tle = {**_VALID_TLE, "TLE_LINE1": None}
    assert propagate(tle, _OBS_TIME, _LAT, _LON, _ALT) is None


def test_missing_line2_returns_none() -> None:
    tle = {**_VALID_TLE, "TLE_LINE2": None}
    assert propagate(tle, _OBS_TIME, _LAT, _LON, _ALT) is None


def test_empty_line1_returns_none() -> None:
    tle = {**_VALID_TLE, "TLE_LINE1": ""}
    assert propagate(tle, _OBS_TIME, _LAT, _LON, _ALT) is None


def test_garbled_tle_lines_produce_nan_or_none() -> None:
    # skyfield accepts malformed lines without raising; propagation yields NaN coords.
    import math
    tle = {**_VALID_TLE, "TLE_LINE1": "not a tle line", "TLE_LINE2": "also garbage"}
    result = propagate(tle, _OBS_TIME, _LAT, _LON, _ALT)
    if result is not None:
        assert math.isnan(result["predicted_ra"]) or math.isnan(result["predicted_dec"])


# ---------------------------------------------------------------------------
# EPOCH field parsing
# ---------------------------------------------------------------------------


def test_malformed_epoch_does_not_raise() -> None:
    tle = {**_VALID_TLE, "EPOCH": "not-a-date"}
    result = propagate(tle, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert result["tle_age_hours"] == pytest.approx(0.0, abs=0.1)


def test_missing_epoch_does_not_raise() -> None:
    tle = {k: v for k, v in _VALID_TLE.items() if k != "EPOCH"}
    result = propagate(tle, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None


def test_epoch_with_z_suffix_parsed() -> None:
    tle = {**_VALID_TLE, "EPOCH": "2024-04-02T12:00:00Z"}
    result = propagate(tle, _OBS_TIME, _LAT, _LON, _ALT)
    assert result is not None
    assert result["tle_age_hours"] == pytest.approx(0.0, abs=0.1)


def test_positive_tle_age_for_stale_tle() -> None:
    # Observe 24 hours after the TLE epoch.
    obs = datetime(2024, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
    result = propagate(_VALID_TLE, obs, _LAT, _LON, _ALT)
    assert result is not None
    assert result["tle_age_hours"] == pytest.approx(24.0, abs=0.01)
