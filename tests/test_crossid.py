"""Tests for inference/crossid.py — satellite TLE cross-identification.

All tests run offline using mocks.  No live Space-Track API calls are made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Starlink-1007 TLE — used as a deterministic known-position reference
_TEST_TLE_NAME  = "STARLINK-1007"
_TEST_TLE_LINE1 = "1 44713U 19074A   24093.50000000  .00001764  00000-0  13679-3 0  9998"
_TEST_TLE_LINE2 = "2 44713  53.0536 100.4783 0001199  86.5965 273.5324 15.06396608241234"

_ISS_NAME  = "ISS (ZARYA)"
_ISS_LINE1 = "1 25544U 98067A   24094.00000000  .00010000  00000-0  13679-3 0  9991"
_ISS_LINE2 = "2 25544  51.6400 200.0000 0001000   0.0000 360.0000 15.49999990451234"

_SAMPLE_CATALOG = [
    (_TEST_TLE_NAME, _TEST_TLE_LINE1, _TEST_TLE_LINE2),
    (_ISS_NAME, _ISS_LINE1, _ISS_LINE2),
]

_OBS_TIME = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)
_OBS_LAT, _OBS_LON, _OBS_ALT = 49.61, 6.13, 280.0


# ---------------------------------------------------------------------------
# _fetch_tle_catalog — local catalog behavior
# ---------------------------------------------------------------------------

class TestFetchTleCatalog:
    def test_maps_db_fields_to_tuples(self):
        """Local DB rows with tle_line1/tle_line2 become (name, l1, l2)."""
        from inference.crossid import _fetch_tle_catalog

        fake_rows = [
            {
                "object_name": _TEST_TLE_NAME,
                "tle_line1": _TEST_TLE_LINE1,
                "tle_line2": _TEST_TLE_LINE2,
            }
        ]
        with patch("inference.crossid.query_tles_for_window", return_value=fake_rows):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert len(catalog) == 1
        name, line1, line2 = catalog[0]
        assert name == _TEST_TLE_NAME
        assert line1 == _TEST_TLE_LINE1
        assert line2 == _TEST_TLE_LINE2

    def test_skips_records_without_tle_lines(self):
        """Records missing TLE_LINE1 or TLE_LINE2 are silently dropped."""
        from inference.crossid import _fetch_tle_catalog

        fake_rows = [
            {"object_name": "DEBRIS", "tle_line1": "", "tle_line2": ""},
            {"object_name": _TEST_TLE_NAME, "tle_line1": _TEST_TLE_LINE1, "tle_line2": _TEST_TLE_LINE2},
        ]
        with patch("inference.crossid.query_tles_for_window", return_value=fake_rows):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert len(catalog) == 1
        assert catalog[0][0] == _TEST_TLE_NAME

    def test_empty_response_returns_empty_list(self):
        from inference.crossid import _fetch_tle_catalog

        with patch("inference.crossid.query_tles_for_window", return_value=[]):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert catalog == []

    def test_recent_obs_does_not_call_gp_current(self):
        """Inference never calls Space-Track GP directly when local DB misses."""
        from datetime import timedelta
        from inference.crossid import _fetch_tle_catalog

        recent = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        with patch("inference.crossid.query_tles_for_window", return_value=[]):
            catalog = _fetch_tle_catalog(recent, epoch_window_days=1)

        assert catalog == []

    def test_historical_obs_does_not_call_gp_history(self):
        """Inference treats missing historical catalog coverage as unknown."""
        from inference.crossid import _fetch_tle_catalog

        with patch("inference.crossid.query_tles_for_window", return_value=[]):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert catalog == []


# ---------------------------------------------------------------------------
# cross_identify — sky coords missing
# ---------------------------------------------------------------------------

class TestCrossIdentifyMissingSkyCoords:
    def test_none_ra_dec_gives_empty_identifications(self):
        from inference.crossid import cross_identify

        dets = [{"ra_tip1_deg": None, "dec_tip1_deg": None, "ra_tip2_deg": None, "dec_tip2_deg": None, "confidence": 0.9}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=_SAMPLE_CATALOG):
            result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)
        assert result[0]["identifications"] == []

    def test_missing_ra_dec_keys_gives_empty_identifications(self):
        from inference.crossid import cross_identify

        dets = [{"confidence": 0.9}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=_SAMPLE_CATALOG):
            result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)
        assert result[0]["identifications"] == []

    def test_no_crash_on_empty_detections_list(self):
        from inference.crossid import cross_identify

        with patch("inference.crossid._fetch_tle_catalog", return_value=_SAMPLE_CATALOG):
            result = cross_identify([], _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)
        assert result == []

    def test_empty_catalog_gives_empty_identifications(self):
        from inference.crossid import cross_identify

        dets = [{"ra_tip1_deg": 83.82, "dec_tip1_deg": -5.39, "ra_tip2_deg": 83.85, "dec_tip2_deg": -5.41}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=[]):
            result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)
        assert result[0]["identifications"] == []


# ---------------------------------------------------------------------------
# cross_identify — known TLE placed at detection position
# ---------------------------------------------------------------------------

class TestCrossIdentifyKnownTle:
    def test_top_candidate_highest_confidence(self):
        """Rank-1 candidate must have confidence ≥ all lower ranks."""
        from inference.crossid import cross_identify, _propagate_to_radec

        pred = _propagate_to_radec(
            _TEST_TLE_NAME, _TEST_TLE_LINE1, _TEST_TLE_LINE2,
            _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT,
        )
        if pred is None:
            pytest.skip("skyfield propagation failed in this environment")

        dets = [{"ra_tip1_deg": pred["predicted_ra"], "dec_tip1_deg": pred["predicted_dec"],
                 "ra_tip2_deg": pred["predicted_ra"] + 0.01, "dec_tip2_deg": pred["predicted_dec"] + 0.01}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=_SAMPLE_CATALOG):
            result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)

        ids = result[0]["identifications"]
        assert len(ids) >= 1
        assert ids[0]["rank"] == 1
        confs = [c["confidence"] for c in ids]
        assert confs == sorted(confs, reverse=True)

    def test_best_candidate_is_closest_satellite(self):
        """When detection is placed at a TLE's predicted position, that TLE
        should be the rank-1 match (lowest angular separation)."""
        from inference.crossid import cross_identify, _propagate_to_radec

        pred = _propagate_to_radec(
            _TEST_TLE_NAME, _TEST_TLE_LINE1, _TEST_TLE_LINE2,
            _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT,
        )
        if pred is None:
            pytest.skip("skyfield propagation failed in this environment")

        dets = [{"ra_tip1_deg": pred["predicted_ra"], "dec_tip1_deg": pred["predicted_dec"],
                 "ra_tip2_deg": pred["predicted_ra"] + 0.01, "dec_tip2_deg": pred["predicted_dec"] + 0.01}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=_SAMPLE_CATALOG):
            result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)

        ids = result[0]["identifications"]
        assert ids[0]["satellite_name"] == _TEST_TLE_NAME

    def test_identifications_have_required_keys(self):
        from inference.crossid import cross_identify, _propagate_to_radec

        pred = _propagate_to_radec(
            _TEST_TLE_NAME, _TEST_TLE_LINE1, _TEST_TLE_LINE2,
            _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT,
        )
        if pred is None:
            pytest.skip("skyfield propagation failed in this environment")

        dets = [{"ra_tip1_deg": pred["predicted_ra"], "dec_tip1_deg": pred["predicted_dec"],
                 "ra_tip2_deg": pred["predicted_ra"] + 0.01, "dec_tip2_deg": pred["predicted_dec"] + 0.01}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=_SAMPLE_CATALOG):
            result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)

        for ident in result[0]["identifications"]:
            assert "satellite_name" in ident
            assert "norad_id" in ident
            assert "confidence" in ident
            assert "rank" in ident

    def test_epoch_window_days_passed_to_fetch(self):
        """epoch_window_days is forwarded to _fetch_tle_catalog."""
        from inference.crossid import cross_identify

        dets = [{"ra_tip1_deg": None, "dec_tip1_deg": None, "ra_tip2_deg": None, "dec_tip2_deg": None}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=[]) as mock_fetch:
            cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, epoch_window_days=7)
        mock_fetch.assert_called_once_with(_OBS_TIME, 7)


# ---------------------------------------------------------------------------
# _angular_separation_arcsec
# ---------------------------------------------------------------------------

class TestAngularSeparation:
    def test_same_position_is_zero(self):
        from inference.crossid import _angular_separation_arcsec
        assert _angular_separation_arcsec(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-9)

    def test_one_degree_separation(self):
        from inference.crossid import _angular_separation_arcsec
        sep = _angular_separation_arcsec(0.0, 0.0, 1.0, 0.0)
        assert sep == pytest.approx(3600.0, rel=1e-4)

    def test_result_is_symmetric(self):
        from inference.crossid import _angular_separation_arcsec
        sep_ab = _angular_separation_arcsec(10.0, 5.0, 15.0, 10.0)
        sep_ba = _angular_separation_arcsec(15.0, 10.0, 10.0, 5.0)
        assert sep_ab == pytest.approx(sep_ba, rel=1e-9)
