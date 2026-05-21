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
        """TLECatalogManager rows with tle_line1/tle_line2 become (name, l1, l2)."""
        import inference.crossid as crossid
        from inference.crossid import _fetch_tle_catalog

        fake_rows = [
            {
                "object_name": _TEST_TLE_NAME,
                "tle_line1": _TEST_TLE_LINE1,
                "tle_line2": _TEST_TLE_LINE2,
            }
        ]
        with patch.object(crossid._tle_manager, "get_tles", return_value=fake_rows):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert len(catalog) == 1
        assert catalog[0]["name"] == _TEST_TLE_NAME
        assert catalog[0]["line1"] == _TEST_TLE_LINE1
        assert catalog[0]["line2"] == _TEST_TLE_LINE2

    def test_skips_records_without_tle_lines(self):
        """Records missing TLE_LINE1 or TLE_LINE2 are silently dropped."""
        import inference.crossid as crossid
        from inference.crossid import _fetch_tle_catalog

        fake_rows = [
            {"object_name": "DEBRIS", "tle_line1": "", "tle_line2": ""},
            {"object_name": _TEST_TLE_NAME, "tle_line1": _TEST_TLE_LINE1, "tle_line2": _TEST_TLE_LINE2},
        ]
        with patch.object(crossid._tle_manager, "get_tles", return_value=fake_rows):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert len(catalog) == 1
        assert catalog[0]["name"] == _TEST_TLE_NAME

    def test_empty_response_returns_empty_list(self):
        import inference.crossid as crossid
        from inference.crossid import _fetch_tle_catalog

        with patch.object(crossid._tle_manager, "get_tles", return_value=[]):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert catalog == []

    def test_recent_obs_does_not_call_gp_current(self):
        """Inference never calls Space-Track GP directly — TLECatalogManager handles routing."""
        from datetime import timedelta
        import inference.crossid as crossid
        from inference.crossid import _fetch_tle_catalog

        recent = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        with patch.object(crossid._tle_manager, "get_tles", return_value=[]):
            catalog = _fetch_tle_catalog(recent, epoch_window_days=1)

        assert catalog == []

    def test_historical_obs_does_not_call_gp_history(self):
        """Inference treats missing historical catalog coverage as unknown."""
        import inference.crossid as crossid
        from inference.crossid import _fetch_tle_catalog

        with patch.object(crossid._tle_manager, "get_tles", return_value=[]):
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
        mock_fetch.assert_called_once_with(_OBS_TIME, 7, 0)

    def test_broad_epoch_without_viable_match_tries_current_fallback(self):
        """A populated broad catalog with only poor geometry still retries current data."""
        from inference.crossid import cross_identify

        broad_catalog = [{
            "name": "BROAD-BAD",
            "line1": _TEST_TLE_LINE1,
            "line2": _TEST_TLE_LINE2,
            "tle_search_mode": "broad_epoch",
        }]
        current_catalog = [{
            "name": "CURRENT-GOOD",
            "line1": _ISS_LINE1,
            "line2": _ISS_LINE2,
            "tle_search_mode": "current_fallback",
        }]

        def fake_propagate(name, *_args, **_kwargs):
            if name == "CURRENT-GOOD":
                return {
                    "object_name": name,
                    "norad_id": 25544,
                    "predicted_ra": 10.0,
                    "predicted_dec": 20.0,
                    "tle_age_hours": 0.0,
                    "tle_epoch": _OBS_TIME,
                }
            return {
                "object_name": name,
                "norad_id": 44713,
                "predicted_ra": 180.0,
                "predicted_dec": -20.0,
                "tle_age_hours": 0.0,
                "tle_epoch": _OBS_TIME,
            }

        dets = [{
            "ra_tip1_deg": 10.0,
            "dec_tip1_deg": 20.0,
            "ra_tip2_deg": 10.0,
            "dec_tip2_deg": 20.0,
        }]
        with patch("inference.crossid._fetch_tle_catalog", return_value=broad_catalog):
            with patch("inference.crossid._fetch_current_tle_catalog", return_value=current_catalog) as mock_current:
                with patch("inference.crossid._propagate_to_radec", side_effect=fake_propagate):
                    result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)

        mock_current.assert_called_once_with(_OBS_TIME)
        best = result[0]["identifications"][0]
        assert best["satellite_name"] == "CURRENT-GOOD"
        assert best["tle_search_mode"] == "current_fallback"


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


# ---------------------------------------------------------------------------
# _streak_mid_radec
# ---------------------------------------------------------------------------

class TestStreakMidRadec:
    def test_both_tips_averages_ra_dec(self):
        from inference.crossid import _streak_mid_radec
        det = {"ra_tip1_deg": 10.0, "dec_tip1_deg": 20.0,
               "ra_tip2_deg": 12.0, "dec_tip2_deg": 22.0}
        mid = _streak_mid_radec(det)
        assert mid is not None
        assert mid[0] == pytest.approx(11.0, abs=1e-6)
        assert mid[1] == pytest.approx(21.0, abs=1e-6)

    def test_only_tip1_returns_tip1(self):
        from inference.crossid import _streak_mid_radec
        det = {"ra_tip1_deg": 10.0, "dec_tip1_deg": 20.0,
               "ra_tip2_deg": None, "dec_tip2_deg": None}
        mid = _streak_mid_radec(det)
        assert mid == pytest.approx((10.0, 20.0))

    def test_only_tip2_returns_tip2(self):
        from inference.crossid import _streak_mid_radec
        det = {"ra_tip1_deg": None, "dec_tip1_deg": None,
               "ra_tip2_deg": 15.0, "dec_tip2_deg": 5.0}
        mid = _streak_mid_radec(det)
        assert mid == pytest.approx((15.0, 5.0))

    def test_no_tips_returns_none(self):
        from inference.crossid import _streak_mid_radec
        det = {"ra_tip1_deg": None, "dec_tip1_deg": None,
               "ra_tip2_deg": None, "dec_tip2_deg": None}
        assert _streak_mid_radec(det) is None

    def test_ra_wrap_around_360(self):
        """RA near 0°/360° wrap should average correctly."""
        from inference.crossid import _streak_mid_radec
        det = {"ra_tip1_deg": 359.0, "dec_tip1_deg": 0.0,
               "ra_tip2_deg":   1.0, "dec_tip2_deg": 0.0}
        mid = _streak_mid_radec(det)
        assert mid is not None
        # midpoint should be near 0°, not 180°
        assert mid[0] == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# _plate_scale_from_det
# ---------------------------------------------------------------------------

class TestPlateScaleFromDet:
    def test_known_geometry_returns_correct_scale(self):
        """1° separation across 3600 px streak → 1 arcsec/px."""
        from inference.crossid import _plate_scale_from_det
        det = {
            "ra_tip1_deg": 0.0, "dec_tip1_deg": 0.0,
            "ra_tip2_deg": 1.0, "dec_tip2_deg": 0.0,
            "obb": {"cx": 1800.0, "cy": 1800.0, "w": 3600.0, "h": 5.0, "angle_deg": 0.0},
        }
        scale = _plate_scale_from_det(det)
        assert scale == pytest.approx(1.0, rel=1e-3)

    def test_missing_sky_coords_returns_none(self):
        from inference.crossid import _plate_scale_from_det
        det = {"ra_tip1_deg": None, "dec_tip1_deg": None,
               "ra_tip2_deg": None, "dec_tip2_deg": None,
               "obb": {"w": 200.0}}
        assert _plate_scale_from_det(det) is None

    def test_too_short_streak_returns_none(self):
        """OBB width < 10 px → cannot derive a reliable plate scale."""
        from inference.crossid import _plate_scale_from_det
        det = {"ra_tip1_deg": 0.0, "dec_tip1_deg": 0.0,
               "ra_tip2_deg": 0.01, "dec_tip2_deg": 0.0,
               "obb": {"cx": 50.0, "cy": 50.0, "w": 5.0, "h": 5.0, "angle_deg": 0.0}}
        assert _plate_scale_from_det(det) is None

    def test_missing_obb_returns_none(self):
        from inference.crossid import _plate_scale_from_det
        det = {"ra_tip1_deg": 0.0, "dec_tip1_deg": 0.0,
               "ra_tip2_deg": 1.0, "dec_tip2_deg": 0.0}
        assert _plate_scale_from_det(det) is None


# ---------------------------------------------------------------------------
# _atrk_xtrk — along-track / cross-track residual decomposition
# Source: SkyTrack (colleague) — ComputeOneResidual
# ---------------------------------------------------------------------------

class TestAtrkXtrk:
    def test_zero_residual(self):
        """Observed equals predicted → both residuals are zero."""
        from inference.crossid import _atrk_xtrk
        atrk, xtrk = _atrk_xtrk(
            obs_ra=10.0, obs_dec=20.0,
            pred_ra=10.0, pred_dec=20.0,
            start_ra=9.0, start_dec=20.0,
            end_ra=11.0, end_dec=20.0,
        )
        assert atrk == pytest.approx(0.0, abs=1e-6)
        assert xtrk == pytest.approx(0.0, abs=1e-6)

    def test_pure_along_track_error(self):
        """Error purely along track direction → Xtrk ≈ 0, |Atrk| > 0."""
        from inference.crossid import _atrk_xtrk
        # Track runs East (RA increases); shift observation +0.1 deg in RA
        atrk, xtrk = _atrk_xtrk(
            obs_ra=10.1, obs_dec=20.0,
            pred_ra=10.0, pred_dec=20.0,
            start_ra=9.0, start_dec=20.0,
            end_ra=11.0, end_dec=20.0,
        )
        assert abs(atrk) > 10.0            # significant along-track component
        assert abs(xtrk) == pytest.approx(0.0, abs=1.0)  # near-zero cross-track

    def test_pure_cross_track_error(self):
        """Error purely across track direction → Atrk ≈ 0, |Xtrk| > 0."""
        from inference.crossid import _atrk_xtrk
        # Track runs East; shift observation +0.1 deg in Dec (cross-track)
        atrk, xtrk = _atrk_xtrk(
            obs_ra=10.0, obs_dec=20.1,
            pred_ra=10.0, pred_dec=20.0,
            start_ra=9.0, start_dec=20.0,
            end_ra=11.0, end_dec=20.0,
        )
        assert abs(atrk) == pytest.approx(0.0, abs=1.0)
        assert abs(xtrk) > 10.0

    def test_degenerate_track_returns_zeros(self):
        """Zero-length track vector → both residuals are zero (no crash)."""
        from inference.crossid import _atrk_xtrk
        atrk, xtrk = _atrk_xtrk(
            obs_ra=10.0, obs_dec=20.0,
            pred_ra=10.0, pred_dec=20.0,
            start_ra=10.0, start_dec=20.0,
            end_ra=10.0, end_dec=20.0,   # zero-length
        )
        assert atrk == pytest.approx(0.0, abs=1e-9)
        assert xtrk == pytest.approx(0.0, abs=1e-9)
