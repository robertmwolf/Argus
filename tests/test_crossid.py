"""Tests for inference/crossid.py — satellite TLE cross-identification.

Unit tests run offline using mocks.
Integration tests (marked @pytest.mark.integration) make live Space-Track API
calls and require SPACETRACK_USER and SPACETRACK_PASS in the environment.
Run them with: pytest -m integration
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
# _fetch_tle_catalog — unit tests against the Space-Track adapter
# ---------------------------------------------------------------------------

class TestFetchTleCatalog:
    def test_maps_json_fields_to_tuples(self):
        """GP_History JSON records with TLE_LINE1/TLE_LINE2 become (name, l1, l2)."""
        from inference.crossid import _fetch_tle_catalog

        fake_records = [
            {
                "OBJECT_NAME": _TEST_TLE_NAME,
                "TLE_LINE1": _TEST_TLE_LINE1,
                "TLE_LINE2": _TEST_TLE_LINE2,
            }
        ]
        with patch("inference.crossid.query_gp_history", return_value=fake_records):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert len(catalog) == 1
        name, line1, line2 = catalog[0]
        assert name == _TEST_TLE_NAME
        assert line1 == _TEST_TLE_LINE1
        assert line2 == _TEST_TLE_LINE2

    def test_skips_records_without_tle_lines(self):
        """Records missing TLE_LINE1 or TLE_LINE2 are silently dropped."""
        from inference.crossid import _fetch_tle_catalog

        fake_records = [
            {"OBJECT_NAME": "DEBRIS", "TLE_LINE1": "", "TLE_LINE2": ""},
            {"OBJECT_NAME": _TEST_TLE_NAME, "TLE_LINE1": _TEST_TLE_LINE1, "TLE_LINE2": _TEST_TLE_LINE2},
        ]
        with patch("inference.crossid.query_gp_history", return_value=fake_records):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert len(catalog) == 1
        assert catalog[0][0] == _TEST_TLE_NAME

    def test_empty_response_returns_empty_list(self):
        from inference.crossid import _fetch_tle_catalog

        with patch("inference.crossid.query_gp_history", return_value=[]):
            catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)

        assert catalog == []


# ---------------------------------------------------------------------------
# cross_identify — sky coords missing
# ---------------------------------------------------------------------------

class TestCrossIdentifyMissingSkyCoords:
    def test_none_ra_dec_gives_empty_identifications(self):
        from inference.crossid import cross_identify

        dets = [{"ra_deg": None, "dec_deg": None, "confidence": 0.9}]
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

        dets = [{"ra_deg": 83.82, "dec_deg": -5.39}]
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

        dets = [{"ra_deg": pred["predicted_ra"], "dec_deg": pred["predicted_dec"]}]
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

        dets = [{"ra_deg": pred["predicted_ra"], "dec_deg": pred["predicted_dec"]}]
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

        dets = [{"ra_deg": pred["predicted_ra"], "dec_deg": pred["predicted_dec"]}]
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

        dets = [{"ra_deg": None, "dec_deg": None}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=[]) as mock_fetch:
            cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, epoch_window_days=7)
        mock_fetch.assert_called_once_with(_OBS_TIME, 7)


# ---------------------------------------------------------------------------
# Integration tests — live Space-Track API + end-to-end cross-ID
# Require: SPACETRACK_USER and SPACETRACK_PASS set in the environment.
# Run with: pytest -m integration
# ---------------------------------------------------------------------------

_ISS_NORAD = "25544"


@pytest.mark.integration
class TestCrossIdIntegration:
    """End-to-end tests that call Space-Track and run the full cross-ID pipeline.

    Each test uses an isolated tmp_path cache so a warm cache from a prior run
    cannot mask connectivity failures.
    """

    def test_fetch_tle_catalog_returns_nonempty_list(
        self, spacetrack_creds, tmp_path, monkeypatch
    ):
        """_fetch_tle_catalog must return at least one (name, line1, line2) tuple."""
        from inference.crossid import _fetch_tle_catalog

        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)
        assert len(catalog) > 0, "Expected TLE records; got empty catalog"

    def test_fetch_tle_catalog_tuples_have_valid_tle_format(
        self, spacetrack_creds, tmp_path, monkeypatch
    ):
        """Every tuple must be (str, '1 …', '2 …') — valid 3-line TLE structure."""
        from inference.crossid import _fetch_tle_catalog

        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)
        assert catalog, "No records returned"
        for name, line1, line2 in catalog[:20]:
            assert isinstance(name, str) and name
            assert line1.startswith("1 "), f"Bad TLE_LINE1: {line1!r}"
            assert line2.startswith("2 "), f"Bad TLE_LINE2: {line2!r}"

    def test_iss_present_in_catalog(self, spacetrack_creds, tmp_path, monkeypatch):
        """ISS (NORAD 25544) must appear in the catalog for the 3-day window."""
        from inference.crossid import _fetch_tle_catalog

        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)
        norad_ids = set()
        for _name, line1, _line2 in catalog:
            try:
                norad_ids.add(line1[2:7].strip())
            except IndexError:
                pass
        assert _ISS_NORAD in norad_ids, (
            f"ISS (NORAD {_ISS_NORAD}) not found in {len(catalog)}-entry catalog."
        )

    def test_cross_identify_iss_is_top_candidate(
        self, spacetrack_creds, tmp_path, monkeypatch
    ):
        """When a detection is placed at ISS's predicted sky position, ISS must
        be the rank-1 identification returned by the live pipeline."""
        from inference.crossid import _fetch_tle_catalog, _propagate_to_radec, cross_identify

        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")

        # Fetch real TLEs from Space-Track
        catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)
        assert catalog, "No TLEs returned from Space-Track"

        # Find ISS entry in the live catalog
        iss_entry = None
        for name, line1, line2 in catalog:
            try:
                if line1[2:7].strip() == _ISS_NORAD:
                    iss_entry = (name, line1, line2)
                    break
            except IndexError:
                continue
        if iss_entry is None:
            pytest.skip(f"ISS (NORAD {_ISS_NORAD}) not found in live catalog")

        # Propagate ISS to obs_time to get its actual sky position
        pred = _propagate_to_radec(
            iss_entry[0], iss_entry[1], iss_entry[2],
            _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT,
        )
        if pred is None:
            pytest.skip("skyfield propagation failed for ISS TLE")

        # Place a detection exactly at ISS's predicted position and cross-ID
        dets = [{"ra_deg": pred["predicted_ra"], "dec_deg": pred["predicted_dec"]}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=catalog):
            result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)

        ids = result[0]["identifications"]
        assert ids, "cross_identify returned no candidates"
        assert ids[0]["norad_id"] == int(_ISS_NORAD), (
            f"Expected ISS (NORAD {_ISS_NORAD}) as rank-1; "
            f"got {ids[0]['satellite_name']} (NORAD {ids[0]['norad_id']})"
        )

    def test_cross_identify_confidence_bounds(
        self, spacetrack_creds, tmp_path, monkeypatch
    ):
        """All confidence scores must be in [0, 1] and rank-1 ≥ rank-2 ≥ rank-3."""
        from inference.crossid import _fetch_tle_catalog, cross_identify

        monkeypatch.setattr("src.matching.spacetrack_query._CACHE_DIR", tmp_path / "cache")
        catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=1)
        assert catalog, "No TLEs returned"

        dets = [{"ra_deg": 83.82, "dec_deg": -5.39}]
        with patch("inference.crossid._fetch_tle_catalog", return_value=catalog):
            result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT)

        ids = result[0]["identifications"]
        for cand in ids:
            assert 0.0 <= cand["confidence"] <= 1.0, (
                f"Confidence out of bounds: {cand['confidence']}"
            )
        confs = [c["confidence"] for c in ids]
        assert confs == sorted(confs, reverse=True), "Candidates not sorted by confidence"


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
