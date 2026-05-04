"""Tests for inference/crossid.py — satellite TLE cross-identification."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Starlink-1007 TLE — used as a deterministic known-position reference
_TEST_TLE_NAME  = "STARLINK-1007"
_TEST_TLE_LINE1 = "1 44713U 19074A   24093.50000000  .00001764  00000-0  13679-3 0  9998"
_TEST_TLE_LINE2 = "2 44713  53.0536 100.4783 0001199  86.5965 273.5324 15.06396608241234"

_OBS_TIME = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)
_OBS_LAT, _OBS_LON, _OBS_ALT = 49.61, 6.13, 280.0

# A 3-line TLE file in string form
_SAMPLE_TLE_CONTENT = f"""{_TEST_TLE_NAME}
{_TEST_TLE_LINE1}
{_TEST_TLE_LINE2}
ISS (ZARYA)
1 25544U 98067A   24094.00000000  .00010000  00000-0  13679-3 0  9991
2 25544  51.6400 200.0000 0001000   0.0000 360.0000 15.49999990451234
"""


def _write_sample_catalog(path: Path, content: str = _SAMPLE_TLE_CONTENT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# _catalog_needs_refresh
# ---------------------------------------------------------------------------

class TestCatalogNeedsRefresh:
    def test_missing_file_returns_true(self, tmp_path):
        from inference.crossid import _catalog_needs_refresh
        assert _catalog_needs_refresh(tmp_path / "nonexistent.tle") is True

    def test_fresh_file_returns_false(self, tmp_path):
        from inference.crossid import _catalog_needs_refresh
        p = tmp_path / "catalog.tle"
        p.write_text("data")
        assert _catalog_needs_refresh(p) is False

    def test_stale_file_returns_true(self, tmp_path):
        from inference.crossid import _catalog_needs_refresh
        p = tmp_path / "catalog.tle"
        p.write_text("data")
        # Back-date the mtime by 25 hours
        stale_mtime = time.time() - 90_000
        import os
        os.utime(p, (stale_mtime, stale_mtime))
        assert _catalog_needs_refresh(p) is True


# ---------------------------------------------------------------------------
# _load_tle_catalog
# ---------------------------------------------------------------------------

class TestLoadTleCatalog:
    def test_parses_name_line1_line2(self, tmp_path):
        from inference.crossid import _load_tle_catalog
        p = tmp_path / "test.tle"
        _write_sample_catalog(p)
        catalog = _load_tle_catalog(p)
        assert len(catalog) == 2
        names = [t[0] for t in catalog]
        assert _TEST_TLE_NAME in names

    def test_returns_empty_list_for_missing_file(self, tmp_path):
        from inference.crossid import _load_tle_catalog
        catalog = _load_tle_catalog(tmp_path / "missing.tle")
        assert catalog == []

    def test_each_entry_is_3tuple(self, tmp_path):
        from inference.crossid import _load_tle_catalog
        p = tmp_path / "test.tle"
        _write_sample_catalog(p)
        for entry in _load_tle_catalog(p):
            assert len(entry) == 3
            name, line1, line2 = entry
            assert line1.startswith("1 ")
            assert line2.startswith("2 ")


# ---------------------------------------------------------------------------
# cross_identify — sky coords missing
# ---------------------------------------------------------------------------

class TestCrossIdentifyMissingSkyCoords:
    def test_none_ra_dec_gives_empty_identifications(self, tmp_path):
        from inference.crossid import cross_identify
        p = tmp_path / "cat.tle"
        _write_sample_catalog(p)
        dets = [{"ra_deg": None, "dec_deg": None, "confidence": 0.9}]
        result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, catalog_path=p)
        assert result[0]["identifications"] == []

    def test_missing_ra_dec_keys_gives_empty_identifications(self, tmp_path):
        from inference.crossid import cross_identify
        p = tmp_path / "cat.tle"
        _write_sample_catalog(p)
        dets = [{"confidence": 0.9}]  # no ra_deg / dec_deg keys at all
        result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, catalog_path=p)
        assert result[0]["identifications"] == []

    def test_no_crash_on_empty_detections_list(self, tmp_path):
        from inference.crossid import cross_identify
        p = tmp_path / "cat.tle"
        _write_sample_catalog(p)
        result = cross_identify([], _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, catalog_path=p)
        assert result == []


# ---------------------------------------------------------------------------
# cross_identify — known TLE placed at detection position
# ---------------------------------------------------------------------------

class TestCrossIdentifyKnownTle:
    def test_top_candidate_highest_confidence(self, tmp_path):
        """Rank-1 candidate must have confidence ≥ rank-2 and rank-3."""
        from inference.crossid import cross_identify, _propagate_to_radec

        # Propagate the test TLE to get its actual predicted position
        pred = _propagate_to_radec(
            _TEST_TLE_NAME, _TEST_TLE_LINE1, _TEST_TLE_LINE2,
            _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT,
        )
        if pred is None:
            pytest.skip("skyfield propagation failed in this environment")

        p = tmp_path / "cat.tle"
        _write_sample_catalog(p)

        # Place detection exactly at the TLE's predicted position
        dets = [{"ra_deg": pred["predicted_ra"], "dec_deg": pred["predicted_dec"]}]
        result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, catalog_path=p)

        ids = result[0]["identifications"]
        assert len(ids) >= 1
        # Ranks should be 1, 2, 3
        assert ids[0]["rank"] == 1
        # Confidences should be non-increasing
        confs = [c["confidence"] for c in ids]
        assert confs == sorted(confs, reverse=True)

    def test_best_candidate_is_closest_satellite(self, tmp_path):
        """When detection is placed at a TLE's predicted position, that TLE
        should be the rank-1 match (lowest angular separation)."""
        from inference.crossid import cross_identify, _propagate_to_radec

        pred = _propagate_to_radec(
            _TEST_TLE_NAME, _TEST_TLE_LINE1, _TEST_TLE_LINE2,
            _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT,
        )
        if pred is None:
            pytest.skip("skyfield propagation failed in this environment")

        p = tmp_path / "cat.tle"
        _write_sample_catalog(p)

        dets = [{"ra_deg": pred["predicted_ra"], "dec_deg": pred["predicted_dec"]}]
        result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, catalog_path=p)

        ids = result[0]["identifications"]
        assert ids[0]["satellite_name"] == _TEST_TLE_NAME

    def test_identifications_have_required_keys(self, tmp_path):
        from inference.crossid import cross_identify, _propagate_to_radec

        pred = _propagate_to_radec(
            _TEST_TLE_NAME, _TEST_TLE_LINE1, _TEST_TLE_LINE2,
            _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT,
        )
        if pred is None:
            pytest.skip("skyfield propagation failed in this environment")

        p = tmp_path / "cat.tle"
        _write_sample_catalog(p)
        dets = [{"ra_deg": pred["predicted_ra"], "dec_deg": pred["predicted_dec"]}]
        result = cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, catalog_path=p)
        ids = result[0]["identifications"]
        for ident in ids:
            assert "satellite_name" in ident
            assert "norad_id" in ident
            assert "confidence" in ident
            assert "rank" in ident


# ---------------------------------------------------------------------------
# _download_catalog — stale catalog triggers refresh
# ---------------------------------------------------------------------------

class TestDownloadCatalog:
    def test_stale_catalog_triggers_download(self, tmp_path):
        """When the catalog is stale, cross_identify should call _download_catalog."""
        import os
        from inference.crossid import cross_identify

        p = tmp_path / "cat.tle"
        _write_sample_catalog(p)
        # Make it look stale
        stale_mtime = time.time() - 90_000
        os.utime(p, (stale_mtime, stale_mtime))

        with patch("inference.crossid._download_catalog") as mock_dl:
            # _download_catalog is mocked so it won't actually touch the file;
            # _load_tle_catalog will still read the existing (stale but present) file.
            mock_dl.return_value = None
            dets = [{"ra_deg": 10.0, "dec_deg": 20.0}]
            cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, catalog_path=p)
            mock_dl.assert_called_once_with(p)

    def test_fresh_catalog_does_not_trigger_download(self, tmp_path):
        from inference.crossid import cross_identify

        p = tmp_path / "cat.tle"
        _write_sample_catalog(p)
        # File is freshly written — no download needed

        with patch("inference.crossid._download_catalog") as mock_dl:
            dets = [{"ra_deg": 10.0, "dec_deg": 20.0}]
            cross_identify(dets, _OBS_TIME, _OBS_LAT, _OBS_LON, _OBS_ALT, catalog_path=p)
            mock_dl.assert_not_called()

    def test_spacetrack_path_raises_not_implemented(self, tmp_path, monkeypatch):
        """USE_SPACETRACK=true should raise NotImplementedError in _download_catalog."""
        from inference.crossid import _download_catalog
        monkeypatch.setenv("USE_SPACETRACK", "true")
        with pytest.raises(NotImplementedError, match="Space-Track"):
            _download_catalog(tmp_path / "cat.tle")


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
