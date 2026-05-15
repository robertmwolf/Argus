"""Tests for src/matching/tle_store.py.

All tests use an in-memory SQLite engine — no disk, no network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """In-memory SQLite engine for each test."""
    from src.matching.tle_store import init_tle_tables
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    init_tle_tables(eng)
    return eng


# Minimal valid TLE records in Space-Track API format (uppercase keys)
_ISS_RECORD = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "NORAD_CAT_ID": "25544",
    "OBJECT_TYPE": "PAYLOAD",
    "MEAN_MOTION": "15.5",
    "TLE_LINE1": "1 25544U 98067A   24094.00000000  .00010000  00000-0  13679-3 0  9991",
    "TLE_LINE2": "2 25544  51.6400 200.0000 0001000   0.0000 360.0000 15.49999990451234",
    "EPOCH": "2024-04-03T00:00:00",
}

_STARLINK_RECORD = {
    "OBJECT_NAME": "STARLINK-1007",
    "NORAD_CAT_ID": "44713",
    "OBJECT_TYPE": "PAYLOAD",
    "MEAN_MOTION": "15.06",
    "TLE_LINE1": "1 44713U 19074A   24093.50000000  .00001764  00000-0  13679-3 0  9998",
    "TLE_LINE2": "2 44713  53.0536 100.4783 0001199  86.5965 273.5324 15.06396608241234",
    "EPOCH": "2024-04-02T12:00:00",
}

_OBS_TIME = datetime(2024, 4, 4, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# init_tle_tables
# ---------------------------------------------------------------------------

class TestInitTleTables:
    def test_tables_are_created(self, engine):
        with engine.connect() as conn:
            tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        names = {r[0] for r in tables}
        assert "tle_catalog" in names
        assert "tle_catalog_coverage" in names

    def test_idempotent(self, engine):
        from src.matching.tle_store import init_tle_tables
        init_tle_tables(engine)  # second call must not raise
        init_tle_tables(engine)


# ---------------------------------------------------------------------------
# upsert_tles
# ---------------------------------------------------------------------------

class TestUpsertTles:
    def test_inserts_valid_record(self, engine):
        from src.matching.tle_store import upsert_tles
        n = upsert_tles([_ISS_RECORD], engine)
        assert n == 1

    def test_duplicate_is_ignored(self, engine):
        from src.matching.tle_store import upsert_tles
        upsert_tles([_ISS_RECORD], engine)
        n = upsert_tles([_ISS_RECORD], engine)
        assert n == 0, "Re-inserting the same (norad_id, epoch) must be a no-op"

    def test_multiple_records(self, engine):
        from src.matching.tle_store import upsert_tles
        n = upsert_tles([_ISS_RECORD, _STARLINK_RECORD], engine)
        assert n == 2

    def test_record_missing_tle_lines_is_skipped(self, engine):
        from src.matching.tle_store import upsert_tles
        bad = {"OBJECT_NAME": "DEBRIS", "NORAD_CAT_ID": "99999", "TLE_LINE1": "", "TLE_LINE2": ""}
        n = upsert_tles([bad], engine)
        assert n == 0

    def test_empty_list_returns_zero(self, engine):
        from src.matching.tle_store import upsert_tles
        assert upsert_tles([], engine) == 0

    def test_accepts_lowercase_keys(self, engine):
        """Normalised dicts (lowercase) must also be accepted."""
        from src.matching.tle_store import upsert_tles
        rec = {
            "object_name": "ISS",
            "norad_id": 25544,
            "tle_line1": _ISS_RECORD["TLE_LINE1"],
            "tle_line2": _ISS_RECORD["TLE_LINE2"],
            "epoch": "2024-04-03T00:00:01Z",
        }
        assert upsert_tles([rec], engine) == 1


# ---------------------------------------------------------------------------
# query_tles_for_window
# ---------------------------------------------------------------------------

class TestQueryTlesForWindow:
    def test_returns_records_within_window(self, engine):
        from src.matching.tle_store import query_tles_for_window, upsert_tles
        upsert_tles([_ISS_RECORD, _STARLINK_RECORD], engine)
        # Both epochs are within 3 days before _OBS_TIME (2024-04-04)
        results = query_tles_for_window(_OBS_TIME, epoch_window_days=3, min_mean_motion=0, engine=engine)
        norad_ids = {r["norad_id"] for r in results}
        assert 25544 in norad_ids
        assert 44713 in norad_ids

    def test_includes_record_at_lower_boundary(self, engine):
        """epoch exactly at (obs_time - window) must be included (inclusive range)."""
        from src.matching.tle_store import query_tles_for_window, upsert_tles
        upsert_tles([_ISS_RECORD], engine)
        # ISS epoch = 2024-04-03T00:00:00Z; window lower bound = obs_time - 1d = same value
        results = query_tles_for_window(_OBS_TIME, epoch_window_days=1, min_mean_motion=0, engine=engine)
        assert any(r["norad_id"] == 25544 for r in results)

    def test_mean_motion_filter_excludes_geo(self, engine):
        from src.matching.tle_store import query_tles_for_window, upsert_tles
        geo = dict(_ISS_RECORD)
        geo["NORAD_CAT_ID"] = "11111"
        geo["MEAN_MOTION"] = "1.0"   # GEO object
        geo["EPOCH"] = "2024-04-03T06:00:00"
        upsert_tles([geo], engine)
        results = query_tles_for_window(
            _OBS_TIME, epoch_window_days=3, min_mean_motion=11.25, engine=engine
        )
        assert all(r["norad_id"] != 11111 for r in results)

    def test_empty_db_returns_empty_list(self, engine):
        from src.matching.tle_store import query_tles_for_window
        assert query_tles_for_window(_OBS_TIME, engine=engine) == []

    def test_result_rows_have_required_keys(self, engine):
        from src.matching.tle_store import query_tles_for_window, upsert_tles
        upsert_tles([_ISS_RECORD], engine)
        results = query_tles_for_window(_OBS_TIME, epoch_window_days=3, min_mean_motion=0, engine=engine)
        required = {"norad_id", "epoch", "object_name", "tle_line1", "tle_line2"}
        for row in results:
            assert required <= row.keys()


# ---------------------------------------------------------------------------
# has_coverage / record_coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_has_coverage_false_before_recording(self, engine):
        from src.matching.tle_store import has_coverage
        assert not has_coverage("zip_2025", engine)

    def test_has_coverage_true_after_recording(self, engine):
        from src.matching.tle_store import has_coverage, record_coverage
        record_coverage("zip_2025", "2025 annual bundle", 500_000, engine)
        assert has_coverage("zip_2025", engine)

    def test_record_coverage_upserts(self, engine):
        from src.matching.tle_store import has_coverage, record_coverage
        record_coverage("zip_2025", "first load", 100, engine)
        record_coverage("zip_2025", "re-load", 0, engine)  # must not raise
        assert has_coverage("zip_2025", engine)

    def test_different_tags_are_independent(self, engine):
        from src.matching.tle_store import has_coverage, record_coverage
        record_coverage("zip_2025", "2025 bundle", 1, engine)
        assert not has_coverage("zip_2024", engine)


# ---------------------------------------------------------------------------
# parse_tle_zip
# ---------------------------------------------------------------------------

class TestParseTleZip:
    def _make_zip(self, tmp_path: Path, content: str, filename: str = "tles.txt") -> Path:
        import zipfile
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(filename, content)
        return zip_path

    def test_parses_3_line_format(self, tmp_path):
        from src.matching.tle_store import parse_tle_zip
        content = (
            "ISS (ZARYA)\n"
            "1 25544U 98067A   24094.00000000  .00010000  00000-0  13679-3 0  9991\n"
            "2 25544  51.6400 200.0000 0001000   0.0000 360.0000 15.49999990451234\n"
        )
        zip_path = self._make_zip(tmp_path, content)
        records = parse_tle_zip(zip_path)
        assert len(records) == 1
        assert records[0]["norad_id"] == 25544
        assert records[0]["object_name"] == "ISS (ZARYA)"

    def test_parses_2_line_format(self, tmp_path):
        from src.matching.tle_store import parse_tle_zip
        content = (
            "1 25544U 98067A   24094.00000000  .00010000  00000-0  13679-3 0  9991\n"
            "2 25544  51.6400 200.0000 0001000   0.0000 360.0000 15.49999990451234\n"
        )
        zip_path = self._make_zip(tmp_path, content)
        records = parse_tle_zip(zip_path)
        assert len(records) == 1
        assert records[0]["norad_id"] == 25544

    def test_parses_multiple_objects(self, tmp_path):
        from src.matching.tle_store import parse_tle_zip
        content = (
            "ISS (ZARYA)\n"
            "1 25544U 98067A   24094.00000000  .00010000  00000-0  13679-3 0  9991\n"
            "2 25544  51.6400 200.0000 0001000   0.0000 360.0000 15.49999990451234\n"
            "STARLINK-1007\n"
            "1 44713U 19074A   24093.50000000  .00001764  00000-0  13679-3 0  9998\n"
            "2 44713  53.0536 100.4783 0001199  86.5965 273.5324 15.06396608241234\n"
        )
        zip_path = self._make_zip(tmp_path, content)
        records = parse_tle_zip(zip_path)
        assert len(records) == 2

    def test_empty_zip_returns_empty_list(self, tmp_path):
        from src.matching.tle_store import parse_tle_zip
        zip_path = self._make_zip(tmp_path, "")
        assert parse_tle_zip(zip_path) == []

    def test_malformed_lines_are_skipped(self, tmp_path):
        from src.matching.tle_store import parse_tle_zip
        content = (
            "JUNK LINE\n"
            "ANOTHER JUNK\n"
            "ISS (ZARYA)\n"
            "1 25544U 98067A   24094.00000000  .00010000  00000-0  13679-3 0  9991\n"
            "2 25544  51.6400 200.0000 0001000   0.0000 360.0000 15.49999990451234\n"
        )
        zip_path = self._make_zip(tmp_path, content)
        records = parse_tle_zip(zip_path)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# crossid._fetch_tle_catalog — DB-first routing
# ---------------------------------------------------------------------------

class TestFetchTleCatalogDbFirst:
    """Verify that _fetch_tle_catalog uses the local DB before the API."""

    def test_db_hit_skips_celestrak(self, engine, monkeypatch):
        """When the DB has records for the window, CelesTrak is not called."""
        import inference.crossid as crossid
        from src.matching.tle_store import upsert_tles
        from inference.crossid import _fetch_tle_catalog

        upsert_tles([_ISS_RECORD, _STARLINK_RECORD], engine)

        fake_rows = [
            {"object_name": "ISS (ZARYA)",
             "tle_line1": _ISS_RECORD["TLE_LINE1"],
             "tle_line2": _ISS_RECORD["TLE_LINE2"]},
        ]
        with patch.object(crossid._tle_manager, "get_tles", return_value=fake_rows):
            with patch("scripts.celestrak_client.fetch_and_upsert") as mock_ct:
                catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=3)

        mock_ct.assert_not_called()
        assert len(catalog) == 1

    def test_db_miss_returns_empty_catalog_without_spacetrack(self, monkeypatch):
        """When the DB misses on a historical obs, Space-Track is never called."""
        import inference.crossid as crossid
        from inference.crossid import _fetch_tle_catalog

        with patch.object(crossid._tle_manager, "get_tles", return_value=[]):
            with patch("src.matching.spacetrack_query.query_gp_current") as mock_gp_current, \
                 patch("src.matching.spacetrack_query.query_gp_history") as mock_gp_history:

                catalog = _fetch_tle_catalog(_OBS_TIME, epoch_window_days=3)

        mock_gp_current.assert_not_called()
        mock_gp_history.assert_not_called()
        assert catalog == []
