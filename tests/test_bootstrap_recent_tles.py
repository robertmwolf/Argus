"""Tests for unified recent TLE bootstrap and gap repair."""

from datetime import date

from sqlalchemy import create_engine, text

from scripts import bootstrap_recent_tles
from src.matching.tle_store import init_tle_tables, record_coverage


def _engine():
    engine = create_engine("sqlite:///:memory:")
    init_tle_tables(engine)
    return engine


def test_get_gap_dates_returns_only_valid_zero_record_days(caplog):
    engine = _engine()
    record_coverage("gp_history_creation_2026_06_01", "empty", 0, engine)
    record_coverage("gp_history_creation_2026_06_02", "loaded", 4, engine)
    record_coverage("zip_2025", "bundle", 0, engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO tle_catalog_coverage "
                "(source_tag, description, record_count) VALUES (:tag, '', 0)"
            ),
            {"tag": "gp_history_creation_invalid"},
        )

    assert bootstrap_recent_tles.get_gap_dates(engine) == [date(2026, 6, 1)]
    assert "Ignoring malformed" in caplog.text


def test_fill_gaps_force_fetches_each_gap(monkeypatch):
    engine = _engine()
    record_coverage("gp_history_creation_2026_06_01", "empty", 0, engine)
    record_coverage("gp_history_creation_2026_06_03", "empty", 0, engine)
    calls = []

    def fake_fetch_day(day, *, force=False, engine=None):
        calls.append((day, force, engine))
        return 2

    monkeypatch.setattr(bootstrap_recent_tles, "fetch_day", fake_fetch_day)
    monkeypatch.setattr(bootstrap_recent_tles, "_warn_if_busy_time", lambda: None)

    assert bootstrap_recent_tles.fill_gaps(engine=engine) == 4
    assert calls == [
        (date(2026, 6, 1), True, engine),
        (date(2026, 6, 3), True, engine),
    ]
