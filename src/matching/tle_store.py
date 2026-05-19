"""Synchronous TLE catalog store backed by the main ARGUS database.

The TLE catalog is a local copy of Space-Track data stored in the ``tle_catalog``
table of ``argus.db``.  Once a record is written it is never re-fetched from
Space-Track, fulfilling the gp_history one-time-download policy.

The main application uses async SQLAlchemy (FastAPI stack).  This module uses
a *synchronous* engine pointing at the same DATABASE_URL so the inference
pipeline (which is synchronous) can read TLEs without ``asyncio.run``.

Database URL conversion:
  sqlite+aiosqlite:///./argus.db  →  sqlite:///./argus.db
  postgresql+asyncpg://...        →  postgresql://...   (requires psycopg2-binary)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine management
# ---------------------------------------------------------------------------

_engine = None


def _sync_url(async_url: str) -> str:
    """Strip the async driver suffix from DATABASE_URL.

    Args:
        async_url: URL with an async driver, e.g. ``sqlite+aiosqlite:///...``.

    Returns:
        Sync-compatible URL, e.g. ``sqlite:///...``.
    """
    url = async_url.replace("+aiosqlite", "").replace("+asyncpg", "")
    # asyncpg uses postgresql scheme; psycopg2 also accepts it
    return url


def get_engine():
    """Return (and cache) a synchronous SQLAlchemy engine for the ARGUS DB.

    Reads DATABASE_URL from the environment exactly as ``db.models`` does.
    """
    global _engine
    if _engine is None:
        from db.models import get_database_url
        url = _sync_url(get_database_url())
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
    return _engine


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS tle_catalog (
    norad_id     INTEGER NOT NULL,
    epoch        TEXT NOT NULL,
    object_name  TEXT NOT NULL,
    object_type  TEXT,
    mean_motion  REAL,
    tle_line1    TEXT NOT NULL,
    tle_line2    TEXT NOT NULL,
    source       TEXT,
    ingested_at  TEXT DEFAULT (CURRENT_TIMESTAMP),
    PRIMARY KEY (norad_id, epoch)
);
CREATE INDEX IF NOT EXISTS idx_tle_catalog_epoch ON tle_catalog(epoch);
CREATE INDEX IF NOT EXISTS idx_tle_catalog_norad  ON tle_catalog(norad_id);
CREATE TABLE IF NOT EXISTS tle_catalog_coverage (
    source_tag    TEXT PRIMARY KEY,
    description   TEXT,
    record_count  INTEGER,
    downloaded_at TEXT DEFAULT (CURRENT_TIMESTAMP)
);
"""


def init_tle_tables(engine=None) -> None:
    """Create tle_catalog and tle_catalog_coverage tables if they do not exist.

    Also applies lightweight migrations (adding columns) to existing tables so
    the function is safe to call on an already-populated database.

    Called automatically by :func:`upsert_tles` and :func:`query_tles_for_window`
    so callers never need to call this explicitly.

    Args:
        engine: Optional sync SQLAlchemy engine; defaults to :func:`get_engine`.
    """
    eng = engine or get_engine()
    with eng.begin() as conn:
        for stmt in _CREATE_TABLES.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    # Migration: add source column to pre-existing tle_catalog tables.
    # Done in a separate connection so a no-op ALTER TABLE never taints the
    # CREATE TABLE transaction above.  PRAGMA table_info is used to guard
    # the ALTER TABLE because SQLite does not support IF NOT EXISTS on ADD COLUMN.
    with eng.connect() as chk:
        pragma_rows = chk.execute(text("PRAGMA table_info(tle_catalog)")).fetchall()
    existing_cols = {row[1] for row in pragma_rows}
    if "source" not in existing_cols:
        with eng.begin() as mig:
            mig.execute(text("ALTER TABLE tle_catalog ADD COLUMN source TEXT"))


# ---------------------------------------------------------------------------
# Coverage tracking
# ---------------------------------------------------------------------------

def has_coverage(source_tag: str, engine=None) -> bool:
    """Return True if *source_tag* exists in tle_catalog_coverage.

    Args:
        source_tag: e.g. ``'zip_2025'`` or ``'gp_current'``.
        engine: Optional sync engine.

    Returns:
        True when the source has already been loaded.
    """
    eng = engine or get_engine()
    init_tle_tables(eng)
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM tle_catalog_coverage WHERE source_tag = :tag"),
            {"tag": source_tag},
        ).fetchone()
    return row is not None


def record_coverage(
    source_tag: str,
    description: str,
    record_count: int,
    engine=None,
) -> None:
    """Insert or replace a coverage record.

    Args:
        source_tag: Unique identifier for this data source, e.g. ``'zip_2025'``.
        description: Human-readable description of what was loaded.
        record_count: Number of TLE records inserted.
        engine: Optional sync engine.
    """
    eng = engine or get_engine()
    with eng.begin() as conn:
        conn.execute(
            text(
                "INSERT OR REPLACE INTO tle_catalog_coverage "
                "(source_tag, description, record_count, downloaded_at) "
                "VALUES (:tag, :desc, :count, :ts)"
            ),
            {
                "tag": source_tag,
                "desc": description,
                "count": record_count,
                "ts": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
    logger.info("Coverage recorded: %s (%d records)", source_tag, record_count)


def get_last_coverage_time(source_tag: str, engine=None) -> datetime | None:
    """Return the ``downloaded_at`` timestamp for *source_tag*, or None.

    Used by TLECatalogManager to rate-gate CelesTrak refreshes.

    Args:
        source_tag: e.g. ``'celestrak_refresh'``.
        engine: Optional sync engine.

    Returns:
        UTC datetime of the last recorded refresh, or None if never refreshed.
    """
    eng = engine or get_engine()
    init_tle_tables(eng)
    with eng.connect() as conn:
        row = conn.execute(
            text(
                "SELECT downloaded_at FROM tle_catalog_coverage "
                "WHERE source_tag = :tag"
            ),
            {"tag": source_tag},
        ).fetchone()
    if row is None:
        return None
    try:
        ts = row[0]
        # Stored as ISO8601 string; parse to UTC datetime
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def get_latest_coverage_time(
    source_tags: list[str] | None = None,
    min_record_count: int = 0,
    engine=None,
) -> datetime | None:
    """Return the newest ``downloaded_at`` timestamp across coverage records.

    Args:
        source_tags: Optional subset of source tags to inspect.  When omitted,
            all coverage records are considered.
        min_record_count: Ignore coverage rows below this record count.  Use 1
            when callers need the last time fresh data actually arrived rather
            than the last attempted refresh.
        engine: Optional sync engine.

    Returns:
        UTC datetime of the newest recorded coverage timestamp, or None.
    """
    eng = engine or get_engine()
    init_tle_tables(eng)

    sql = "SELECT downloaded_at FROM tle_catalog_coverage"
    params: dict[str, Any] = {}
    clauses: list[str] = []
    if source_tags:
        placeholders = ", ".join(f":tag{i}" for i, _ in enumerate(source_tags))
        clauses.append(f"source_tag IN ({placeholders})")
        params = {f"tag{i}": tag for i, tag in enumerate(source_tags)}
    if min_record_count > 0:
        clauses.append("record_count >= :min_record_count")
        params["min_record_count"] = min_record_count
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    with eng.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    latest: datetime | None = None
    for row in rows:
        try:
            dt = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if latest is None or dt > latest:
            latest = dt
    return latest


# ---------------------------------------------------------------------------
# TLE record normalisation
# ---------------------------------------------------------------------------

def _parse_epoch_from_line1(line1: str) -> str:
    """Extract the TLE epoch from line 1 and return an ISO8601 UTC string.

    Args:
        line1: Standard TLE line 1.

    Returns:
        ISO8601 string like ``'2025-03-14T06:23:11Z'``.
    """
    try:
        epoch_str = line1[18:32].strip()
        yr2 = int(epoch_str[:2])
        year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
        day_frac = float(epoch_str[2:])
        epoch_dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_frac - 1)
        return epoch_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _normalise(record: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Space-Track API dict (uppercase keys) to a DB row dict.

    Also accepts dicts already in lowercase form (idempotent).

    Args:
        record: Raw dict from Space-Track API or a pre-normalised dict.

    Returns:
        Dict ready for insertion, or None if TLE lines are missing.
    """
    # Support both uppercase (API) and lowercase (already normalised)
    def get(upper: str, lower: str) -> Any:
        return record.get(upper) or record.get(lower)

    line1 = get("TLE_LINE1", "tle_line1") or ""
    line2 = get("TLE_LINE2", "tle_line2") or ""
    if not line1 or not line2:
        return None

    norad_raw = get("NORAD_CAT_ID", "norad_id")
    try:
        norad_id = int(str(norad_raw).strip())
    except (TypeError, ValueError):
        try:
            norad_id = int(line2[2:7].strip())
        except ValueError:
            return None

    epoch = get("EPOCH", "epoch") or _parse_epoch_from_line1(line1)
    if not epoch:
        return None

    # Normalise epoch to ISO8601
    epoch = re.sub(r"\.\d+$", "", epoch.replace("T", "T").replace(" ", "T"))
    if not epoch.endswith("Z"):
        epoch = epoch + "Z"

    return {
        "norad_id": norad_id,
        "epoch": epoch,
        "object_name": (get("OBJECT_NAME", "object_name") or f"NORAD-{norad_id}").strip(),
        "object_type": get("OBJECT_TYPE", "object_type"),
        "mean_motion": _to_float(get("MEAN_MOTION", "mean_motion")),
        "tle_line1": line1.strip(),
        "tle_line2": line2.strip(),
    }


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def upsert_tles(records: list[dict[str, Any]], engine=None, *, source: str | None = None) -> int:
    """Bulk insert TLE records, ignoring duplicates (same norad_id + epoch).

    Accepts both Space-Track API dicts (uppercase keys) and normalised dicts.

    Args:
        records: List of TLE dicts.
        source: Optional provenance tag stored in the ``source`` column,
            e.g. ``'bootstrap'``, ``'celestrak'``, ``'spacetrack_gp'``.
        engine: Optional sync engine.

    Returns:
        Number of rows actually inserted (duplicates skipped).
    """
    eng = engine or get_engine()
    init_tle_tables(eng)

    rows = [r for rec in records if (r := _normalise(rec)) is not None]
    if not rows:
        return 0

    if source is not None:
        for row in rows:
            row["source"] = source

    inserted = 0
    with eng.begin() as conn:
        for row in rows:
            result = conn.execute(
                text(
                    "INSERT OR IGNORE INTO tle_catalog "
                    "(norad_id, epoch, object_name, object_type, mean_motion, "
                    " tle_line1, tle_line2, source) "
                    "VALUES (:norad_id, :epoch, :object_name, :object_type, "
                    "        :mean_motion, :tle_line1, :tle_line2, :source)"
                ),
                {**row, "source": row.get("source")},
            )
            inserted += result.rowcount

    logger.debug("upsert_tles: %d/%d rows inserted", inserted, len(rows))
    return inserted


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def query_tles_for_window(
    obs_time: datetime,
    epoch_window_days: int = 3,
    min_mean_motion: float = 11.25,
    engine=None,
) -> list[dict[str, Any]]:
    """Return TLE records whose epoch falls in (obs_time - window, obs_time].

    Args:
        obs_time: UTC observation time.
        epoch_window_days: How many days before obs_time to search.
        min_mean_motion: Minimum mean_motion in rev/day (11.25 = LEO).
            Pass 0 to return all orbit classes.
        engine: Optional sync engine.

    Returns:
        List of dicts with keys: norad_id, epoch, object_name, object_type,
        mean_motion, tle_line1, tle_line2.
    """
    eng = engine or get_engine()
    init_tle_tables(eng)

    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    epoch_start = (obs_time - timedelta(days=epoch_window_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    epoch_end = obs_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    sql = (
        "SELECT norad_id, epoch, object_name, object_type, mean_motion, "
        "       tle_line1, tle_line2, source "
        "FROM tle_catalog "
        "WHERE epoch >= :start AND epoch <= :end "
    )
    params: dict[str, Any] = {"start": epoch_start, "end": epoch_end}

    if min_mean_motion > 0:
        sql += "AND (mean_motion IS NULL OR mean_motion >= :mm) "
        params["mm"] = min_mean_motion

    sql += "ORDER BY epoch DESC"

    with eng.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    result = [dict(r._mapping) for r in rows]
    logger.debug(
        "query_tles_for_window: %d records for window %s → %s",
        len(result), epoch_start, epoch_end,
    )
    return result


def _parse_iso_utc(value: Any) -> datetime | None:
    """Parse a stored UTC ISO timestamp into an aware datetime."""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _with_epoch_drift(rows: list[dict[str, Any]], obs_time: datetime) -> list[dict[str, Any]]:
    """Attach absolute epoch drift and sort rows nearest to the observation."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        epoch_dt = _parse_iso_utc(row.get("epoch"))
        drift_hours = (
            abs((obs_time - epoch_dt).total_seconds()) / 3600.0
            if epoch_dt is not None else float("inf")
        )
        enriched.append({**row, "epoch_drift_hours": drift_hours})
    enriched.sort(key=lambda r: r["epoch_drift_hours"])
    return enriched


def query_tles_for_epoch_drift(
    obs_time: datetime,
    epoch_window_days: int = 30,
    min_mean_motion: float = 11.25,
    engine=None,
) -> list[dict[str, Any]]:
    """Return TLE records within a broad symmetric epoch window.

    This is the deliberately wide fallback used when the normal local DB
    window has no candidates.  Rows are ordered by absolute epoch drift from
    the photo time so downstream scoring can penalize stale or future TLEs.

    Args:
        obs_time: UTC observation time.
        epoch_window_days: Days on either side of obs_time to search.
        min_mean_motion: Minimum mean_motion in rev/day.  Pass 0 for all orbits.
        engine: Optional sync engine.

    Returns:
        List of TLE row dicts with ``epoch_drift_hours`` attached.
    """
    eng = engine or get_engine()
    init_tle_tables(eng)

    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    epoch_start = (obs_time - timedelta(days=epoch_window_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    epoch_end = (obs_time + timedelta(days=epoch_window_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    sql = (
        "SELECT norad_id, epoch, object_name, object_type, mean_motion, "
        "       tle_line1, tle_line2, source "
        "FROM tle_catalog "
        "WHERE epoch >= :start AND epoch <= :end "
    )
    params: dict[str, Any] = {"start": epoch_start, "end": epoch_end}
    if min_mean_motion > 0:
        sql += "AND (mean_motion IS NULL OR mean_motion >= :mm) "
        params["mm"] = min_mean_motion

    with eng.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    result = _with_epoch_drift([dict(r._mapping) for r in rows], obs_time)
    logger.debug(
        "query_tles_for_epoch_drift: %d records for broad window %s → %s",
        len(result), epoch_start, epoch_end,
    )
    return result


def query_latest_tles(
    min_mean_motion: float = 11.25,
    engine=None,
) -> list[dict[str, Any]]:
    """Return the latest locally stored TLE for each NORAD object.

    Used as the final current-data fallback after local observation-window
    searches fail.  This may include TLEs far from the photo date; callers must
    apply a confidence penalty based on epoch drift.

    Args:
        min_mean_motion: Minimum mean_motion in rev/day.  Pass 0 for all orbits.
        engine: Optional sync engine.

    Returns:
        Latest TLE row per NORAD ID.
    """
    eng = engine or get_engine()
    init_tle_tables(eng)

    sql = (
        "SELECT c.norad_id, c.epoch, c.object_name, c.object_type, c.mean_motion, "
        "       c.tle_line1, c.tle_line2, c.source "
        "FROM tle_catalog c "
        "JOIN (SELECT norad_id, MAX(epoch) AS max_epoch FROM tle_catalog GROUP BY norad_id) latest "
        "  ON c.norad_id = latest.norad_id AND c.epoch = latest.max_epoch "
    )
    params: dict[str, Any] = {}
    if min_mean_motion > 0:
        sql += "WHERE (c.mean_motion IS NULL OR c.mean_motion >= :mm) "
        params["mm"] = min_mean_motion
    sql += "ORDER BY c.norad_id ASC"

    with eng.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    result = [dict(r._mapping) for r in rows]
    logger.debug("query_latest_tles: %d latest records", len(result))
    return result


# ---------------------------------------------------------------------------
# Bulk loader for large text files (streaming, batch inserts)
# ---------------------------------------------------------------------------

_INSERT_SQL = (
    "INSERT OR IGNORE INTO tle_catalog "
    "(norad_id, epoch, object_name, object_type, mean_motion, tle_line1, tle_line2, source) "
    "VALUES (:norad_id, :epoch, :object_name, :object_type, :mean_motion, :tle_line1, :tle_line2, :source)"
)


def bulk_load_tle_file(
    file_path: Path,
    source_tag: str,
    source: str = "bootstrap",
    engine=None,
    batch_size: int = 10_000,
) -> int:
    """Stream-parse a large TLE text file and bulk-insert into the DB.

    Uses SQLite WAL mode and executemany for high throughput.  Reads the file
    line-by-line so it never loads the full content into memory.

    Skips the load and returns 0 immediately if *source_tag* is already in
    tle_catalog_coverage (safe to re-run).

    Args:
        file_path: Path to a plain-text TLE file (2-line or 3-line format).
        source_tag: Coverage tag, e.g. ``'txt_2025'``.
        engine: Optional sync SQLAlchemy engine.
        batch_size: Rows per INSERT batch.

    Returns:
        Total rows inserted.
    """
    eng = engine or get_engine()
    init_tle_tables(eng)

    if has_coverage(source_tag, eng):
        logger.info("Skipping %s — already loaded (tag: %s)", file_path.name, source_tag)
        return 0

    total_inserted = 0
    total_parsed = 0
    batch: list[dict[str, Any]] = []

    def _flush(conn, b: list[dict[str, Any]]) -> int:
        if not b:
            return 0
        result = conn.execute(text(_INSERT_SQL), b)
        return result.rowcount

    # Enable WAL mode and relax durability for the bulk load session.
    # These are connection-level pragmas; they revert when the connection closes.
    with eng.connect() as setup_conn:
        setup_conn.execute(text("PRAGMA journal_mode=WAL"))
        setup_conn.execute(text("PRAGMA synchronous=NORMAL"))
        setup_conn.execute(text("PRAGMA cache_size=-65536"))   # 64 MB page cache
        setup_conn.commit()

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = _nonblank_lines(fh)
        with eng.begin() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.execute(text("PRAGMA cache_size=-65536"))

            for rec in _stream_tle_pairs(lines):
                total_parsed += 1
                rec["source"] = source
                batch.append(rec)
                if len(batch) >= batch_size:
                    total_inserted += _flush(conn, batch)
                    batch.clear()
                    if total_parsed % 500_000 == 0:
                        logger.info(
                            "  … %s: %d parsed, %d inserted",
                            file_path.name, total_parsed, total_inserted,
                        )

            # Final partial batch
            total_inserted += _flush(conn, batch)

    logger.info(
        "bulk_load_tle_file: %d parsed, %d inserted from %s",
        total_parsed, total_inserted, file_path.name,
    )
    record_coverage(
        source_tag,
        description=f"Bulk load from {file_path.name}",
        record_count=total_inserted,
        engine=eng,
    )
    return total_inserted


def _nonblank_lines(fh):
    """Yield non-blank stripped lines from a file handle."""
    for line in fh:
        line = line.rstrip()
        if line:
            yield line


def _stream_tle_pairs(lines):
    """Generate normalised TLE dicts from a stream of non-blank lines.

    Handles both 2-line (no name) and 3-line (name + line1 + line2) formats.
    Malformed blocks are silently skipped.
    """
    buf: list[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) < 2:
            continue

        # Try to detect a complete block at the end of buf
        if buf[-2].startswith("1 ") and buf[-1].startswith("2 "):
            name_candidate = buf[-3] if len(buf) >= 3 else ""
            if name_candidate and not name_candidate.startswith(("1 ", "2 ")):
                # 3-line block
                rec = _normalise({
                    "OBJECT_NAME": name_candidate.strip(),
                    "TLE_LINE1": buf[-2],
                    "TLE_LINE2": buf[-1],
                })
                buf.clear()
            else:
                # 2-line block
                rec = _normalise({"TLE_LINE1": buf[-2], "TLE_LINE2": buf[-1]})
                buf = buf[:-2]

            if rec:
                yield rec


# ---------------------------------------------------------------------------
# TLE zip file parser
# ---------------------------------------------------------------------------

def parse_tle_zip(zip_path: Path) -> list[dict[str, Any]]:
    """Parse all TLEs from a Space-Track annual zip bundle.

    Zip bundles contain one or more text files in 3-line TLE format::

        OBJECT NAME
        1 NNNNN U ...
        2 NNNNN ...

    Falls back to 2-line format (no name line) if the file uses that convention.

    Args:
        zip_path: Path to the downloaded ``.zip`` file.

    Returns:
        List of normalised TLE dicts ready for :func:`upsert_tles`.
    """
    import zipfile

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".txt", ".tle", "")):
                # skip directories and obviously non-TLE files
                if "." in Path(name).suffix:
                    continue
            try:
                content = zf.read(name).decode("utf-8", errors="replace")
            except Exception as exc:
                logger.warning("Could not read %s in zip: %s", name, exc)
                continue
            records.extend(_parse_tle_text(content))

    logger.info("Parsed %d TLE records from %s", len(records), zip_path.name)
    return records


def _parse_tle_text(content: str) -> list[dict[str, Any]]:
    """Parse TLE text (3-line or 2-line blocks) into normalised dicts.

    Args:
        content: Raw text content of a TLE file.

    Returns:
        List of normalised dicts.
    """
    lines = [ln.rstrip() for ln in content.splitlines()]
    # Filter blank lines while preserving positional structure
    lines = [ln for ln in lines if ln.strip()]

    records: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        l0 = lines[i]
        l1 = lines[i + 1] if i + 1 < len(lines) else ""
        l2 = lines[i + 2] if i + 2 < len(lines) else ""

        if l1.startswith("1 ") and l2.startswith("2 "):
            # 3-line block: l0 = name
            rec = _normalise({
                "OBJECT_NAME": l0.strip(),
                "TLE_LINE1": l1,
                "TLE_LINE2": l2,
            })
            if rec:
                records.append(rec)
            i += 3
        elif l0.startswith("1 ") and l1.startswith("2 "):
            # 2-line block (no name)
            rec = _normalise({"TLE_LINE1": l0, "TLE_LINE2": l1})
            if rec:
                records.append(rec)
            i += 2
        else:
            i += 1

    return records


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    eng = get_engine()
    init_tle_tables(eng)
    print("Tables created / verified.")

    sample = [
        {
            "OBJECT_NAME": "ISS (ZARYA)",
            "NORAD_CAT_ID": "25544",
            "TLE_LINE1": "1 25544U 98067A   24094.00000000  .00010000  00000-0  13679-3 0  9991",
            "TLE_LINE2": "2 25544  51.6400 200.0000 0001000   0.0000 360.0000 15.49999990451234",
            "MEAN_MOTION": "15.5",
            "OBJECT_TYPE": "PAYLOAD",
        }
    ]
    n = upsert_tles(sample, eng)
    print(f"Inserted {n} record(s).")

    obs = datetime(2024, 4, 3, 0, 0, 0, tzinfo=timezone.utc)
    results = query_tles_for_window(obs, epoch_window_days=1, min_mean_motion=0, engine=eng)
    print(f"Query returned {len(results)} record(s).")
    for r in results:
        print(f"  {r['object_name']}  NORAD={r['norad_id']}  epoch={r['epoch']}")

    sys.exit(0 if results else 1)
