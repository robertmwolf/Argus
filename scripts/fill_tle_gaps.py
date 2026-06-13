"""One-shot script to re-fetch all zero-record gap days from Space-Track.

Reads .env for credentials, queries the DB for coverage rows with record_count=0,
and calls fetch_day(force=True) for each.  Safe to re-run.
"""
import sys, logging, os
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env if credentials not already in environment
if not os.environ.get("SPACETRACK_USER"):
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    # Ensure production endpoint is used
    os.environ.setdefault("ARGUS_ENV", "production")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.matching.tle_store import get_engine, init_tle_tables
from sqlalchemy import text

eng = get_engine()
init_tle_tables(eng)

with eng.connect() as conn:
    rows = conn.execute(text(
        "SELECT source_tag FROM tle_catalog_coverage "
        "WHERE source_tag LIKE 'gp_history_creation_%' AND record_count = 0 "
        "ORDER BY source_tag"
    )).fetchall()

tag_prefix = "gp_history_creation_"
gap_dates = []
for r in rows:
    date_str = r[0][len(tag_prefix):]
    y, m, d = date_str.split("_")
    gap_dates.append(date(int(y), int(m), int(d)))

print(f"Found {len(gap_dates)} zero-record gap days to re-fetch:")
for d in gap_dates:
    print(f"  {d}")
print()

from scripts.bootstrap_recent_tles import fetch_day

total_inserted = 0
for i, d in enumerate(gap_dates):
    print(f"[{i+1}/{len(gap_dates)}] Fetching {d} ...", flush=True)
    try:
        inserted = fetch_day(d, force=True, engine=eng)
        total_inserted += inserted
        print(f"  → {inserted:,} new records", flush=True)
    except Exception as exc:
        logger.error("Failed %s: %s", d, exc)
        print(f"  → ERROR: {exc}", flush=True)

print(f"\nDone. Total new TLE records inserted: {total_inserted:,}")
