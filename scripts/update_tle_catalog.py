"""Explicitly update the local TLE catalog using the Space-Track GP class.

This is an operator-run maintenance tool, not part of the inference path.
Run it at most once per hour if ARGUS is deliberately configured to ingest
current/live TLEs.  It fetches the latest active TLEs and inserts any new
records into ``tle_catalog``.

Manual usage:

    python scripts/update_tle_catalog.py

The script self-enforces the one-hour limit via the 55-minute disk cache in
:func:`src.matching.spacetrack_query.query_gp_current`.  Inference never invokes
this script or the GP class automatically.

Schedule recommendation: run at HH:12 and HH:48 (10–20 min off the hour)
to avoid Space-Track peak load periods.

Environment variables required:
    SPACETRACK_USER, SPACETRACK_PASS
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matching.spacetrack_query import query_gp_current
from src.matching.tle_store import get_engine, init_tle_tables, record_coverage, upsert_tles

logger = logging.getLogger(__name__)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Update the ARGUS TLE catalog with current Space-Track GP data."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    engine = get_engine()
    init_tle_tables(engine)

    logger.info("Querying GP class for current active TLEs …")
    records = query_gp_current()   # honours 55-min cache; no-op if called recently
    if not records:
        logger.warning("GP class returned no records.")
        sys.exit(0)

    inserted = upsert_tles(records, engine)
    record_coverage(
        "gp_current",
        description="GP class snapshot — active satellites",
        record_count=inserted,
    )
    print(f"GP current update complete. New records inserted: {inserted:,} / {len(records):,} returned")


if __name__ == "__main__":
    main()
