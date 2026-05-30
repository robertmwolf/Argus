"""CelesTrak GP data client for the ARGUS live TLE edge.

Pulls the active satellite catalog and analyst-range (80000-series) objects
from CelesTrak's public GP API and upserts them into the local ``tle_catalog``
table.  No authentication is required.

CelesTrak rate limits
---------------------
- GP data updates at most once every 2 hours.
- 250 MB/day bandwidth cap per IP.
- High-volume groups (active, starlink) must not be re-downloaded within one
  update cycle.  :func:`fetch_and_upsert` enforces this via the
  ``tle_catalog_coverage`` table (tag ``'celestrak_refresh'``).

This client is intentionally separate from Space-Track integration.  It does
not touch ``SPACETRACK_USER`` / ``SPACETRACK_PASS`` and does not use the
Space-Track test site.  CelesTrak is a public mirror with its own endpoint.

Usage (standalone)
------------------
::

    python scripts/celestrak_client.py          # full refresh
    python scripts/celestrak_client.py --force  # bypass cooldown (testing only)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matching.tle_store import (
    get_engine,
    get_last_coverage_time,
    init_tle_tables,
    record_coverage,
    upsert_tles,
)

logger = logging.getLogger(__name__)

_CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php"
_TIMEOUT_S = 60
_COOLDOWN_HOURS = 2.0
_COVERAGE_TAG = "celestrak_refresh"

# Groups to fetch.  Each entry is either GROUP=<name> or SPECIAL=<name>.
# GPZ-PLUS covers 80000-series analyst/unclassified objects (the "NEO"
# population in space-surveillance parlance).
_QUERIES: list[dict[str, str]] = [
    {"GROUP": "active"},
    {"SPECIAL": "GPZ-PLUS"},
]


def _fetch_group(params: dict[str, str]) -> list[dict]:
    """Fetch one CelesTrak group and return parsed OMM JSON records.

    Args:
        params: Query parameters — one of ``{"GROUP": "active"}`` or
            ``{"SPECIAL": "GPZ-PLUS"}``.

    Returns:
        List of OMM JSON dicts (same field names as Space-Track GP API).

    Raises:
        requests.HTTPError: On non-2xx response.
        requests.Timeout: If the request exceeds ``_TIMEOUT_S`` seconds.
    """
    query = {**params, "FORMAT": "json"}
    label = next(iter(params.values()))
    logger.info("CelesTrak fetch: %s …", label)
    resp = requests.get(_CELESTRAK_BASE, params=query, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    records = resp.json()
    logger.info("CelesTrak %s: %d records received", label, len(records))
    return records


def fetch_and_upsert(force: bool = False, engine=None) -> int:
    """Pull all configured CelesTrak groups and upsert into ``tle_catalog``.

    Respects the 2-hour cooldown unless *force* is True.  Cooldown state is
    stored in ``tle_catalog_coverage`` under the tag ``'celestrak_refresh'``
    so it persists across process restarts.

    Args:
        force: If True, bypass the cooldown check (use in tests or manual
            maintenance only).
        engine: Optional sync SQLAlchemy engine.

    Returns:
        Total number of new rows inserted across all groups.
    """
    eng = engine or get_engine()
    init_tle_tables(eng)

    if not force:
        last = get_last_coverage_time(_COVERAGE_TAG, eng)
        if last is not None:
            age_hours = (datetime.now(tz=timezone.utc) - last).total_seconds() / 3600
            if age_hours < _COOLDOWN_HOURS:
                logger.debug(
                    "CelesTrak cooldown active (%.1fh / %.1fh) — skipping refresh",
                    age_hours,
                    _COOLDOWN_HOURS,
                )
                return 0

    total_inserted = 0
    total_records = 0

    for params in _QUERIES:
        try:
            records = _fetch_group(params)
        except Exception as exc:
            logger.warning("CelesTrak fetch failed for %s: %s", params, exc)
            continue

        if not records:
            continue

        total_records += len(records)
        inserted = upsert_tles(records, eng, source="celestrak")
        total_inserted += inserted
        logger.debug("CelesTrak %s: %d new rows inserted", next(iter(params.values())), inserted)

    record_coverage(
        _COVERAGE_TAG,
        description=f"CelesTrak GP refresh — {total_records} records fetched",
        record_count=total_records,
        engine=eng,
    )
    logger.info(
        "CelesTrak refresh complete: %d/%d new rows inserted",
        total_inserted,
        total_records,
    )
    return total_inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh the ARGUS TLE catalog from CelesTrak (active + analyst objects)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the 2-hour cooldown (for testing or manual maintenance).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    n = fetch_and_upsert(force=args.force)
    print(f"CelesTrak refresh: {n:,} new rows inserted.")
    sys.exit(0)
