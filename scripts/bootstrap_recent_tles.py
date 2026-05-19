"""Bootstrap and daily keep-up for Space-Track historical TLE data.

Downloads historical TLEs from the Space-Track ``gp_history`` API using
CREATION_DATE filtering — one calendar day per query.  This approach was
explicitly approved by Space-Track admin:

    "Step through each day starting on 1 Jan 2026, advancing one day at a
    time, using class/gp_history/CREATION_DATE/2026-01-01--2026-01-02/...
    You only need to run each day's historical query ONCE.  Keep up by
    running an updated version once per day after 0000 UTC."

The script is **idempotent**: days already recorded in ``tle_catalog_coverage``
are skipped automatically.  It doubles as both the one-time bootstrap tool and
the daily keep-up cron job — in either case it fetches only the days that are
not yet covered.

Usage
-----
::

    # Bootstrap last 90 days (default):
    python scripts/bootstrap_recent_tles.py

    # Bootstrap a specific number of days:
    python scripts/bootstrap_recent_tles.py --days 60

    # Bootstrap a specific date range:
    python scripts/bootstrap_recent_tles.py --start 2026-01-01 --end 2026-04-30

    # Re-fetch days that are already in the DB:
    python scripts/bootstrap_recent_tles.py --force

    # Daily cron keep-up (fetches only yesterday; all earlier days cached):
    python scripts/bootstrap_recent_tles.py

Daily cron example (runs at 00:16 UTC — off-hour as required by Space-Track)::

    16 0 * * * cd /path/to/Argus && \\
      SPACETRACK_USER=... SPACETRACK_PASS=... ARGUS_ENV=production \\
      /path/to/miniconda3/envs/satid/bin/python scripts/bootstrap_recent_tles.py \\
      >> logs/tle_keepup.log 2>&1

Scheduling rule (Space-Track requirement)
-----------------------------------------
Never schedule at :00 or :30 past the hour.  Use :16 or :44.
The script warns if it detects it is running near a busy boundary.

Environment variables required
-------------------------------
::

    SPACETRACK_USER   your@email.com
    SPACETRACK_PASS   yourpassword
    ARGUS_ENV         production   # omit to use Space-Track's test site

ARGUS routes to the test site (https://for-testing-only.space-track.org/)
by default unless ``ARGUS_ENV=production`` is set.  Both sites use the same
credentials and honour the same rate limits.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on the path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matching.spacetrack_query import _get_client, _rate_limit
from src.matching.tle_store import (
    get_engine,
    has_coverage,
    init_tle_tables,
    record_coverage,
    upsert_tles,
)
from src.matching.tle_store import _parse_tle_text  # private but stable

logger = logging.getLogger(__name__)

# Coverage tag prefix — one tag per calendar day.
_TAG_PREFIX = "gp_history_creation_"

# Space-Track scheduling — warn if running within this many minutes of :00 or :30.
_BUSY_MINUTE_RANGES = list(range(0, 5)) + list(range(28, 33))


def _tag(d: date) -> str:
    """Return the coverage tag for *d*, e.g. ``gp_history_creation_2026_04_27``."""
    return f"{_TAG_PREFIX}{d.strftime('%Y_%m_%d')}"


def _date_range(start: date, end: date):
    """Yield each date from *start* to *end* inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _warn_if_busy_time() -> None:
    """Log a warning if the current minute is near :00 or :30."""
    now_minute = datetime.now(tz=timezone.utc).minute
    if now_minute in _BUSY_MINUTE_RANGES:
        logger.warning(
            "Running at minute :%02d — near a Space-Track busy boundary. "
            "Space-Track recommends scheduling 5–25 minutes before or after "
            "the hour (e.g. :16 or :44). Consider adjusting your cron job.",
            now_minute,
        )


def fetch_day(
    d: date,
    *,
    force: bool = False,
    engine=None,
) -> int:
    """Fetch TLEs created on calendar day *d* and insert them into the DB.

    Skips the fetch if the day's coverage tag is already recorded (idempotent).

    Args:
        d: Calendar day to fetch (UTC CREATION_DATE).
        force: Re-fetch and re-insert even if already covered.
        engine: Optional SQLAlchemy engine; defaults to the project default.

    Returns:
        Number of new TLE records inserted.
    """
    tag = _tag(d)
    eng = engine or get_engine()

    if not force and has_coverage(tag, eng):
        logger.debug("Day %s already loaded (tag: %s) — skipping.", d, tag)
        return 0

    next_day = d + timedelta(days=1)
    creation_range = f"{d.isoformat()}--{next_day.isoformat()}"

    logger.info("Fetching GP_History CREATION_DATE %s …", creation_range)

    _rate_limit()  # honour 3 s inter-request sleep
    client = _get_client()

    try:
        raw = client.gp_history(creation_date=creation_range, format="tle")
    except Exception as exc:
        logger.error("GP_History query failed for %s: %s", d.isoformat(), exc)
        raise

    # raw is a string when format='tle'; list[dict] when format='json'.
    if isinstance(raw, (list, tuple)):
        # Unexpected JSON response — normalise to text so _parse_tle_text works.
        lines = []
        for rec in raw:
            name = rec.get("OBJECT_NAME") or rec.get("object_name") or ""
            l1 = rec.get("TLE_LINE1") or rec.get("tle_line1") or ""
            l2 = rec.get("TLE_LINE2") or rec.get("tle_line2") or ""
            if l1 and l2:
                if name:
                    lines.extend([name, l1, l2])
                else:
                    lines.extend([l1, l2])
        raw = "\n".join(lines)

    records = _parse_tle_text(raw)
    logger.debug("Day %s: parsed %d TLE records.", d, len(records))

    inserted = upsert_tles(records, engine=eng, source="gp_history_creation")
    record_coverage(
        tag,
        description=f"GP_History CREATION_DATE {d.isoformat()}",
        record_count=inserted,
        engine=eng,
    )

    logger.info("Day %s: %d new records inserted (of %d fetched).", d, inserted, len(records))
    return inserted


def bootstrap(
    start: date,
    end: date,
    *,
    force: bool = False,
    engine=None,
) -> int:
    """Fetch all days in [start, end] that are not yet covered.

    Args:
        start: First date to fetch (inclusive).
        end: Last date to fetch (inclusive).  Typically yesterday.
        force: Re-fetch covered days.
        engine: Optional SQLAlchemy engine.

    Returns:
        Total number of new TLE records inserted across all days.
    """
    eng = engine or get_engine()
    init_tle_tables(eng)

    _warn_if_busy_time()

    days = list(_date_range(start, end))
    total_days = len(days)
    skipped = sum(1 for d in days if not force and has_coverage(_tag(d), eng))
    to_fetch = total_days - skipped

    logger.info(
        "Bootstrap range: %s → %s (%d days; %d already covered, %d to fetch).",
        start, end, total_days, skipped, to_fetch,
    )
    if to_fetch == 0:
        logger.info("Nothing to fetch — all days in range already covered.")
        return 0

    total_inserted = 0
    for i, d in enumerate(days):
        tag = _tag(d)
        if not force and has_coverage(tag, eng):
            print(f"  [{i+1}/{total_days}] {d}: already loaded — skipping.")
            continue

        try:
            inserted = fetch_day(d, force=force, engine=eng)
        except Exception as exc:
            logger.error("Failed to fetch day %s: %s — continuing.", d, exc)
            print(f"  [{i+1}/{total_days}] {d}: ERROR — {exc}")
            continue

        total_inserted += inserted
        print(f"  [{i+1}/{total_days}] {d}: {inserted:,} new records.")

    return total_inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap and daily keep-up for Space-Track historical TLE data.\n"
            "Fetches one CREATION_DATE day at a time from GP_History."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--days",
        type=int,
        default=90,
        metavar="N",
        help="Number of days to look back from yesterday (default: 90).",
    )
    date_group.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        metavar="YYYY-MM-DD",
        help="First day to fetch (use with --end).",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        metavar="YYYY-MM-DD",
        help="Last day to fetch inclusive (defaults to yesterday when --start is given).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch and re-ingest days that are already in the DB.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    today = datetime.now(tz=timezone.utc).date()
    yesterday = today - timedelta(days=1)

    if args.start:
        start = args.start
        end = args.end if args.end else yesterday
    else:
        end = yesterday
        start = end - timedelta(days=args.days - 1)

    if start > end:
        parser.error(f"--start ({start}) must be on or before --end ({end}).")

    if end >= today:
        logger.warning(
            "End date %s is today or in the future. "
            "Today's TLEs are not yet complete — using yesterday (%s) instead.",
            end, yesterday,
        )
        end = yesterday

    print(f"\nSpace-Track GP_History bootstrap: {start} → {end}")
    print(f"  Days in range : {(end - start).days + 1}")
    print(f"  Force re-fetch: {args.force}")
    print()

    engine = get_engine()
    init_tle_tables(engine)

    total = bootstrap(start, end, force=args.force, engine=engine)

    print(f"\nDone. Total new TLE records inserted: {total:,}")
    if total == 0:
        already_covered = sum(
            1 for d in _date_range(start, end)
            if has_coverage(_tag(d), engine)
        )
        if already_covered == (end - start).days + 1:
            print("  All days in range were already loaded (use --force to re-ingest).")


if __name__ == "__main__":
    main()
