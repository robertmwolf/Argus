"""Bootstrap the local TLE catalog for a new ARGUS environment.

Downloads Space-Track annual TLE zip bundles and loads them into the
``tle_catalog`` table of the ARGUS database.  Re-running this script is safe:
years that are already present in the database are skipped automatically.

Space-Track API policy: gp_history is a one-time download resource.  This
script fulfils that requirement by loading data into a local database so it
never needs to be re-fetched.

Usage
-----
1.  Download the annual zip bundles from Space-Track's cloud storage:
    https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu
    Save the zip file(s) to a local directory (e.g. ``data/tle_zips/``).

2.  Run this script:

    # Load a single zip file:
    python scripts/bootstrap_tle_catalog.py --zip data/tle_zips/TLE_2025.zip

    # Load all zips in a directory (default year filter: 2025):
    python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/

    # Load zips for specific years found in a directory:
    python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2023 2024 2025

    # Optional explicit maintenance only: pull the latest TLEs via the GP class:
    python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --update-current

Environment variables required only for --update-current:
    SPACETRACK_USER, SPACETRACK_PASS
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matching.tle_store import (
    bulk_load_tle_file,
    get_engine,
    has_coverage,
    init_tle_tables,
    parse_tle_zip,
    record_coverage,
    upsert_tles,
)

logger = logging.getLogger(__name__)


def _year_tag(year: int) -> str:
    return f"zip_{year}"


def _infer_year(zip_path: Path) -> int | None:
    """Try to extract a 4-digit year from the zip filename."""
    import re
    match = re.search(r"(20\d{2})", zip_path.stem)
    return int(match.group(1)) if match else None


def load_file(file_path: Path, force: bool = False) -> int:
    """Load one TLE file (zip or plain text) into the TLE catalog.

    Plain ``.txt`` files are streamed line-by-line with batch inserts for
    large archives (tens of millions of records).  Zip files are extracted
    then parsed in memory.

    Args:
        file_path: Path to a ``.zip`` or ``.txt`` TLE file.
        force: If True, re-load even if coverage is already recorded.

    Returns:
        Number of new rows inserted.
    """
    year = _infer_year(file_path)
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".tle"):
        tag = f"txt_{year}" if year else f"txt_{file_path.stem}"
    else:
        tag = _year_tag(year) if year else f"zip_{file_path.stem}"

    if not force and has_coverage(tag):
        logger.info("Skipping %s — already loaded (tag: %s)", file_path.name, tag)
        return 0

    if suffix in (".txt", ".tle"):
        logger.info("Bulk-loading %s (streaming) …", file_path)
        inserted = bulk_load_tle_file(file_path, source_tag=tag, batch_size=10_000)
    else:
        logger.info("Parsing zip %s …", file_path)
        records = parse_tle_zip(file_path)
        if not records:
            logger.warning("No TLE records found in %s", file_path.name)
            return 0
        logger.info("Inserting %d records from %s …", len(records), file_path.name)
        inserted = upsert_tles(records)
        record_coverage(
            tag,
            description=f"Annual TLE bundle loaded from {file_path.name}",
            record_count=inserted,
        )

    logger.info("Loaded %d new records from %s (tag: %s)", inserted, file_path.name, tag)
    return inserted


def update_current() -> int:
    """Pull the latest active TLEs via the GP class and store them.

    Returns:
        Number of new rows inserted.
    """
    from src.matching.spacetrack_query import query_gp_current

    logger.info("Fetching current TLEs via GP class (≤ once/hour) …")
    records = query_gp_current()
    inserted = upsert_tles(records)
    record_coverage(
        "gp_current",
        description="GP class snapshot — active satellites only",
        record_count=inserted,
    )
    logger.info("GP current: %d new records stored", inserted)
    return inserted


def _find_tle_files(zip_dir: Path, years: list[int] | None) -> list[Path]:
    """Return zip and text TLE files in zip_dir, optionally filtered to given years."""
    candidates = sorted(
        p for p in zip_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".zip", ".txt", ".tle")
    )
    if not years:
        return candidates
    filtered = []
    for p in candidates:
        year = _infer_year(p)
        if year is None or year in years:
            filtered.append(p)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap the ARGUS TLE catalog from Space-Track annual zip bundles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--zip",
        metavar="PATH",
        type=Path,
        help="Load a single zip file.",
    )
    source.add_argument(
        "--zip-dir",
        metavar="DIR",
        type=Path,
        help="Load all *.zip files found in this directory.",
    )
    parser.add_argument(
        "--years",
        metavar="YEAR",
        type=int,
        nargs="+",
        default=[2025],
        help="When using --zip-dir, only load zips whose filename contains one "
             "of these years. Default: 2025.",
    )
    parser.add_argument(
        "--update-current",
        action="store_true",
        help="Optional explicit maintenance: after loading zips, also fetch "
             "the latest TLEs via the GP class (requires SPACETRACK_USER and "
             "SPACETRACK_PASS). Inference never does this automatically.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-load zips even if they are already recorded in the coverage log.",
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

    # Ensure tables exist
    engine = get_engine()
    init_tle_tables(engine)

    total_inserted = 0

    if args.zip:
        if not args.zip.exists():
            logger.error("File not found: %s", args.zip)
            sys.exit(1)
        total_inserted += load_file(args.zip, force=args.force)

    elif args.zip_dir:
        if not args.zip_dir.is_dir():
            logger.error("Directory not found: %s", args.zip_dir)
            sys.exit(1)
        files = _find_tle_files(args.zip_dir, args.years)
        if not files:
            logger.warning(
                "No TLE files (*.zip, *.txt, *.tle) found in %s matching years %s.\n"
                "Download bundles from:\n"
                "  https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu\n"
                "then re-run with --zip-dir %s",
                args.zip_dir, args.years, args.zip_dir,
            )
            sys.exit(1)
        for file_path in files:
            total_inserted += load_file(file_path, force=args.force)

    else:
        parser.print_help()
        print(
            "\nDownload annual TLE zip bundles from:\n"
            "  https://ln5.sync.com/dl/afd354190/"
            "c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu\n"
            "then run:\n"
            "  python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/\n"
        )
        sys.exit(0)

    if args.update_current:
        total_inserted += update_current()

    print(f"\nBootstrap complete. Total new records inserted: {total_inserted:,}")


if __name__ == "__main__":
    main()
