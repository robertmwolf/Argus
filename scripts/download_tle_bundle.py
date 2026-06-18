"""Download Space-Track annual TLE zip bundles and load them into the ARGUS database.

Uses the Space-Track ``fileshare`` API class to discover and download annual TLE
bundles, then feeds them into the existing ``bootstrap_tle_catalog`` loader.

The annual bundles are the only policy-compliant bulk source for historical TLE
data.  Each bundle covers one full calendar year; a partial bundle for the
current year is typically available and grows as the year progresses.

Usage
-----
::

    # List what bundles are available on Space-Track:
    python scripts/download_tle_bundle.py --list

    # Download bundle(s) that cover the last 3 months (default):
    python scripts/download_tle_bundle.py

    # Explicit date window or year:
    python scripts/download_tle_bundle.py --months 6
    python scripts/download_tle_bundle.py --year 2026

    # Download and also refresh the live GP snapshot (≤ once/hour):
    python scripts/download_tle_bundle.py --update-current

Environment variables required
-------------------------------
::

    SPACETRACK_USER   your@email.com
    SPACETRACK_PASS   yourpassword
    ARGUS_ENV         production   # omit to use Space-Track's test site

The test site uses the same credentials and the same fileshare API, but the
file catalogue may differ.  Always set ARGUS_ENV=production for real data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matching.spacetrack_query import _get_client, _rate_limit
from src.matching.tle_store import get_engine, has_coverage, init_tle_tables

logger = logging.getLogger(__name__)

# Where to write downloaded bundle files before ingestion.
_DEFAULT_ZIP_DIR = Path("data/tle_zips")

# Minimum string fragments to recognise an annual TLE bundle by filename.
_BUNDLE_SUFFIXES = (".zip", ".txt", ".tle")
_BUNDLE_YEAR_MIN = 2020


def _years_needed(months: int) -> list[int]:
    """Return the calendar years that span today - months to today.

    Args:
        months: Number of months to look back.

    Returns:
        Sorted list of calendar years (e.g. [2025, 2026]).
    """
    now = datetime.now(tz=timezone.utc)
    years = set()
    years.add(now.year)
    month = now.month - months
    year = now.year
    while month <= 0:
        month += 12
        year -= 1
    years.add(year)
    return sorted(years)


def _looks_like_bundle(filename: str) -> bool:
    """Return True if *filename* looks like an annual TLE bundle file."""
    name = filename.lower()
    if not any(name.endswith(s) for s in _BUNDLE_SUFFIXES):
        return False
    for year in range(_BUNDLE_YEAR_MIN, datetime.now(tz=timezone.utc).year + 1):
        if str(year) in name:
            return True
    return False


def _year_from_filename(filename: str) -> int | None:
    """Extract a 4-digit year from a filename, or return None."""
    import re
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else None


def list_available_bundles(verbose: bool = False) -> list[dict]:
    """Fetch the Space-Track fileshare catalogue and return annual TLE bundles.

    Args:
        verbose: Log all fileshare entries, not just bundles.

    Returns:
        List of file dicts with keys: FILE_ID, FILE_NAME, FOLDER_NAME, FILE_SIZE.
    """
    client = _get_client()
    _rate_limit()

    logger.info("Fetching fileshare catalogue …")
    all_files = list(client.file(format="json"))
    logger.debug("Fileshare total entries: %d", len(all_files))

    bundles = []
    for f in all_files:
        name = f.get("FILE_NAME") or f.get("file_name") or ""
        if verbose:
            logger.debug("fileshare entry: %s", f)
        if _looks_like_bundle(name):
            bundles.append(f)

    bundles.sort(key=lambda f: f.get("FILE_NAME") or "")
    return bundles


def download_bundle(file_entry: dict, dest_dir: Path) -> Path:
    """Download one fileshare entry to *dest_dir* with streaming progress.

    Skips the download if the local file already exists at the same size.

    Args:
        file_entry: Dict from Space-Track fileshare catalogue.
        dest_dir: Directory to write the file into.

    Returns:
        Path to the downloaded (or existing) local file.
    """
    # Normalise key names — Space-Track may return upper or lower case.
    def _get(d: dict, *keys: str):
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
        return None

    file_id = _get(file_entry, "FILE_ID", "file_id")
    file_name = _get(file_entry, "FILE_NAME", "file_name") or f"tle_bundle_{file_id}"
    file_size = _get(file_entry, "FILE_SIZE", "file_size")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file_name

    if dest_path.exists():
        local_size = dest_path.stat().st_size
        if file_size and abs(local_size - int(file_size)) < 1024:
            logger.info("Already downloaded: %s (%d bytes)", file_name, local_size)
            return dest_path
        logger.warning(
            "Local file %s exists but size mismatch (local=%d, remote=%s); re-downloading.",
            file_name, local_size, file_size,
        )

    remote_size_mb = f"{int(file_size) / 1_048_576:.0f} MB" if file_size else "unknown size"
    logger.info("Downloading %s (%s) …", file_name, remote_size_mb)
    print(f"  Downloading {file_name} ({remote_size_mb}) — this may take several minutes.")

    client = _get_client()
    _rate_limit()

    bytes_written = 0
    last_reported = 0

    with open(dest_path, "wb") as fh:
        for chunk in client.fileshare.download(
            file_id=file_id,
            iter_content=True,
        ):
            fh.write(chunk)
            bytes_written += len(chunk)
            mb_written = bytes_written / 1_048_576
            if mb_written - last_reported >= 100:
                print(f"    … {mb_written:.0f} MB downloaded", end="\r", flush=True)
                last_reported = mb_written

    print(f"  Downloaded {bytes_written / 1_048_576:.1f} MB → {dest_path}")
    logger.info("Downloaded %s (%d bytes)", file_name, bytes_written)
    return dest_path


def _print_fallback_instructions(years: list[int]) -> None:
    print(
        "\nCould not find TLE bundle(s) for the requested years via the fileshare API.\n"
        "\nDownload manually from:\n"
        "  https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu\n"
        f"\nSave the file(s) for year(s) {years} to:  data/tle_zips/\n"
        "\nThen load them:\n"
        f"  python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ "
        f"--years {' '.join(str(y) for y in years)}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Space-Track annual TLE bundles and ingest them into ARGUS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available bundles on Space-Track and exit.",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        metavar="N",
        help="Download bundle(s) covering today - N months (default: 3).",
    )
    parser.add_argument(
        "--year",
        type=int,
        metavar="YYYY",
        help="Download bundle for a specific year instead of using --months.",
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=_DEFAULT_ZIP_DIR,
        metavar="DIR",
        help=f"Directory for downloaded files (default: {_DEFAULT_ZIP_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-ingest even if coverage is already recorded.",
    )
    parser.add_argument(
        "--update-current",
        action="store_true",
        help="After loading bundles, also fetch the latest active TLEs via the GP class.",
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

    # Determine which years are needed.
    if args.year:
        years_needed = [args.year]
    else:
        years_needed = _years_needed(args.months)

    logger.info("Years needed: %s", years_needed)

    # Ensure DB tables exist before we start.
    engine = get_engine()
    init_tle_tables(engine)

    # --list mode: just print available bundles and exit.
    if args.list:
        try:
            bundles = list_available_bundles(verbose=args.verbose)
        except Exception as exc:
            print(f"Error fetching fileshare catalogue: {exc}")
            sys.exit(1)

        if not bundles:
            print("No annual TLE bundles found in the Space-Track fileshare.")
            _print_fallback_instructions(years_needed)
            sys.exit(0)

        print(f"\n{'FILE_ID':<10} {'SIZE':>10}  FILE_NAME")
        print("-" * 60)
        for f in bundles:
            fid = f.get("FILE_ID") or f.get("file_id") or "?"
            fname = f.get("FILE_NAME") or f.get("file_name") or "?"
            fsize = f.get("FILE_SIZE") or f.get("file_size")
            size_str = f"{int(fsize) / 1_048_576:.0f} MB" if fsize else "?"
            print(f"{str(fid):<10} {size_str:>10}  {fname}")
        sys.exit(0)

    # Normal mode: download and ingest.
    # Late import to avoid circular-import issues.
    from scripts.bootstrap_tle_catalog import load_file, update_current  # type: ignore[import]

    try:
        bundles = list_available_bundles(verbose=args.verbose)
    except Exception as exc:
        logger.error("Failed to list fileshare: %s", exc)
        _print_fallback_instructions(years_needed)
        sys.exit(1)

    if not bundles:
        logger.warning("No annual TLE bundles found in fileshare.")
        _print_fallback_instructions(years_needed)
        sys.exit(1)

    # Map year → file entry.
    year_to_bundle: dict[int, dict] = {}
    for f in bundles:
        fname = f.get("FILE_NAME") or f.get("file_name") or ""
        year = _year_from_filename(fname)
        if year:
            year_to_bundle[year] = f

    logger.debug("Available years in fileshare: %s", sorted(year_to_bundle))

    total_inserted = 0
    for year in years_needed:
        coverage_tag = f"txt_{year}"
        if not args.force and has_coverage(coverage_tag, engine):
            logger.info("Year %d already loaded (tag: %s) — skipping.", year, coverage_tag)
            print(f"  Year {year}: already in DB — skipping (use --force to re-ingest).")
            continue

        if year not in year_to_bundle:
            print(f"\n  Year {year}: no bundle found in Space-Track fileshare.")
            logger.warning("No fileshare entry for year %d.", year)
            continue

        try:
            local_path = download_bundle(year_to_bundle[year], args.zip_dir)
        except Exception as exc:
            logger.error("Download failed for year %d: %s", year, exc)
            print(f"  Year {year}: download failed — {exc}")
            continue

        logger.info("Ingesting %s …", local_path)
        inserted = load_file(local_path, force=args.force)
        total_inserted += inserted
        print(f"  Year {year}: {inserted:,} new TLE records loaded.")

    if args.update_current:
        print("\nRefreshing current GP snapshot (≤ once/hour) …")
        gp_inserted = update_current()
        total_inserted += gp_inserted
        print(f"  GP snapshot: {gp_inserted:,} new records.")

    print(f"\nBootstrap complete. Total new records inserted: {total_inserted:,}")

    if not total_inserted and not any(has_coverage(f"txt_{y}", engine) for y in years_needed):
        _print_fallback_instructions(years_needed)


if __name__ == "__main__":
    main()
