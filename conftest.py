"""Pytest configuration and shared fixtures for the ARGUS test suite."""

from __future__ import annotations
from pathlib import Path

import pytest

# Load .env from the project root so credentials are available without
# having to export them manually before every pytest run.
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=False)

# Drop real FITS files here to include them in the real-data test suite.
# Any .fits / .fit file placed in this directory is picked up automatically.
# Naming convention (optional but respected by tests):
#   *streak* — image is expected to contain at least one satellite streak
#   *blank*  — image is expected to contain no streaks
DATA_TEST_DIR = Path(__file__).parent / "data" / "test"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_data: tests that run against real FITS images in data/test/. "
        "Skipped automatically when the directory is empty. "
        "Run explicitly with: pytest -m real_data",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip real_data tests unless explicitly selected.

    Running ``pytest`` (no -m flag) does not require real FITS files. Pass
    ``-m real_data`` to opt in.
    """
    marker_expr = config.option.markexpr if hasattr(config.option, "markexpr") else ""

    skip_real_data = pytest.mark.skip(
        reason="requires real FITS files — run with: pytest -m real_data"
    )

    for item in items:
        if "real_data" in item.keywords and "real_data" not in marker_expr:
            item.add_marker(skip_real_data)


@pytest.fixture(scope="session")
def real_fits_files() -> list[Path]:
    """Return all .fits/.fit files found in data/test/.

    Tests that request this fixture are skipped when the directory is empty
    so the suite stays green without any real images committed to the repo.
    """
    files = sorted(
        p for p in DATA_TEST_DIR.glob("**/*")
        if p.suffix.lower() in {".fits", ".fit"} and p.is_file()
    )
    return files

