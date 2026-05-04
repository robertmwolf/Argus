"""Pytest configuration and shared fixtures for the ARGUS test suite."""

from __future__ import annotations

import os
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
        "integration: tests that make live network calls to external APIs "
        "(Space-Track, etc.). Require SPACETRACK_USER and SPACETRACK_PASS "
        "in the environment or in a .env file at the project root.",
    )
    config.addinivalue_line(
        "markers",
        "real_data: tests that run against real FITS images in data/test/. "
        "Skipped automatically when the directory is empty. "
        "Run explicitly with: pytest -m real_data",
    )


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


@pytest.fixture
def spacetrack_creds():
    """Provide Space-Track credentials from the environment.

    Credentials are loaded from SPACETRACK_USER / SPACETRACK_PASS env vars,
    which are automatically populated from a .env file in the project root
    if one exists.  The test fails (not skips) when credentials are absent
    so missing configuration is immediately visible.
    """
    user = os.environ.get("SPACETRACK_USER", "")
    pw   = os.environ.get("SPACETRACK_PASS", "")
    if not user or not pw:
        pytest.fail(
            "Space-Track credentials not found. "
            "Set SPACETRACK_USER and SPACETRACK_PASS in your environment "
            "or in a .env file at the project root."
        )
    return {"user": user, "password": pw}
