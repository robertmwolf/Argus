"""Pytest configuration and shared fixtures for the ARGUS test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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
        "(Space-Track, etc.). Skipped automatically when credentials are absent. "
        "Run explicitly with: pytest -m integration",
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
    """Skip the test if Space-Track credentials are not set in the environment.

    Integration tests that call the live Space-Track API must request this
    fixture.  When SPACETRACK_USER or SPACETRACK_PASS is absent, the test is
    skipped rather than failing, so CI without credentials stays green.
    """
    user = os.environ.get("SPACETRACK_USER", "")
    pw   = os.environ.get("SPACETRACK_PASS", "")
    if not user or not pw:
        pytest.skip(
            "Space-Track credentials not set. "
            "Export SPACETRACK_USER and SPACETRACK_PASS to run integration tests."
        )
    return {"user": user, "password": pw}
