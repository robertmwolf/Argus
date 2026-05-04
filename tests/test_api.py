"""Tests for the ARGUS FastAPI application (api/main.py)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.queue import InMemoryQueue
from api.storage import LocalStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SYNTH_FITS = Path("data/sample/synth_streak_000.fits")

# Minimal valid FITS-like bytes (magic bytes + padding)
_FITS_MAGIC_BYTES = b"SIMPLE  " + b" " * 2872  # 2880-byte FITS block


def _make_app(tmp_path: Path):
    """Return a fresh app instance wired to isolated in-memory DB and tmp storage."""
    from api.queue import InMemoryQueue
    from api.storage import LocalStorage
    from db.models import get_engine, get_session_factory, init_db

    engine = get_engine("sqlite+aiosqlite:///:memory:")
    queue = InMemoryQueue()
    storage = LocalStorage(tmp_path)
    return engine, queue, storage


@pytest_asyncio.fixture
async def client(tmp_path):
    """Async HTTP client with a fully isolated app instance."""
    from db.models import get_engine, get_session_factory, init_db

    engine = get_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    session_factory = get_session_factory(engine)
    queue = InMemoryQueue()
    storage = LocalStorage(tmp_path)

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.queue = queue
    app.state.storage = storage
    app.state.model_loaded = False

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    await engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_valid_fits_returns_job_id(client, tmp_path):
    """POST /api/upload with a valid FITS file should return 200 and a job_id."""
    response = await client.post(
        "/api/upload",
        files={"file": ("test.fits", _FITS_MAGIC_BYTES, "application/octet-stream")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert len(body["job_id"]) == 36  # UUID format


@pytest.mark.asyncio
async def test_upload_oversized_file_returns_413(client):
    """POST /api/upload with a file exceeding the size limit should return 413."""
    import api.main as api_main
    with patch.object(api_main, "_MAX_UPLOAD_BYTES", 1024):  # 1 KB limit for the test
        big_data = b"X" * 2048
        response = await client.post(
            "/api/upload",
            files={"file": ("big.fits", big_data, "application/octet-stream")},
        )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_upload_invalid_extension_returns_422(client):
    """POST /api/upload with an unsupported extension should return 422."""
    response = await client.post(
        "/api/upload",
        files={"file": ("image.jpg", b"some_data", "image/jpeg")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_result_unknown_job_returns_404(client):
    """GET /api/result for an unknown job_id should return 404."""
    response = await client.get("/api/result/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_health_endpoint_returns_expected_keys(client):
    """GET /health should return 200 with status, model_loaded, and db_connected."""
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "db_connected" in body
    assert body["db_connected"] is True


@pytest.mark.asyncio
async def test_full_upload_poll_result_cycle(client, tmp_path):
    """Integration: upload FITS → process with mocked pipeline → verify detections."""
    if not _SYNTH_FITS.exists():
        pytest.skip("data/sample/synth_streak_000.fits not found")

    fake_detection = {
        "confidence": 0.92,
        "bbox": [10.0, 20.0, 300.0, 40.0],
        "obb": {"cx": 155.0, "cy": 30.0, "w": 295.0, "h": 12.0, "angle_deg": 5.0},
        "streak_length_px": 295.0,
        "ra_deg": 120.0,
        "dec_deg": -5.0,
        "identifications": [],
    }

    # Upload
    fits_bytes = _SYNTH_FITS.read_bytes()
    response = await client.post(
        "/api/upload",
        files={"file": ("synth.fits", fits_bytes, "application/octet-stream")},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # Drain the queue and run the worker synchronously inside the mock context
    from api.main import _process_job, app as _app

    with patch("inference.pipeline.run", return_value=[fake_detection]):
        queued_id = await _app.state.queue.dequeue()
        await _process_job(queued_id, _app)

    # Result should now be complete
    body = (await client.get(f"/api/result/{job_id}")).json()
    assert body["status"] == "complete"
    assert len(body["detections"]) == 1
    assert body["detections"][0]["confidence"] == pytest.approx(0.92)
