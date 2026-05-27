"""Tests for the ARGUS FastAPI application (api/main.py)."""

from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
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


def _make_jpeg_bytes() -> bytes:
    """Return a tiny valid JPEG image for upload tests."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), color=(32, 64, 96)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_app(tmp_path: Path):
    """Return a fresh app instance wired to isolated in-memory DB and tmp storage."""
    from api.queue import InMemoryQueue
    from api.storage import LocalStorage
    from db.models import get_engine, get_session_factory, init_db

    engine = get_engine("sqlite+aiosqlite:///:memory:")
    queue = InMemoryQueue()
    storage = LocalStorage(tmp_path)
    return engine, queue, storage


def test_copy_matching_wcs_sidecar_from_upload_storage(tmp_path, monkeypatch):
    """Temporary pipeline FITS files should receive the uploaded .wcs sidecar."""
    from api.main import _copy_matching_wcs_sidecar

    monkeypatch.chdir(tmp_path)
    job_id = "job-1"
    upload_dir = tmp_path / "data" / "uploads" / job_id
    upload_dir.mkdir(parents=True)
    (upload_dir / "image.wcs").write_text("WCS SIDE")
    temp_fits = tmp_path / "tmpabc.fits"
    temp_fits.write_bytes(_FITS_MAGIC_BYTES)

    copied = _copy_matching_wcs_sidecar("image.fits", temp_fits, job_id)

    assert copied == temp_fits.with_suffix(".wcs")
    assert copied.read_text() == "WCS SIDE"


def test_configure_api_logging_adds_file_handler_once(tmp_path, monkeypatch):
    """API file logging should be enabled without duplicate handlers."""
    import logging
    from api.main import _configure_api_logging

    log_path = tmp_path / "api.log"
    monkeypatch.setenv("ARGUS_API_LOG", str(log_path))

    before = len(logging.getLogger().handlers)
    _configure_api_logging()
    _configure_api_logging()

    matching = [
        handler for handler in logging.getLogger().handlers
        if isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path
    ]
    assert len(matching) == 1
    assert len(logging.getLogger().handlers) == before + 1

    logging.getLogger().removeHandler(matching[0])
    matching[0].close()


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
    app.state.pipeline_model = None
    app.state.pipeline_device = None
    app.state.pipeline_model_key = None

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    await engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_startup_jobs_requeues_queued_and_processing(tmp_path):
    """Startup recovery should revive stranded in-memory queue jobs."""
    from api.main import _recover_startup_jobs, app as _app
    from db.models import Observation, get_engine, get_session_factory, init_db

    engine = get_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    session_factory = get_session_factory(engine)
    queue = InMemoryQueue()

    _app.state.session_factory = session_factory
    _app.state.queue = queue

    async with session_factory() as session:
        session.add_all([
            Observation(id="job-queued", filename="queued.fits", status="queued"),
            Observation(id="job-processing", filename="processing.fits", status="processing"),
            Observation(id="job-complete", filename="complete.fits", status="complete"),
            Observation(id="job-failed", filename="failed.fits", status="failed"),
        ])
        await session.commit()

    recovered = await _recover_startup_jobs(_app)

    assert recovered == 2
    recovered_ids = {await queue.dequeue(), await queue.dequeue()}
    assert recovered_ids == {"job-queued", "job-processing"}

    async with session_factory() as session:
        queued = await session.get(Observation, "job-queued")
        processing = await session.get(Observation, "job-processing")
        complete = await session.get(Observation, "job-complete")
        failed = await session.get(Observation, "job-failed")

    assert queued.status == "queued"
    assert processing.status == "queued"
    assert complete.status == "complete"
    assert failed.status == "failed"
    await engine.dispose()


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
async def test_upload_valid_jpeg_returns_job_id_and_preview(client):
    """POST /api/upload with a valid JPEG file should return 200 and save a PNG preview."""
    response = await client.post(
        "/api/upload",
        files={"file": ("image.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "queued"

    preview = await app.state.storage.load_preview(body["job_id"])
    assert preview is not None
    assert preview.startswith(b"\x89PNG\r\n\x1a\n")


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
        files={"file": ("image.gif", b"some_data", "image/gif")},
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
    assert "space_track_data_refreshed_at" in body
    assert "db_connected" in body
    assert body["db_connected"] is True


@pytest.mark.asyncio
async def test_full_upload_poll_result_cycle(client, tmp_path):
    """Integration: upload FITS → process with mocked pipeline → verify detections."""
    if not _SYNTH_FITS.exists():
        pytest.skip("data/sample/synth_streak_000.fits not found")

    fake_detection = {
        "method": "ml",
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
    queued_body = (await client.get(f"/api/result/{job_id}")).json()
    assert queued_body["obs_epoch"] == "2024-04-02T02:00:00Z"

    # Drain the queue and run the worker synchronously inside the mock context
    from api.main import _process_job, app as _app

    _fake_array = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("inference.pipeline.load_model", return_value=(object(), object())), \
         patch("inference.pipeline.run_with_array", return_value=([fake_detection], _fake_array)):
        queued_id = await _app.state.queue.dequeue()
        await _process_job(queued_id, _app)

    # Result should now be complete
    body = (await client.get(f"/api/result/{job_id}")).json()
    assert body["status"] == "complete"
    assert body["obs_epoch"] == "2024-04-02T02:00:00Z"
    assert body["image_width"] is not None
    assert body["image_height"] is not None
    assert len(body["detections"]) == 1
    det = body["detections"][0]
    # Top-level method is always "unified"; individual method is in sources
    assert det["method"] == "unified"
    # A single detector keeps its own confidence; reliability weights only
    # affect corroboration boosts from additional non-ASTRiDE detectors.
    assert det["confidence"] == pytest.approx(0.92)
    sources = det["sources"]
    assert sources[0]["method"] == "unified"
    assert sources[1]["method"] == "ml"


@pytest.mark.asyncio
async def test_upload_detector_selection_is_passed_to_pipeline(client):
    """Selected upload detectors should be persisted and used by the worker."""
    response = await client.post(
        "/api/upload",
        files={"file": ("image.jpg", _make_jpeg_bytes(), "image/jpeg")},
        data={"enabled_detectors": '["classical"]'},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    from api.main import _process_job, app as _app

    _fake_array = np.zeros((8, 6, 3), dtype=np.uint8)
    with patch("inference.pipeline.load_model") as mock_load_model, \
         patch("inference.pipeline.resolve_model_specs", return_value=[
             {"id": "dinov3_vitb", "size": "dinov3_vitb"},
         ]), \
         patch("inference.pipeline.run_with_array", return_value=([], _fake_array)) as mock_run:
        queued_id = await _app.state.queue.dequeue()
        await _process_job(queued_id, _app)

    assert queued_id == job_id
    mock_load_model.assert_not_called()
    _, kwargs = mock_run.call_args
    assert kwargs["enabled_detectors"] == {"classical"}
    assert kwargs["models"] == []
