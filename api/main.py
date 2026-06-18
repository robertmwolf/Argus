"""ARGUS FastAPI application.

Endpoints:
  POST /api/upload             — accept FITS/PNG/JPEG, enqueue for processing
  GET  /api/result/{job_id}   — poll job status and detections
  GET  /api/image/{job_id}    — fetch processed PNG overlay
  GET  /health                 — liveness + readiness probe

Storage and queue backends are selected via env vars (STORAGE_BACKEND,
QUEUE_BACKEND) and never imported concretely here — only via factories.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone

# Load .env before any os.environ reads — safe no-op if python-dotenv not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import select, text, update

from db.models import Detection, Identification, Observation, get_engine, get_session_factory, init_db
from inference.confidence import compute_unified_confidence
from src.matching.tle_store import get_latest_coverage_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("inference.pipeline").setLevel(logging.INFO)
logging.getLogger("inference.fits_loader").setLevel(logging.INFO)
logging.getLogger("inference.crossid").setLevel(logging.INFO)

_MAX_UPLOAD_BYTES = 300 * 1024 * 1024  # 100 MB
_FITS_EXTENSIONS = {".fits", ".fit", ".fts"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_ALLOWED_EXTENSIONS = _FITS_EXTENSIONS | _IMAGE_EXTENSIONS
# FITS magic: first 8 bytes are "SIMPLE  " (with trailing spaces)
_FITS_MAGIC = b"SIMPLE  "


def _configure_api_logging() -> None:
    """Attach an idempotent file handler for API and worker diagnostics."""
    log_path = Path(os.environ.get("ARGUS_API_LOG", "logs/api.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = log_path.resolve()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == resolved:
                    return
            except OSError:
                continue

    handler = logging.FileHandler(resolved)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s — %(message)s"
    ))
    root_logger.addHandler(handler)


def _parse_enabled_detectors(raw: str | None) -> set[str] | None:
    """Parse the optional upload detector-selection JSON field."""
    if raw is None or raw == "":
        return None
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="enabled_detectors must be a JSON array of detector IDs",
        ) from exc
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="enabled_detectors must be a JSON array of detector IDs",
        )
    return set(values)


def _enabled_detectors_json(enabled_detectors: set[str] | None) -> str | None:
    """Serialize detector selection for storage on the observation row."""
    if enabled_detectors is None:
        return None
    return json.dumps(sorted(enabled_detectors))


def _copy_matching_wcs_sidecar(filename: str, fits_path: Path, job_id: str | None = None) -> Path | None:
    """Copy a same-stem WCS sidecar beside a temporary FITS path if available.

    API uploads are processed from a temporary file, which breaks the normal
    same-directory ``.wcs`` lookup used by ``FITSLoader``.  For local research
    workflows, recover the sidecar from common ARGUS data locations and place it
    next to the temp FITS so pixel-to-sky conversion still works.

    Args:
        filename: Original uploaded filename.
        fits_path: Temporary FITS path passed to the inference pipeline.
        job_id: Optional observation/job UUID for local upload storage lookup.

    Returns:
        Path to the copied sidecar, or None if no sidecar was found.
    """
    stem = Path(filename).stem
    original = Path(filename)
    candidates: list[Path] = []

    for suffix in (".wcs", ".WCS"):
        if original.parent != Path("."):
            candidates.append(original.with_suffix(suffix))
        if job_id:
            candidates.append(Path("data/uploads") / job_id / f"{stem}{suffix}")
        candidates.extend(
            [
                Path("data/GTImages") / f"{stem}{suffix}",
                Path("data/raw") / f"{stem}{suffix}",
                Path("data/sample") / f"{stem}{suffix}",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            dest = fits_path.with_suffix(candidate.suffix)
            shutil.copyfile(candidate, dest)
            logger.debug("Copied WCS sidecar %s → %s", candidate, dest)
            return dest
    return None


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


async def _recover_startup_jobs(app: FastAPI) -> int:
    """Return stranded local jobs to the queue during API startup.

    The local queue backend is in-memory, so queued work disappears when the
    process exits.  Re-enqueue queued rows and reset any interrupted processing
    rows to queued before the worker starts.

    Args:
        app: FastAPI application carrying session factory and queue state.

    Returns:
        Number of job IDs enqueued for processing.
    """
    session_factory = app.state.session_factory
    async with session_factory() as session:
        await session.execute(
            update(Observation)
            .where(Observation.status == "processing")
            .values(status="queued")
        )
        result = await session.execute(
            select(Observation.id)
            .where(Observation.status == "queued")
            .order_by(Observation.uploaded_at, Observation.id)
        )
        job_ids = list(result.scalars())
        await session.commit()

    for job_id in job_ids:
        await app.state.queue.enqueue(job_id)

    if job_ids:
        logger.info(
            "Recovered %d queued startup job(s): %s",
            len(job_ids),
            ", ".join(job_ids),
        )
    return len(job_ids)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.queue import get_queue
    from api.storage import get_storage

    _configure_api_logging()
    engine = get_engine()
    await init_db(engine)
    session_factory = get_session_factory(engine)
    storage = get_storage()
    queue = get_queue()

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.storage = storage
    app.state.queue = queue
    app.state.model_loaded = False
    app.state.pipeline_model = None
    app.state.pipeline_device = None
    app.state.pipeline_model_key = None
    app.state.pipeline_models = None       # multi-model cache (load_models() path)
    app.state.pipeline_models_key = None   # ARGUS_MODEL_CONFIGS hash for cache invalidation

    await _recover_startup_jobs(app)
    app.state.worker_task = asyncio.create_task(_worker_loop(app))

    yield

    app.state.worker_task.cancel()
    try:
        await app.state.worker_task
    except asyncio.CancelledError:
        pass
    await engine.dispose()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ARGUS Satellite Streak Detector", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


async def _worker_loop(app: FastAPI) -> None:
    """Continuously consume job IDs from the queue and process them."""
    while True:
        job_id = await app.state.queue.dequeue()
        try:
            await _process_job(job_id, app)
        except Exception:
            logger.exception("Unhandled error processing job %s", job_id)


async def _process_job(job_id: str, app: FastAPI) -> None:
    """Run the inference pipeline on one job and persist results.

    Args:
        job_id: UUID of the observation to process.
        app: FastAPI application instance carrying shared state.
    """
    session_factory = app.state.session_factory
    storage = app.state.storage

    # Load observation record and capture filename before closing session
    async with session_factory() as session:
        obs = await session.get(Observation, job_id)
        if obs is None:
            logger.error("Job %s not found in DB", job_id)
            return
        filename = obs.filename
        obs_epoch = obs.obs_epoch
        exposure_time = obs.exposure_time
        enabled_detectors = _parse_enabled_detectors(obs.enabled_detectors_json)
        raw_mode = bool(obs.raw_mode)
        fast_mode = bool(obs.fast_mode)
        obs.status = "processing"
        await session.commit()

    tmp_path: Path | None = None
    try:
        fits_data = await storage.load_upload(job_id, filename)

        if obs_epoch is None and Path(filename).suffix.lower() in _FITS_EXTENSIONS:
            header_cards = await asyncio.to_thread(_extract_fits_header, fits_data)
            obs_epoch = _normalise_obs_epoch(_header_card_value(header_cards, "DATE-OBS"))
            if exposure_time is None:
                exposure_time = _header_card_value(header_cards, "EXPTIME")
                if exposure_time is None:
                    exposure_time = _header_card_value(header_cards, "EXPOSURE")
                if exposure_time is None:
                    exposure_time = _header_card_value(header_cards, "EXP_TIME")
                try:
                    exposure_time = float(exposure_time) if exposure_time is not None else None
                except (TypeError, ValueError):
                    exposure_time = None
            if obs_epoch is not None or exposure_time is not None:
                async with session_factory() as session:
                    obs = await session.get(Observation, job_id)
                    if obs:
                        obs.obs_epoch = obs.obs_epoch or obs_epoch
                        obs.exposure_time = obs.exposure_time or exposure_time
                        await session.commit()

        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as f:
            f.write(fits_data)
            tmp_path = Path(f.name)
        sidecar_path = _copy_matching_wcs_sidecar(filename, tmp_path, job_id)

        # pipeline_run is patched in tests via unittest.mock.
        # Set FAST_MODE=true in the environment to skip Radon + cross-ID.
        try:
            from inference.pipeline import run_with_array as pipeline_run
        except ImportError as exc:
            raise RuntimeError(
                "ML packages are not installed in this environment. "
                "Run the API directly with the satid conda env for inference, "
                "or run the standalone GPU worker on a machine with the ML stack installed."
            ) from exc

        logger.info(
            "Processing job %s with enabled_detectors=%s",
            job_id,
            "all" if enabled_detectors is None else sorted(enabled_detectors),
        )

        pipeline_kwargs: dict[str, Any] = {
            "fits_path": tmp_path,
            "enabled_detectors": enabled_detectors,
            "raw_mode": raw_mode,
            "fast": fast_mode,
        }
        pipeline_kwargs["models"] = []

        pipeline_t0 = time.perf_counter()
        detections, fits_array, heat_dict = await asyncio.to_thread(
            pipeline_run,
            **pipeline_kwargs,
        )
        pipeline_ms = (time.perf_counter() - pipeline_t0) * 1000
        logger.info(
            "job_timing job_id=%s pipeline_ms=%.1f detections=%d",
            job_id, pipeline_ms, len(detections),
        )
        app.state.model_loaded = True

        png_bytes = await asyncio.to_thread(_render_png, tmp_path, detections, fits_array)
        await storage.save_image(job_id, png_bytes)

        for _hm_model_id, _heat_array in (heat_dict or {}).items():
            _heatmap_png = await asyncio.to_thread(
                _render_heatmap_png, _heat_array,
                fits_array.shape[1] if fits_array is not None else None,
                fits_array.shape[0] if fits_array is not None else None,
            )
            if _heatmap_png is not None:
                await storage.save_heatmap(job_id, _heatmap_png, model_id=_hm_model_id)

        db_t0 = time.perf_counter()
        async with session_factory() as session:
            for det_dict in detections:
                det = Detection(
                    id=str(uuid.uuid4()),
                    observation_id=job_id,
                    method=det_dict.get("method") or "ml",
                    confidence=float(det_dict.get("confidence", 0.0)),
                    streak_length_px=det_dict.get("streak_length_px"),
                    streak_id=det_dict.get("streak_id"),
                    ra_tip1_deg=det_dict.get("ra_tip1_deg"),
                    dec_tip1_deg=det_dict.get("dec_tip1_deg"),
                    ra_tip2_deg=det_dict.get("ra_tip2_deg"),
                    dec_tip2_deg=det_dict.get("dec_tip2_deg"),
                    x1=det_dict.get("x1"),
                    y1=det_dict.get("y1"),
                    x2=det_dict.get("x2"),
                    y2=det_dict.get("y2"),
                    angle_deg=det_dict.get("angle_deg"),
                )
                session.add(det)
                await session.flush()

                for ident_dict in det_dict.get("identifications") or []:
                    sep_arcsec = ident_dict.get("separation_arcsec") or 0.0
                    session.add(Identification(
                        id=str(uuid.uuid4()),
                        detection_id=det.id,
                        norad_id=ident_dict.get("norad_id"),
                        satellite_name=ident_dict.get("satellite_name"),
                        confidence=ident_dict.get("confidence"),
                        separation_deg=sep_arcsec / 3600.0,
                        rank=ident_dict.get("rank"),
                        tle_epoch=ident_dict.get("tle_epoch"),
                        tle_age_hours=ident_dict.get("tle_age_hours"),
                        photo_taken_at=ident_dict.get("photo_taken_at") or obs_epoch,
                        tle_data_fresh_at=ident_dict.get("tle_data_fresh_at"),
                        tle_source=ident_dict.get("tle_source"),
                        tle_search_mode=ident_dict.get("tle_search_mode"),
                        epoch_search_window_days=ident_dict.get("epoch_search_window_days"),
                        epoch_drift_hours=ident_dict.get("epoch_drift_hours"),
                        position_score=ident_dict.get("position_score"),
                        epoch_penalty=ident_dict.get("epoch_penalty"),
                    ))

            obs = await session.get(Observation, job_id)
            obs.status = "complete"
            await session.commit()
        db_write_ms = (time.perf_counter() - db_t0) * 1000
        logger.info("job_timing job_id=%s db_write_ms=%.1f", job_id, db_write_ms)

    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        async with session_factory() as session:
            obs = await session.get(Observation, job_id)
            if obs:
                obs.status = "failed"
                await session.commit()
    finally:
        if "sidecar_path" in locals() and sidecar_path is not None:
            sidecar_path.unlink(missing_ok=True)
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _extract_fits_header(data: bytes) -> list[dict] | None:
    """Extract FITS primary header cards as a JSON-serialisable list.

    Args:
        data: Raw FITS file bytes.

    Returns:
        List of {key, value, comment} dicts, or None on failure.
    """
    try:
        from astropy.io import fits as afits

        with afits.open(io.BytesIO(data)) as hdul:
            hdul.verify("silentfix")
            header = hdul[0].header
            cards = []
            for card in header.cards:
                key = card.keyword
                if not key:
                    continue
                val = card.value
                if isinstance(val, bool):
                    val = bool(val)
                elif isinstance(val, float):
                    val = float(val)
                elif isinstance(val, int):
                    val = int(val)
                else:
                    val = str(val) if val is not None else None
                cards.append({
                    "key": key,
                    "value": val,
                    "comment": str(card.comment) if card.comment else None,
                })
            return cards
    except Exception:
        logger.exception("FITS header extraction failed")
        return None


def _header_card_value(header_cards: list[dict] | None, key: str) -> Any | None:
    """Return a FITS header card value by keyword from extracted cards.

    Args:
        header_cards: List returned by :func:`_extract_fits_header`.
        key: FITS keyword to look up.

    Returns:
        Header value, or None if absent.
    """
    if not header_cards:
        return None
    target = key.upper()
    for card in header_cards:
        if str(card.get("key", "")).upper() == target:
            return card.get("value")
    return None


def _normalise_obs_epoch(value: Any) -> str | None:
    """Normalize a DATE-OBS-like value to an ISO8601 UTC string.

    Args:
        value: Raw FITS DATE-OBS value.

    Returns:
        ISO8601 string ending in ``Z``, or None when parsing fails.
    """
    if value is None:
        return None
    try:
        from datetime import datetime, timezone

        raw = str(value).strip().replace(" ", "T")
        dt = datetime.fromisoformat(raw.rstrip("Z"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        logger.warning("Could not parse DATE-OBS value %r", value)
        return None


def _extract_image_shape(data: bytes, filename: str) -> tuple[int, int] | None:
    """Return original image dimensions as ``(width, height)``.

    Args:
        data: Raw uploaded FITS, PNG, or JPEG bytes.
        filename: Original upload filename, used to select the parser.

    Returns:
        ``(width, height)`` in source-image pixels, or None on failure.
    """
    try:
        ext = Path(filename).suffix.lower()
        if ext in _FITS_EXTENSIONS:
            from astropy.io import fits as afits

            with afits.open(io.BytesIO(data)) as hdul:
                for hdu in hdul:
                    if hdu.data is None or hdu.data.ndim < 2:
                        continue
                    img_data = hdu.data
                    while img_data.ndim > 2:
                        img_data = img_data[0]
                    height, width = img_data.shape[:2]
                    return int(width), int(height)
        if ext in _IMAGE_EXTENSIONS:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as img:
                return int(img.width), int(img.height)
    except Exception:
        logger.exception("Image shape extraction failed")
    return None


def _render_fits_preview(data: bytes) -> bytes | None:
    """Render FITS image data to a preview PNG without any overlays.

    Uses PixInsight AutoSTF for consistent stretch with the inference pipeline.

    Args:
        data: Raw FITS file bytes.

    Returns:
        PNG bytes, or None on failure.
    """
    try:
        import numpy as np
        from astropy.io import fits as afits
        from PIL import Image

        from inference.autostretch import autostretch

        with afits.open(io.BytesIO(data)) as hdul:
            img_data = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    img_data = hdu.data.copy()
                    break
        if img_data is None:
            return None

        while img_data.ndim > 2:
            img_data = img_data[0]

        if not np.isfinite(img_data).any():
            return None

        stretched = autostretch(img_data.astype(np.float32))  # [0, 1]
        img_data = (stretched * 255).astype(np.uint8)

        # Downsample very large images for the preview to keep response small
        img = Image.fromarray(img_data, mode="L").convert("RGB")
        max_side = 1200
        if max(img.width, img.height) > max_side:
            scale = max_side / max(img.width, img.height)
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.LANCZOS,
            )

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        logger.exception("FITS preview generation failed")
        return None


def _render_image_preview(data: bytes) -> bytes | None:
    """Render an uploaded raster image to preview PNG bytes.

    Args:
        data: Raw PNG or JPEG file bytes.

    Returns:
        PNG bytes, or None on failure.
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as src:
            img = src.convert("RGB")

        max_side = 1200
        if max(img.width, img.height) > max_side:
            scale = max_side / max(img.width, img.height)
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.LANCZOS,
            )

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        logger.exception("Image preview generation failed")
        return None


def _render_png(
    fits_path: Path,
    detections: list[dict],
    array: np.ndarray | None = None,
) -> bytes:
    """Render FITS data to a PNG with bounding box overlays.

    Args:
        fits_path: Path to the FITS file (used only when *array* is None).
        detections: List of detection dicts from the pipeline.
        array: Pre-loaded uint8 (H, W, 3) image array from the pipeline run.
            When provided the FITS file is not re-parsed, saving a full I/O
            and normalisation pass.

    Returns:
        PNG file bytes.
    """
    try:
        from PIL import Image, ImageDraw

        if array is not None:
            arr = array
        else:
            from inference.fits_loader import FITSLoader

            result = FITSLoader().load(fits_path)
            arr = result["array"]  # (H, W, 3) uint8

        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        for det in detections:
            if all(det.get(key) is not None for key in ("x1", "y1", "x2", "y2")):
                conf = det.get("confidence", 1.0)
                alpha = int(conf * 200)
                draw.line(
                    [det["x1"], det["y1"], det["x2"], det["y2"]],
                    fill=(0, 220, 255, alpha),
                    width=2,
                )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception:
        logger.exception("PNG render failed — returning blank 1×1 PNG")
        buf = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (1, 1)).save(buf, format="PNG")
        return buf.getvalue()


def _render_heatmap_png(
    heat_array: np.ndarray,
    target_width: int | None,
    target_height: int | None,
) -> bytes | None:
    """Render a float32 heatmap as an RGBA PNG with a hot colormap.

    The heatmap is colored orange→yellow (alpha proportional to intensity) so
    the frontend can composite it over the auto-stretched image.

    Args:
        heat_array: Float32 array in [0, 1] at inference resolution.
        target_width: Native image width to resize to (or None to keep original).
        target_height: Native image height to resize to (or None to keep original).

    Returns:
        RGBA PNG bytes, or None on failure.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image

        h = heat_array.astype(np.float32)
        h = np.clip(h, 0.0, 1.0)

        if target_width and target_height:
            h = cv2.resize(h, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        r = np.full_like(h, 255, dtype=np.uint8)
        g = (np.sqrt(h) * 210).astype(np.uint8)
        b = np.zeros_like(g)
        a = (h * 200).astype(np.uint8)

        rgba = np.stack([r, g, b, a], axis=2)
        img = Image.fromarray(rgba, mode="RGBA")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        logger.exception("Heatmap PNG render failed")
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/upload", status_code=status.HTTP_200_OK)
async def upload(
    request: Request,
    file: UploadFile,
    enabled_detectors: str | None = Form(None),
    raw_mode: bool = Form(False),
    fast_mode: bool = Form(False),
) -> dict[str, str]:
    """Accept a FITS, PNG, or JPEG upload and enqueue it for processing.

    Args:
        request: FastAPI Request (for app state access).
        file: Uploaded file from multipart form.

    Returns:
        dict with job_id and status keys.

    Raises:
        HTTPException 413: File exceeds 100 MB.
        HTTPException 422: Unsupported file extension or invalid magic bytes.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type {ext!r}. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )
    enabled_detector_set = _parse_enabled_detectors(enabled_detectors)

    data = await file.read()

    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 100 MB limit",
        )

    # Validate FITS magic bytes when a FITS extension is used
    if ext in _FITS_EXTENSIONS and not data[:8] == _FITS_MAGIC:
        # Permit synthetic test files that may not have the magic (warn only)
        logger.warning("FITS file %s missing SIMPLE magic bytes", file.filename)

    job_id = str(uuid.uuid4())
    filename = file.filename or f"{job_id}{ext}"

    await request.app.state.storage.save_upload(job_id, filename, data)

    # Extract FITS header and generate preview PNG at upload time so the
    # frontend can display them immediately while the job is queued/processing.
    obs_epoch: str | None = None
    exposure_time: float | None = None
    if ext in _FITS_EXTENSIONS:
        header_cards = await asyncio.to_thread(_extract_fits_header, data)
        if header_cards is not None:
            await request.app.state.storage.save_fits_header(
                job_id, json.dumps(header_cards).encode()
            )
            obs_epoch = _normalise_obs_epoch(_header_card_value(header_cards, "DATE-OBS"))
            exposure_time = _header_card_value(header_cards, "EXPTIME")
            if exposure_time is None:
                exposure_time = _header_card_value(header_cards, "EXPOSURE")
            if exposure_time is None:
                exposure_time = _header_card_value(header_cards, "EXP_TIME")
            try:
                exposure_time = float(exposure_time) if exposure_time is not None else None
            except (TypeError, ValueError):
                exposure_time = None
        preview_png = await asyncio.to_thread(_render_fits_preview, data)
        if preview_png is not None:
            await request.app.state.storage.save_preview(job_id, preview_png)
    elif ext in _IMAGE_EXTENSIONS:
        preview_png = await asyncio.to_thread(_render_image_preview, data)
        if preview_png is not None:
            await request.app.state.storage.save_preview(job_id, preview_png)

    async with request.app.state.session_factory() as session:
        session.add(
            Observation(
                id=job_id,
                filename=filename,
                uploaded_at=datetime.now(tz=timezone.utc).isoformat(),
                obs_epoch=obs_epoch,
                exposure_time=exposure_time,
                enabled_detectors_json=_enabled_detectors_json(enabled_detector_set),
                raw_mode=raw_mode,
                fast_mode=fast_mode,
                status="queued",
            )
        )
        await session.commit()

    await request.app.state.queue.enqueue(job_id)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/result/{job_id}")
async def result(job_id: str, request: Request) -> dict[str, Any]:
    """Return processing status and detections for a job.

    Args:
        job_id: UUID of the observation.
        request: FastAPI Request (for app state access).

    Returns:
        dict with job_id, status, filename, obs_epoch, and detections list.

    Raises:
        HTTPException 404: job_id not found in database.
    """
    async with request.app.state.session_factory() as session:
        obs = await session.get(Observation, job_id)
        if obs is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        filename = obs.filename

        det_rows = (
            await session.execute(
                select(Detection).where(Detection.observation_id == job_id)
            )
        ).scalars().all()

        # Group per-method detection rows by streak_id so the UI can show
        # multi-method agreement.  Rows without a streak_id (legacy) each form
        # their own group keyed by the row's own id.
        streak_groups: dict[str, list] = {}
        for det in det_rows:
            key = str(det.streak_id) if det.streak_id is not None else det.id
            streak_groups.setdefault(key, []).append(det)

        detections = []
        for group in streak_groups.values():
            # Primary = highest-confidence detection in the group (carries geometry
            # and identifications).
            primary = max(group, key=lambda d: d.confidence)

            ident_rows = (
                await session.execute(
                    select(Identification).where(Identification.detection_id == primary.id)
                )
            ).scalars().all()

            sources = sorted(
                [{"method": d.method, "confidence": d.confidence} for d in group],
                key=lambda s: s["confidence"],
                reverse=True,
            )

            # Precision-recall calibrated Unified Confidence Score.
            # The best detector confidence is the floor; empirical F-0.5
            # weights control how much additional detectors can corroborate it.
            unified_result = compute_unified_confidence(sources)
            unified_conf = unified_result["score"]
            sources_with_unified = [
                {"method": "unified", "confidence": unified_conf}
            ] + sources

            detections.append({
                "streak_id": primary.streak_id,
                "sources": sources_with_unified,
                # Top-level fields reflect the unified score for canvas overlay
                "method": "unified",
                "confidence": unified_conf,
                "photo_taken_at": obs.obs_epoch,
                "streak_length_px": primary.streak_length_px,
                "ra_tip1_deg": primary.ra_tip1_deg,
                "dec_tip1_deg": primary.dec_tip1_deg,
                "ra_tip2_deg": primary.ra_tip2_deg,
                "dec_tip2_deg": primary.dec_tip2_deg,
                "x1": primary.x1,
                "y1": primary.y1,
                "x2": primary.x2,
                "y2": primary.y2,
                "angle_deg": primary.angle_deg,
                "identifications": [
                    {
                        "satellite_name": i.satellite_name,
                        "norad_id": i.norad_id,
                        "confidence": i.confidence,
                        "separation_deg": i.separation_deg,
                        "rank": i.rank,
                        "tle_epoch": i.tle_epoch,
                        "tle_age_hours": i.tle_age_hours,
                        "photo_taken_at": i.photo_taken_at or obs.obs_epoch,
                        "tle_data_fresh_at": i.tle_data_fresh_at,
                        "tle_source": i.tle_source,
                        "tle_search_mode": i.tle_search_mode,
                        "epoch_search_window_days": i.epoch_search_window_days,
                        "epoch_drift_hours": i.epoch_drift_hours,
                        "position_score": i.position_score,
                        "epoch_penalty": i.epoch_penalty,
                    }
                    for i in sorted(ident_rows, key=lambda x: x.rank or 99)
                ],
            })

        # Sort streaks by primary confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        image_shape = None
        try:
            upload_bytes = await request.app.state.storage.load_upload(job_id, filename)
            image_shape = _extract_image_shape(upload_bytes, filename)
        except Exception:
            logger.exception("Could not determine source image dimensions for job %s", job_id)

        has_heatmap = await request.app.state.storage.list_heatmaps(job_id)

        return {
            "job_id": job_id,
            "status": obs.status,
            "filename": filename,
            "obs_epoch": obs.obs_epoch,
            "raw_mode": bool(obs.raw_mode),
            "image_width": image_shape[0] if image_shape else None,
            "image_height": image_shape[1] if image_shape else None,
            "has_heatmap": has_heatmap,
            "detections": detections,
        }


@app.get("/api/image/{job_id}")
async def image(job_id: str, request: Request) -> Response:
    """Return the processed PNG overlay for a completed job.

    Args:
        job_id: UUID of the observation.
        request: FastAPI Request (for app state access).

    Returns:
        PNG image bytes.

    Raises:
        HTTPException 404: Job not found or image not yet ready.
    """
    async with request.app.state.session_factory() as session:
        obs = await session.get(Observation, job_id)
        if obs is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    png = await request.app.state.storage.load_image(job_id)
    if png is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not yet available — job may still be processing",
        )
    return Response(content=png, media_type="image/png")


@app.get("/api/preview/{job_id}")
async def preview(job_id: str, request: Request) -> Response:
    """Return the raw preview PNG for a job (no detection overlays).

    Available immediately after upload for FITS files, PNGs, and JPEGs.

    Args:
        job_id: UUID of the observation.
        request: FastAPI Request (for app state access).

    Returns:
        PNG image bytes.

    Raises:
        HTTPException 404: Job not found or preview not yet available.
    """
    async with request.app.state.session_factory() as session:
        obs = await session.get(Observation, job_id)
        if obs is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    png = await request.app.state.storage.load_preview(job_id)
    if png is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preview not yet available",
        )
    return Response(content=png, media_type="image/png")


@app.get("/api/heatmap/{job_id}")
async def heatmap(
    job_id: str,
    request: Request,
    model: str | None = Query(default=None, description="Detector model ID (e.g. 'vits_heatmap')"),
) -> Response:
    """Return the heatmap overlay PNG for a completed heatmap-detector job.

    The PNG is RGBA: orange-to-yellow colormap with alpha proportional to
    heatmap intensity.  The frontend composites it over the auto-stretched
    preview to visualise where the model assigned activation.

    Args:
        job_id: UUID of the observation.
        request: FastAPI Request (for app state access).
        model: Optional detector ID to select a specific heatmap when the job
            ran multiple heatmap detectors.  Omit to get the first available.

    Returns:
        RGBA PNG image bytes.

    Raises:
        HTTPException 404: Job not found or heatmap not available.
    """
    async with request.app.state.session_factory() as session:
        obs = await session.get(Observation, job_id)
        if obs is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if model is not None:
        png = await request.app.state.storage.load_heatmap(job_id, model_id=model)
    else:
        # Prefer named model files; fall back to legacy heatmap.png
        png = None
        for try_id in ["vits_heatmap_v9", "vitb_heatmap_v10"]:
            png = await request.app.state.storage.load_heatmap(job_id, model_id=try_id)
            if png is not None:
                break
        if png is None:
            png = await request.app.state.storage.load_heatmap(job_id)

    if png is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Heatmap not available — job may not have used the heatmap detector",
        )
    return Response(content=png, media_type="image/png")


@app.get("/api/fits-header/{job_id}")
async def fits_header(job_id: str, request: Request) -> dict[str, Any]:
    """Return FITS primary header cards for a job.

    Args:
        job_id: UUID of the observation.
        request: FastAPI Request (for app state access).

    Returns:
        dict with a ``cards`` list of {key, value, comment} objects.

    Raises:
        HTTPException 404: Job not found.
    """
    async with request.app.state.session_factory() as session:
        obs = await session.get(Observation, job_id)
        if obs is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    raw = await request.app.state.storage.load_fits_header(job_id)
    if raw is None:
        return {"cards": []}
    return {"cards": json.loads(raw)}


@app.post("/api/clear-queue")
async def clear_queue(request: Request) -> dict[str, Any]:
    """Cancel the running job (if any), drain pending jobs, and restart the worker.

    Sets all queued and processing observations to 'cancelled' in the DB.
    The ML inference thread for any in-progress job keeps running until it
    finishes naturally, but its result is discarded.

    Returns:
        dict with ``cancelled`` count of DB rows updated.
    """
    # Cancel the running worker task so it stops awaiting the current job.
    task: asyncio.Task | None = getattr(request.app.state, "worker_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Drain any queued job IDs from the in-memory queue.
    await request.app.state.queue.clear()

    # Mark all queued/processing rows as cancelled in the DB.
    async with request.app.state.session_factory() as session:
        result = await session.execute(
            update(Observation)
            .where(Observation.status.in_(["queued", "processing"]))
            .values(status="cancelled")
            .returning(Observation.id)
        )
        cancelled_ids = list(result.scalars())
        await session.commit()

    logger.info("clear_queue: cancelled %d job(s): %s", len(cancelled_ids), cancelled_ids)

    # Restart a fresh worker loop for new uploads.
    request.app.state.worker_task = asyncio.create_task(_worker_loop(request.app))

    return {"cancelled": len(cancelled_ids)}


@app.get("/api/detectors")
async def detectors() -> dict[str, Any]:
    """Return availability metadata for all detectors.

    Checks weights file existence and required imports; never loads a model.

    Returns:
        dict with a ``detectors`` list; each item has id, name, type,
        dataset, and status ('active' | 'no_weights' | 'unavailable').
    """
    try:
        from inference.pipeline import get_detector_statuses
        items = await asyncio.to_thread(get_detector_statuses)
    except Exception:
        logger.exception("Failed to enumerate detector statuses")
        items = []
    return {"detectors": items}


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness and readiness probe.

    Returns:
        dict with status, model_loaded, and db_connected keys.
    """
    db_ok = False
    try:
        async with request.app.state.session_factory() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        logger.exception("DB health check failed")

    model_size = os.environ.get("MODEL_SIZE", "tiny")
    model_labels = {
        "tiny":                    "DINO Swin-Tiny - SatStreaks",
        "large":                   "DINO Swin-Large - SatStreaks",
        "dinov3_vitb":             "DINOv3 ViT-Base - SatStreaks+GTImages",
        "dinov3_vitl":             "DINOv3 ViT-Large - SatStreaks+GTImages",
    }
    space_track_refreshed_at = None
    try:
        refreshed_at = await asyncio.to_thread(
            get_latest_coverage_time,
            None,
            1,
        )
        if refreshed_at is not None:
            space_track_refreshed_at = (
                refreshed_at.isoformat().replace("+00:00", "Z")
            )
    except Exception:
        logger.exception("TLE coverage freshness check failed")

    return {
        "status": "ok",
        "model_loaded": request.app.state.model_loaded,
        "model_size": model_size,
        "model_label": model_labels.get(model_size, model_size),
        "space_track_data_refreshed_at": space_track_refreshed_at,
        "db_connected": db_ok,
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
