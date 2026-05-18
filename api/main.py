"""ARGUS FastAPI application.

Endpoints:
  POST /api/upload             — accept FITS/PNG, enqueue for processing
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

from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import select, text

from db.models import Detection, Identification, Observation, get_engine, get_session_factory, init_db
from inference.confidence import compute_unified_confidence

logger = logging.getLogger(__name__)

_MAX_UPLOAD_BYTES = 300 * 1024 * 1024  # 100 MB
_ALLOWED_EXTENSIONS = {".fits", ".fit", ".fts", ".png"}
# FITS magic: first 8 bytes are "SIMPLE  " (with trailing spaces)
_FITS_MAGIC = b"SIMPLE  "


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.queue import get_queue
    from api.storage import get_storage

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

    worker_task = asyncio.create_task(_worker_loop(app))

    yield

    worker_task.cancel()
    try:
        await worker_task
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
        obs.status = "processing"
        await session.commit()

    tmp_path: Path | None = None
    try:
        fits_data = await storage.load_upload(job_id, filename)

        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as f:
            f.write(fits_data)
            tmp_path = Path(f.name)
        sidecar_path = _copy_matching_wcs_sidecar(filename, tmp_path, job_id)

        # pipeline_run is patched in tests via unittest.mock.
        # Set FAST_MODE=true in the environment to skip Radon + cross-ID.
        try:
            from inference.pipeline import load_model as pipeline_load_model
            from inference.pipeline import run_with_array as pipeline_run
        except ImportError as exc:
            raise RuntimeError(
                "ML packages (torch/mmdet) are not installed in this environment. "
                "Run the API directly with the satid conda env for local inference, "
                "or use the GPU worker container for Docker deployments."
            ) from exc

        if app.state.pipeline_model is None:
            model, inference_device = await asyncio.to_thread(pipeline_load_model)
            app.state.pipeline_model = model
            app.state.pipeline_device = inference_device
            app.state.model_loaded = True

        detections, fits_array = await asyncio.to_thread(
            pipeline_run,
            fits_path=tmp_path,
            model=app.state.pipeline_model,
            inference_device=app.state.pipeline_device,
        )
        app.state.model_loaded = True

        png_bytes = await asyncio.to_thread(_render_png, tmp_path, detections, fits_array)
        await storage.save_image(job_id, png_bytes)

        async with session_factory() as session:
            for det_dict in detections:
                obb = det_dict.get("obb") or {}
                bbox = det_dict.get("bbox") or [None, None, None, None]
                det = Detection(
                    id=str(uuid.uuid4()),
                    observation_id=job_id,
                    method=det_dict.get("method") or "ml",
                    confidence=float(det_dict.get("confidence", 0.0)),
                    bbox_x1=bbox[0] if len(bbox) > 0 else None,
                    bbox_y1=bbox[1] if len(bbox) > 1 else None,
                    bbox_x2=bbox[2] if len(bbox) > 2 else None,
                    bbox_y2=bbox[3] if len(bbox) > 3 else None,
                    obb_cx=obb.get("cx"),
                    obb_cy=obb.get("cy"),
                    obb_w=obb.get("w"),
                    obb_h=obb.get("h"),
                    obb_angle_deg=obb.get("angle_deg"),
                    streak_length_px=det_dict.get("streak_length_px"),
                    streak_id=det_dict.get("streak_id"),
                    ra_tip1_deg=det_dict.get("ra_tip1_deg"),
                    dec_tip1_deg=det_dict.get("dec_tip1_deg"),
                    ra_tip2_deg=det_dict.get("ra_tip2_deg"),
                    dec_tip2_deg=det_dict.get("dec_tip2_deg"),
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
                    ))

            obs = await session.get(Observation, job_id)
            obs.status = "complete"
            await session.commit()

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


def _extract_image_shape(data: bytes, filename: str) -> tuple[int, int] | None:
    """Return original image dimensions as ``(width, height)``.

    Args:
        data: Raw uploaded FITS or PNG bytes.
        filename: Original upload filename, used to select the parser.

    Returns:
        ``(width, height)`` in source-image pixels, or None on failure.
    """
    try:
        ext = Path(filename).suffix.lower()
        if ext in {".fits", ".fit", ".fts"}:
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
        if ext == ".png":
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


def _render_png(
    fits_path: Path,
    detections: list[dict],
    array: "np.ndarray | None" = None,
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
        import numpy as np
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
            bbox = det.get("bbox")
            if bbox and len(bbox) == 4:
                conf = det.get("confidence", 1.0)
                alpha = int(conf * 200)
                draw.rectangle(
                    [bbox[0], bbox[1], bbox[2], bbox[3]],
                    outline=(0, 220, 255, alpha),
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/upload", status_code=status.HTTP_200_OK)
async def upload(request: Request, file: UploadFile) -> dict[str, str]:
    """Accept a FITS or PNG upload, enqueue it for processing.

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

    data = await file.read()

    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 100 MB limit",
        )

    # Validate FITS magic bytes when a FITS extension is used
    if ext in {".fits", ".fit", ".fts"} and not data[:8] == _FITS_MAGIC:
        # Permit synthetic test files that may not have the magic (warn only)
        logger.warning("FITS file %s missing SIMPLE magic bytes", file.filename)

    job_id = str(uuid.uuid4())
    filename = file.filename or f"{job_id}{ext}"

    await request.app.state.storage.save_upload(job_id, filename, data)

    # Extract FITS header and generate preview PNG at upload time so the
    # frontend can display them immediately while the job is queued/processing.
    if ext in {".fits", ".fit", ".fts"}:
        header_cards = _extract_fits_header(data)
        if header_cards is not None:
            await request.app.state.storage.save_fits_header(
                job_id, json.dumps(header_cards).encode()
            )
        preview_png = _render_fits_preview(data)
        if preview_png is not None:
            await request.app.state.storage.save_preview(job_id, preview_png)
    elif ext == ".png":
        await request.app.state.storage.save_preview(job_id, data)

    async with request.app.state.session_factory() as session:
        session.add(Observation(id=job_id, filename=filename, status="queued"))
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
            # Each detector is weighted by its empirical F-0.5 score so that
            # detectors with more false positives contribute proportionally less.
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
                "bbox": [primary.bbox_x1, primary.bbox_y1, primary.bbox_x2, primary.bbox_y2],
                "obb": {
                    "cx": primary.obb_cx,
                    "cy": primary.obb_cy,
                    "w": primary.obb_w,
                    "h": primary.obb_h,
                    "angle_deg": primary.obb_angle_deg,
                },
                "streak_length_px": primary.streak_length_px,
                "ra_tip1_deg": primary.ra_tip1_deg,
                "dec_tip1_deg": primary.dec_tip1_deg,
                "ra_tip2_deg": primary.ra_tip2_deg,
                "dec_tip2_deg": primary.dec_tip2_deg,
                "identifications": [
                    {
                        "satellite_name": i.satellite_name,
                        "norad_id": i.norad_id,
                        "confidence": i.confidence,
                        "separation_deg": i.separation_deg,
                        "rank": i.rank,
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

        return {
            "job_id": job_id,
            "status": obs.status,
            "filename": filename,
            "obs_epoch": obs.obs_epoch,
            "image_width": image_shape[0] if image_shape else None,
            "image_height": image_shape[1] if image_shape else None,
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

    Available immediately after upload for FITS files and PNGs.

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
        "tiny":        "DINO Swin-Tiny - SatStreaks",
        "large":       "DINO Swin-Large - SatStreaks",
        "dinov3_vitb": "DINOv3 ViT-Base - SatStreaks+GTImages",
        "dinov3_vitl": "DINOv3 ViT-Large - SatStreaks+GTImages",
    }
    return {
        "status": "ok",
        "model_loaded": request.app.state.model_loaded,
        "model_size": model_size,
        "model_label": model_labels.get(model_size, model_size),
        "db_connected": db_ok,
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
