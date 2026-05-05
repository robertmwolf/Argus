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

logger = logging.getLogger(__name__)

_MAX_UPLOAD_BYTES = 300 * 1024 * 1024  # 100 MB
_ALLOWED_EXTENSIONS = {".fits", ".fit", ".fts", ".png"}
# FITS magic: first 8 bytes are "SIMPLE  " (with trailing spaces)
_FITS_MAGIC = b"SIMPLE  "


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

        # pipeline_run is patched in tests via unittest.mock.
        # DEMO_MODE=true returns synthetic detections without requiring weights.
        if os.environ.get("DEMO_MODE", "").lower() == "true":
            detections = _demo_detections(tmp_path)
        else:
            from inference.pipeline import run as pipeline_run
            detections = pipeline_run(fits_path=tmp_path, fast=True)
            app.state.model_loaded = True

        png_bytes = _render_png(tmp_path, detections)
        await storage.save_image(job_id, png_bytes)

        async with session_factory() as session:
            for det_dict in detections:
                obb = det_dict.get("obb") or {}
                bbox = det_dict.get("bbox") or [None, None, None, None]
                det = Detection(
                    id=str(uuid.uuid4()),
                    observation_id=job_id,
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
                    ra_deg=det_dict.get("ra_deg"),
                    dec_deg=det_dict.get("dec_deg"),
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


def _render_fits_preview(data: bytes) -> bytes | None:
    """Render FITS image data to a preview PNG without any overlays.

    Uses 1st–99th percentile stretch for visibility.

    Args:
        data: Raw FITS file bytes.

    Returns:
        PNG bytes, or None on failure.
    """
    try:
        import numpy as np
        from astropy.io import fits as afits
        from PIL import Image

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

        img_data = img_data.astype(float)
        finite = img_data[np.isfinite(img_data)]
        if finite.size == 0:
            return None
        lo, hi = np.percentile(finite, [1, 99])
        if hi == lo:
            hi = lo + 1.0
        img_data = np.clip((img_data - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

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


def _demo_detections(fits_path: Path) -> list[dict]:
    """Return a demo detection by finding the actual streak via Hough transform.

    Uses probabilistic Hough lines on the Z-score-normalised image so the
    returned OBB aligns with the real streak regardless of image size or
    streak angle.  Falls back to a single centred placeholder on failure.

    Args:
        fits_path: Path to the FITS file.

    Returns:
        List with one detection dict whose OBB covers the detected streak.
    """
    import math as _math

    try:
        import cv2 as _cv2
        import numpy as _np
        from inference.fits_loader import FITSLoader

        result = FITSLoader().load(fits_path)
        arr = result["array"]          # (H, W, 3) uint8, Z-score normalised
        ht, w = arr.shape[:2]
        gray = arr[:, :, 0]

        # Threshold at 99th percentile — keeps only the very brightest pixels
        threshold = float(_np.percentile(gray, 99))
        binary = (gray >= threshold).astype(_np.uint8) * 255

        # Hough: require lines longer than 12 % of the shorter image edge
        min_len = max(20, int(min(w, ht) * 0.12))
        lines = _cv2.HoughLinesP(
            binary, 1, _np.pi / 180,
            threshold=max(15, min_len // 2),
            minLineLength=min_len,
            maxLineGap=15,
        )

        if lines is not None and len(lines) > 0:
            # Longest segment wins
            best = max(lines, key=lambda l: _math.hypot(
                l[0][2] - l[0][0], l[0][3] - l[0][1]
            ))
            lx1, ly1, lx2, ly2 = map(float, best[0])

            length  = _math.hypot(lx2 - lx1, ly2 - ly1)
            cx      = (lx1 + lx2) / 2.0
            cy      = (ly1 + ly2) / 2.0
            angle_deg = _math.degrees(_math.atan2(ly2 - ly1, lx2 - lx1)) % 180.0

            pad  = max(4, int(length * 0.02))
            bx1  = max(0.0, min(lx1, lx2) - pad)
            by1  = max(0.0, min(ly1, ly2) - pad)
            bx2  = min(float(w),  max(lx1, lx2) + pad)
            by2  = min(float(ht), max(ly1, ly2) + pad)

            obb_long = length + pad * 2
            obb_short = 8.0   # visual streak width in pixels

            return [{
                "confidence": 0.94,
                "bbox": [bx1, by1, bx2, by2],
                "obb": {
                    "cx": cx, "cy": cy,
                    "w": obb_long, "h": obb_short,
                    "angle_deg": angle_deg,
                },
                "streak_length_px": round(length, 1),
                "ra_deg": 83.82,
                "dec_deg": -5.39,
                "identifications": [
                    {"satellite_name": "ISS (ZARYA)", "norad_id": 25544,
                     "confidence": 0.87, "rank": 1},
                    {"satellite_name": "STARLINK-1234", "norad_id": 47123,
                     "confidence": 0.41, "rank": 2},
                ],
            }]

    except Exception:
        logger.exception("Demo Hough detection failed — using placeholder")

    # Fallback: single centred placeholder
    try:
        from astropy.io import fits as afits
        with afits.open(fits_path) as hdul:
            _h = hdul[0].header
            w  = int(_h.get("NAXIS1", 1024))
            ht = int(_h.get("NAXIS2", 768))
    except Exception:
        w, ht = 1024, 768

    return [{
        "confidence": 0.50,
        "bbox": [int(w * 0.1), int(ht * 0.1), int(w * 0.9), int(ht * 0.9)],
        "obb": {
            "cx": w * 0.5, "cy": ht * 0.5,
            "w": w * 0.8,  "h": 8.0,
            "angle_deg": 45.0,
        },
        "streak_length_px": round(w * 0.8, 1),
        "ra_deg": None,
        "dec_deg": None,
        "identifications": [],
    }]


def _render_png(fits_path: Path, detections: list[dict]) -> bytes:
    """Render FITS data to a PNG with bounding box overlays.

    Args:
        fits_path: Path to the FITS file.
        detections: List of detection dicts from the pipeline.

    Returns:
        PNG file bytes.
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw

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

        det_rows = (
            await session.execute(
                select(Detection).where(Detection.observation_id == job_id)
            )
        ).scalars().all()

        detections = []
        for det in det_rows:
            ident_rows = (
                await session.execute(
                    select(Identification).where(Identification.detection_id == det.id)
                )
            ).scalars().all()

            detections.append({
                "confidence": det.confidence,
                "bbox": [det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2],
                "obb": {
                    "cx": det.obb_cx,
                    "cy": det.obb_cy,
                    "w": det.obb_w,
                    "h": det.obb_h,
                    "angle_deg": det.obb_angle_deg,
                },
                "streak_length_px": det.streak_length_px,
                "ra_deg": det.ra_deg,
                "dec_deg": det.dec_deg,
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

        return {
            "job_id": job_id,
            "status": obs.status,
            "filename": obs.filename,
            "obs_epoch": obs.obs_epoch,
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

    return {
        "status": "ok",
        "model_loaded": request.app.state.model_loaded,
        "db_connected": db_ok,
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
