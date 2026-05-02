# Service Deployment Roadmap

## Goal
Wrap the ARGUS inference pipeline in a deployable service — either self-hosted
or cloud-hosted — without changing the core pipeline code.

## Phases

### Phase S1 — Standalone Inference Pipeline (prerequisite)
Complete Phases 1–3 (classical baseline through hybrid consensus).
The pipeline must accept a FITS file path and return `CandidateMatch` results
as plain Python with no web layer.

**Done when:** `python src/pipeline.py path/to/image.fits` prints ranked matches.

---

### Phase S2 — FastAPI Service (local)

**New files:**
```
src/
├── api/
│   ├── main.py          ← FastAPI app, routes
│   ├── models.py        ← Pydantic request/response schemas
│   └── jobs.py          ← In-memory job tracker (dict[str, JobState])
```

**API surface:**
```
POST /jobs               → accept FITS upload, enqueue job, return job_id
GET  /jobs/{id}          → poll status: pending | running | done | failed
GET  /jobs/{id}/result   → return CandidateMatch list as JSON
GET  /jobs/{id}/image    → return annotated PNG with OBB overlays
```

**Storage (local, Phase S2):**
- Uploaded FITS files → `data/uploads/{job_id}.fits`
- Result JSON → `data/results/{job_id}.json`
- Annotated PNG → `data/results/{job_id}.png`
- Job state → in-memory `dict` (lost on restart; acceptable for local dev)

**Background execution:**
Use `asyncio.create_task()` or `concurrent.futures.ThreadPoolExecutor` to run
the blocking pipeline off the event loop. Keep it simple — no Celery, no Redis
in this phase.

**Done when:** `curl -F file=@image.fits http://localhost:8000/jobs` returns a
job_id and polling shows results.

---

### Phase S3 — Frontend (canvas OBB rendering)

**Stack:** Single-page app. Vanilla JS or lightweight framework (e.g., Preact).
No build step required for v1 — serve static files from FastAPI's `StaticFiles`.

**UI features:**
- File picker → POST to `/jobs`
- Poll `/jobs/{id}` every 2 s until done
- Fetch `/jobs/{id}/image` and display annotated PNG on `<canvas>`
- Render OBB overlays client-side from `/jobs/{id}/result` JSON
  (so users can toggle individual detections on/off)
- Show ranked candidate table: NORAD ID, name, confidence score, ambiguity flag

**Done when:** Upload FITS in browser, see annotated image with candidate list.

---

### Phase S4 — Docker + docker-compose

**Two containers:**

| Service | Image | Port |
|---------|-------|------|
| `api`   | `argus-api` | 8000 |
| `frontend` | `nginx:alpine` (serves static build) | 80 |

**docker-compose.yml volumes:**
- `./data:/app/data` — persist uploads and results across restarts
- `.env` file for `SPACETRACK_USER` / `SPACETRACK_PASS`

**Done when:** `docker compose up` produces a working service accessible at
`http://localhost`.

---

### Phase S5 — Self-Hosted Public Access (optional)

Use **Cloudflare Tunnel** (`cloudflared`) to expose the local service without
opening firewall ports or managing TLS certificates.

```bash
cloudflared tunnel create argus
cloudflared tunnel route dns argus argus.yourdomain.com
cloudflared tunnel run argus
```

Add `cloudflared` as a fourth container in `docker-compose.yml` so tunnel
starts automatically with the stack.

**Done when:** `https://argus.yourdomain.com` reaches the service from the
public internet.

---

### Phase S6 — Cloud Scale Path (if needed later)

Swap components one at a time — the API code does not change, only the
backing implementations:

| Component | Local (S2–S5) | Cloud |
|-----------|--------------|-------|
| Job queue | `dict` in memory | Redis (self-hosted) or AWS SQS |
| File storage | `data/uploads/` | AWS S3 or Azure Blob |
| Workers | `ThreadPoolExecutor` | Separate worker containers (same image, `CMD=worker`) |
| Deployment | `docker compose` | ECS Fargate / Cloud Run / Fly.io |

**Design constraint:** write `jobs.py` and storage calls behind thin interfaces
from the start so the swap is a config change, not a rewrite.

---

## Dependency Notes

### New packages (Phase S2+)
```bash
pip install fastapi uvicorn[standard] python-multipart aiofiles
```

### Frontend (Phase S3, no npm required for v1)
Serve from `src/api/static/`. Use a CDN-hosted Preact or plain JS.

### Docker (Phase S4)
- Base image: `python:3.11-slim` for the API
- Install conda dependencies via `pip` (export `conda env export --from-history`
  then install via pip in Docker for smaller images)

---

## What Does NOT Change
The core pipeline (`src/ingest/`, `src/detection/`, `src/astrometry/`,
`src/matching/`) is **not modified** for service deployment. The API layer
calls the same functions the CLI does. This is the entire point of keeping
Phase S1 as a prerequisite.
