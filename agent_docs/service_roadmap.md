# Service Deployment Roadmap

## Goal
A self-hostable, docker-compose deployable service on a single GPU machine,
cloud-deployable without code changes — only config/env changes.

The storage and queue backends are swappable via env vars with zero changes
to `api/main.py` or `inference/pipeline.py`.

---

## Architecture: Three Containers

| Container  | Image base                            | Port | Role |
|------------|---------------------------------------|------|------|
| `db`       | postgres:16-alpine                    | 5432 | PostgreSQL (or SQLite for local dev) |
| `api`      | python:3.11-slim                      | 8000 | FastAPI: upload, result, health endpoints |
| `worker`   | pytorch/pytorch:2.2.0-cuda12.1-...   | —    | GPU inference: FITS→Co-DINO→cross-ID→DB |
| `frontend` | nginx:alpine (multi-stage build)      | 80   | React/Vite static + OBB canvas rendering |

For local development without Docker, use SQLite (`DATABASE_URL=sqlite+aiosqlite:///./argus.db`)
and run api + worker in separate terminals.

---

## Phase S1 — Standalone CLI Pipeline (prerequisite)

Complete ARGUS Phases 1–3 (data pipeline + model + cross-ID).
The pipeline must accept a FITS file path and return detections + identifications
as plain Python objects with no web layer.

**Done when:**
```bash
python -m inference.pipeline path/to/image.fits
# prints ranked detections with satellite identifications
```

---

## Phase S2 — FastAPI Service (local, no Docker)

Builds `api/`:
- `main.py` — FastAPI routes
- `models.py` — Pydantic request/response schemas
- `storage.py` — `LocalStorage` / `S3Storage` behind abstract interface
- `queue.py` — `InMemoryQueue` / `SQSQueue` behind abstract interface

Run locally:
```bash
uvicorn api.main:app --reload --port 8000
```

Storage: `./uploads/` directory.
Queue: `asyncio.Queue`, worker runs as a background task in the same process.

**Done when:** `curl -F file=@image.fits http://localhost:8000/api/upload`
returns a job_id and polling `/api/result/{job_id}` eventually shows results.

---

## Phase S3 — Frontend (React + Vite)

Builds `frontend/`:
- `UploadZone.jsx` — drag-drop upload, status polling
- `ResultViewer.jsx` — canvas OBB rendering
- `DetectionTable.jsx` — detection list, row-click highlights OBB

Dev server:
```bash
cd frontend && npm run dev
```

**Done when:** Upload FITS in browser, see annotated image with OBBs and
ranked candidate table.

---

## Phase S4 — Docker + docker-compose

### `docker-compose.yml` (repo root)

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: argus
      POSTGRES_USER: argus
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/schema.sql:/docker-entrypoint-initdb.d/schema.sql

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    environment:
      DATABASE_URL: postgresql+asyncpg://argus:${DB_PASSWORD}@db/argus
      STORAGE_BACKEND: local
      QUEUE_BACKEND: memory
    volumes:
      - ./uploads:/app/uploads
    ports:
      - "8000:8000"
    depends_on: [db]

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      DATABASE_URL: postgresql+asyncpg://argus:${DB_PASSWORD}@db/argus
      STORAGE_BACKEND: local
      QUEUE_BACKEND: memory
      MODEL_WEIGHTS: /app/weights/argus_dino.pth
      SPACETRACK_USER: ${SPACETRACK_USER}
      SPACETRACK_PASS: ${SPACETRACK_PASS}
    volumes:
      - ./uploads:/app/uploads
      - ./weights:/app/weights
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on: [db, api]

  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "80:80"
    depends_on: [api]

volumes:
  postgres_data:
```

**Done when:** `docker compose up` → service at `http://localhost`.

---

## Phase S5 — Self-Hosted Public Access (optional)

Use Cloudflare Tunnel to expose the service without opening firewall ports.

```bash
cloudflared tunnel create argus
cloudflared tunnel route dns argus argus.yourdomain.com
cloudflared tunnel run argus
```

Add `cloudflared` as a fifth container in `docker-compose.yml`.

---

## Phase S6 — Cloud Scale Path

Swap backends via env vars — no code changes:

| Component     | Local (S2–S5)          | Cloud                         |
|---------------|------------------------|-------------------------------|
| Job queue     | `asyncio.Queue`        | AWS SQS (`QUEUE_BACKEND=sqs`) |
| File storage  | `./uploads/`           | AWS S3 (`STORAGE_BACKEND=s3`) |
| Database      | SQLite (dev) / Postgres| Cloud Postgres (RDS, etc.)    |
| Workers       | Background task        | Separate GPU containers       |
| Deployment    | `docker compose`       | ECS Fargate / Cloud Run       |

### `docker-compose.cloud.yml` (override file)

```yaml
services:
  api:
    environment:
      STORAGE_BACKEND: s3
      QUEUE_BACKEND: sqs
      S3_BUCKET: ${S3_BUCKET}
      AWS_REGION: ${AWS_REGION}
  worker:
    environment:
      STORAGE_BACKEND: s3
      QUEUE_BACKEND: sqs
```

Run cloud deployment:
```bash
docker compose -f docker-compose.yml -f docker-compose.cloud.yml up
```

---

## What Does NOT Change Between Deployments

The inference pipeline (`inference/`) is never modified for
deployment. The API layer calls the same functions the CLI does. Storage
and queue are injected via factory functions, not imported directly.
