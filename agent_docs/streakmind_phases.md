# StreakMind Phases 2–8

Detailed specifications for each phase after the data pipeline.
Each phase has a gate condition — do not start the next phase until
the current one passes its gate.

> **Path note:** All paths are relative to the repo root `Argus/`.
> The old `streakmind/` prefix no longer exists — directories were
> flattened to top-level `inference/`, `training/`, `models/`, etc.

---

## Phase 2 — DINO Model  ✅ COMPLETE

### What was built

| File | Description |
|------|-------------|
| `inference/device.py` | `get_device()` (CUDA→MPS→CPU), `get_device_config()`, `safe_autocast()` |
| `models/dino/streak_codino_swin_t.py` | DINO Swin-T MMDet config — Mac dev (batch=1, workers=0, 400px, 300 queries) |
| `models/dino/streak_codino_swin_l.py` | DINO Swin-L MMDet config — A100 cloud (batch=2, workers=4, 800px, 900 queries) |
| `training/train_dino.py` | Two-stage training, Stage2UnfreezeHook (epoch 21), CostGuardrailHook, --smoke-test |
| `scripts/download_weights.py` | Downloads Swin-T (~160 MB) or Swin-L (~828 MB) DINO COCO pretrain weights |
| `scripts/make_test_fits.py` | Synthetic FITS generator (Poisson noise + stars + streak injection) |

### Implementation notes

- **Co-DINO vs DINO**: Co-DINO (Co-Deformable DETR) is not in the mmdet 3.3.0
  pip release. Configs use the standard `DINO` detector class, which is the
  transformer core of Co-DINO. On the cloud machine, mmdet can be installed
  from GitHub source to enable full Co-DINO (adds auxiliary RPN+ROI heads).
  Performance difference for single-class detection is minor.
- **Two-stage schedule**: backbone `lr_mult=0.0` (frozen) for epochs 1–20;
  `Stage2UnfreezeHook` sets backbone `lr_mult=0.1` at epoch 21.
- **Cost guardrail**: after epoch 1 prints estimated total time + Lambda cost
  ($1.29/hr), then `sleep(30)` before epoch 2.
- **Swin-T weights**: using DINO R50 4-scale COCO checkpoint as placeholder;
  replace with actual Swin-T checkpoint once identified/available.

### Gate ✅ cleared
- Both configs parse with mmengine: `DINO` model type, 1 class `streak`,
  Z-score mean/std, gradient checkpointing enabled
- `get_device()` returns `mps` on Mac; `get_device_config()` returns
  correct MPS-safe values
- `make_test_fits.py` generates valid FITS with WCS headers
- 206 tests passing (at time of Phase 2 completion)

---

## Phase 3 — Inference Pipeline  ← CURRENT

### Gate condition
Phase 2 configs parse cleanly and `device.py` works on MPS. ✅

### Files to build

#### `inference/pipeline.py`

Main inference orchestrator.  Accepts a FITS path, returns detections.

```python
def run(
    fits_path: str | Path,
    fast: bool = False,
) -> list[dict]:
    """Run the full inference pipeline on a single FITS image.

    Args:
        fits_path: Path to the input FITS file.
        fast: If True, skip Radon refinement, crossid, and DB write.
              Uses image_size=256. Target: <60 s on Mac.

    Returns:
        List of detection dicts with keys:
          confidence, bbox [x1,y1,x2,y2], obb {cx,cy,w,h,angle_deg},
          streak_length_px, ra_deg, dec_deg,
          identifications [{satellite_name, norad_id, confidence, rank}]

    Timing: log fits_load_ms, inference_ms, postprocess_ms,
            crossid_ms, db_write_ms at DEBUG level.
    """
```

Env vars controlling behaviour:
- `FAST_MODE=true` → same as `fast=True`
- `MODEL_SIZE=tiny|large` → selects config
- `MODEL_WEIGHTS=path/to/weights.pth` → checkpoint override

#### `inference/postprocess.py`

Radon-based angle refinement and NMS.

```python
def refine_angle(
    image_crop: np.ndarray,
    obb: dict,
    angle_search_range: float = 15.0,
) -> float:
    """Refine OBB angle using the Radon transform on the streak crop.

    Source: StreakMind — Radon angle refinement
    Ref: agent_docs/streakmind_phases.md

    Args:
        image_crop: Greyscale uint8 crop centred on the streak.
        obb: Detection OBB dict {cx, cy, w, h, angle_deg}.
        angle_search_range: ±degrees around DINO's predicted angle to search.

    Returns:
        Refined angle in degrees (replaces obb['angle_deg']).
    """

def nms_detections(
    detections: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Non-maximum suppression on OBB detections using Shapely polygon IoU."""
```

#### `inference/crossid.py`

Satellite ephemeris cross-matching against Space-Track GP_History TLEs.
Delegates TLE fetching to `src.matching.spacetrack_query.query_gp_history`,
which handles authentication, rate limiting, and disk caching.

Requires `SPACETRACK_USER` and `SPACETRACK_PASS` environment variables.

```python
# Source: Danarianto et al. — Gaussian confidence scoring approach
# Ref: cite per published paper

def cross_identify(
    detections: list[dict],
    obs_time: datetime,
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
    epoch_window_days: int = 3,
) -> list[dict]:
    """Cross-match detections against Space-Track GP_History TLEs.

    Fetches TLEs for the epoch window ending at obs_time from Space-Track
    (cached to disk), propagates each to obs_time via SGP4, and scores
    candidates using Gaussian position confidence.
    """
```

TLE data source: Space-Track GP_History API (authoritative).
Caching: disk cache in `data/cache/` via `diskcache` (TTL 48 h for historical,
2 h for recent observations).  No local TLE file is maintained.

### Tests to write

`tests/test_pipeline.py`:
- `run()` on a synthetic FITS completes without error in fast mode
- Returns list of dicts with required keys
- `FAST_MODE=true` skips Radon refinement (mock postprocess)

`tests/test_postprocess.py`:
- `refine_angle` on a synthetic streak returns angle within ±5° of ground truth
- `nms_detections` removes overlapping boxes, keeps highest confidence

`tests/test_crossid.py`:
- `cross_identify` with a known TLE returns top-3 candidates
- Candidate with lowest angular separation has highest confidence
- Missing sky coords → identifications empty list, no crash
- `_fetch_tle_catalog` maps GP_History JSON fields to (name, line1, line2) tuples

### Gate condition for Phase 4
`inference/pipeline.py --fast --image data/sample/synth_streak_000.fits`
completes in <60 seconds and returns at least one detection.

---

## Phase 4 — Database

### Gate condition
Phase 3 pipeline returns detections in fast mode.

### File: `db/schema.sql`

```sql
-- Compatible with PostgreSQL 16 and SQLite (via aiosqlite).

CREATE TABLE observations (
    id            TEXT PRIMARY KEY,   -- UUID as string for SQLite compat
    filename      TEXT NOT NULL,
    uploaded_at   TEXT DEFAULT (datetime('now')),
    exposure_time REAL,
    obs_epoch     TEXT,               -- ISO8601
    fits_wcs_json TEXT,               -- JSON-serialized WCS params
    status        TEXT DEFAULT 'queued'
    -- status: queued / processing / complete / failed
);

CREATE TABLE detections (
    id               TEXT PRIMARY KEY,
    observation_id   TEXT REFERENCES observations(id),
    confidence       REAL NOT NULL,
    bbox_x1          REAL, bbox_y1 REAL,
    bbox_x2          REAL, bbox_y2 REAL,
    obb_cx           REAL, obb_cy  REAL,
    obb_w            REAL, obb_h   REAL,
    obb_angle_deg    REAL,
    streak_length_px REAL,
    ra_deg           REAL,
    dec_deg          REAL
);

CREATE TABLE identifications (
    id             TEXT PRIMARY KEY,
    detection_id   TEXT REFERENCES detections(id),
    norad_id       INTEGER,
    satellite_name TEXT,
    confidence     REAL,
    separation_deg REAL,
    rank           INTEGER    -- 1 = best match, up to 3
);

CREATE TABLE tracklets (
    id         TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE tracklet_detections (
    tracklet_id  TEXT REFERENCES tracklets(id),
    detection_id TEXT REFERENCES detections(id),
    frame_index  INTEGER,
    PRIMARY KEY (tracklet_id, detection_id)
);
```

SQLAlchemy async setup:
- PostgreSQL: `asyncpg` driver
- SQLite (default): `aiosqlite` driver
- `DATABASE_URL` env var; default: `sqlite+aiosqlite:///./argus.db`

### Tests: `tests/test_db.py`
- [ ] Schema creates without error on SQLite
- [ ] Observation record inserts and queries by id
- [ ] Detection record references observation correctly
- [ ] Identification references detection correctly
- [ ] Status transition queued → processing → complete works

---

## Phase 5 — API

### Gate condition
Database CRUD tests pass.

### Endpoints: `api/main.py`

```
POST /api/upload
  Accept multipart FITS/PNG (max 100 MB)
  Validate extension + magic bytes
  → {job_id, status: "queued"}

GET /api/result/{job_id}
  → {job_id, status, filename, obs_epoch, detections: [{...}]}

GET /api/image/{job_id}
  → processed PNG as image/png

GET /health
  → {status: "ok", model_loaded: bool, db_connected: bool}
```

### `api/storage.py`
Abstract `StorageBackend` with `LocalStorage` and `S3Storage`.
Selected by `STORAGE_BACKEND=local|s3`.

### `api/queue.py`
Abstract `JobQueue` with `InMemoryQueue` and `SQSQueue`.
Selected by `QUEUE_BACKEND=memory|sqs`.

**Constraint**: `api/main.py` and `inference/pipeline.py` never import
concrete storage/queue classes — only via factory function from env vars.

### Tests: `tests/test_api.py`
- [ ] Upload valid FITS → 200 + job_id
- [ ] Upload oversized file → 413
- [ ] Upload invalid extension → 422
- [ ] Result for unknown id → 404
- [ ] Health endpoint → 200 with expected keys
- [ ] Full upload→poll→result cycle with synthetic FITS (integration)

---

## Phase 6 — Frontend

### Stack
React 18 + Vite + Tailwind CSS.

### Components

`frontend/src/components/UploadZone.jsx`
- Drag-and-drop, accepts `.fits .fit .fts .png`
- POSTs to `/api/upload`, polls `/api/result/{job_id}` every 2 s
- Status: Queued → Processing → Complete

`frontend/src/components/ResultViewer.jsx`
- HTML `<canvas>` renders image + rotated OBBs
- OBB colour: `#00DCFF` (cyan), opacity = confidence
- Hover tooltip: confidence, length, RA/Dec, best ID match

`frontend/src/components/DetectionTable.jsx`
- Columns: `#`, Confidence, Length (px), RA, Dec, Best ID, ID Confidence
- Row click highlights that OBB in canvas (`#FF6B35`, 4px stroke)

### Gate condition for Phase 7
Upload FITS in browser, see annotated image with OBBs and table.

---

## Phase 7 — Docker

### Dockerfiles: `docker/`

`Dockerfile.api` — python:3.11-slim, requirements-api.txt, uvicorn
`Dockerfile.worker` — pytorch/pytorch:2.2.0-cuda12.1, full requirements.txt
`Dockerfile.frontend` — node:20-alpine build → nginx:alpine serve

### `docker-compose.yml` (repo root)

Services: `db` (postgres:16), `api` (port 8000), `worker` (GPU), `frontend` (port 80)
Override `docker-compose.cloud.yml` for S3 + SQS deployment.

### Gate condition for Phase 8
`docker compose up` → working service at `http://localhost`.

---

## Phase 8 — Evaluation

### `eval/metrics.py`

```python
def evaluate(predictions, ground_truth) -> dict:
    """Returns: precision, recall, F1 @ IoU 0.5; mAP@0.5, mAP@0.75;
    mean_angle_error_deg; per_band (short/medium/long streaks)."""
```

### `eval/benchmark.py`
Head-to-head: DINO vs YOLO11-OBB baseline on same test split.
Output markdown table + save per-image results to `eval/results/`.

Target metrics (from StreakMind paper):
- DINO: ≥94% precision, ≥97% recall
- YOLO baseline: reference comparison

### Results template: `results/phase8_benchmark.json`
```json
{
  "date_recorded": "",
  "model": "dino_swin_l",
  "map_50": 0.0, "map_75": 0.0,
  "recall": 0.0, "precision": 0.0, "f1": 0.0,
  "mean_angle_error_deg": 0.0,
  "per_band": {
    "short":  {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "medium": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "long":   {"precision": 0.0, "recall": 0.0, "f1": 0.0}
  },
  "yolo_baseline": {"map_50": 0.0, "recall": 0.0, "precision": 0.0}
}
```
