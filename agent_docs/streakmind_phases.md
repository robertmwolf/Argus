# StreakMind Phases 2–8

Detailed specifications for each phase after the data pipeline.
Each phase has a gate condition — do not start the next phase until
the current one passes its gate.

---

## Phase 2 — Co-DINO Model

### Gate condition
Phase 1 produced a valid COCO JSON and `FITSStreakDataset` iterates
without error.

### Files

#### `streakmind/models/dino/streak_codino_swin_l.py`

MMDetection config. Inherit from:
`mmdet::co_dino/co_dino_5scale_swin_l_16xb1_3x_coco.py`

Override these fields only:

```python
model = dict(
    query_head=dict(num_classes=1),
    roi_head=[dict(bbox_head=dict(num_classes=1))],
    backbone=dict(
        with_cp=True,        # gradient checkpointing
        pretrained=None,
    ),
)

data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],   # our Z-score normalized images
    std=[51.0, 51.0, 51.0],
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(metainfo=dict(classes=('streak',))),
)

val_dataloader = dict(batch_size=1)

optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0),   # Stage 1: frozen backbone
            'neck': dict(lr_mult=0.1),
            'query_head': dict(lr_mult=1.0),
        }
    ),
)

train_cfg = dict(max_epochs=50, val_interval=5)

load_from = 'weights/co_dino_5scale_swin_l_16xb1_3x_coco.pth'
```

Also create `streak_codino_swin_t.py` — identical but with Swin-T backbone
for machines with < 12 GB VRAM.

#### `streakmind/training/train_dino.py`
- Launch MMDetection training with the config above
- Accept `--config` and `--work-dir` CLI args
- Checkpoint to `weights/` directory

#### `streakmind/training/train_baseline.py`
- Train YOLO11-OBB using Ultralytics API on the same dataset
- Used only for evaluation comparison in Phase 8

### Training strategy (two stages)
- **Stage 1 (epochs 1–20):** Backbone frozen (`lr_mult=0.0`). Only neck + query head train.
- **Stage 2 (epochs 21–50):** Unfreeze backbone with `lr_mult=0.1`.

Update the config override for Stage 2 by changing backbone `lr_mult=0.1`.

### Gate condition for Phase 3
Training converges (val loss decreasing). Inference on 5 sample images
produces bounding boxes. Show sample predictions before proceeding.

---

## Phase 3 — Satellite Cross-Identification

### Gate condition
Phase 2 model produces bounding boxes on validation images.

### File: `streakmind/inference/crossid.py`

```
# Source: Danarianto et al. — Gaussian confidence scoring approach
# Ref: cite per published paper
```

#### Class: `SatelliteCrossIdentifier`

```python
def __init__(
    self,
    catalog_path: str | None = None,
    use_spacetrack: bool = False,
) -> None:
    """Initialize with a TLE catalog file or Space-Track credentials.

    Catalog auto-refresh: if catalog file is older than 24 hours,
    re-download from Celestrak active satellite catalog.
    Default catalog path: streakmind/data/catalogs/active_sats.tle
    Falls back to cached file if download fails (logs warning).
    """

def propagate_to_epoch(
    self,
    satellite: EarthSatellite,
    epoch_utc: datetime,
) -> tuple[float, float]:
    """Propagate satellite to epoch_utc via SGP4.

    Returns (ra_deg, dec_deg) in ICRS frame.
    """

def cross_identify(
    self,
    detections: list[dict],
    observation_epoch: datetime,
    wcs: WCS,
    search_radius_deg: float = 0.5,
) -> list[dict]:
    """Cross-match detections against satellite catalog.

    For each detection:
      1. Convert OBB midpoint (cx, cy) to RA/Dec via wcs
      2. Propagate all catalog TLEs to observation_epoch
      3. Compute angular separation for each satellite
      4. Gaussian score: sigma = search_radius_deg / 3
         score = exp(-separation² / (2σ²))
      5. Attach top-3 candidates sorted by score

    Returns detections with added 'identifications' key.
    """

def score_across_frames(
    self,
    detections_by_frame: list[list[dict]],
) -> list[dict]:
    """Multiply per-frame Gaussian scores for tracklet stabilization.

    Source: Danarianto et al. — multi-frame score multiplication
    """
```

Catalog source: `https://celestrak.org/SOCRATES/query.php` active satellites.
Store at `streakmind/data/catalogs/active_sats.tle`.
Refresh if file is older than 24 hours.

### Test file: `tests/streakmind/test_crossid.py`
- [ ] `propagate_to_epoch` returns (ra, dec) in valid degree ranges
- [ ] `cross_identify` attaches `identifications` list to each detection
- [ ] Each identification has `satellite_name`, `norad_id`, `confidence`, `rank`
- [ ] `rank=1` has highest confidence
- [ ] Stale catalog file triggers re-download (mock the download in tests)
- [ ] Missing WCS → sky coords None → identifications empty list, no crash
- [ ] TLE catalog unavailable → falls back to cached file, logs warning

### Gate condition for Phase 4
`cross_identify()` returns ranked candidates for a known Starlink pass.
Correct NORAD ID appears in top-3.

---

## Phase 4 — Database

### File: `streakmind/db/schema.sql`

```sql
-- Compatible with PostgreSQL 16 and SQLite (via aiosqlite).
-- UUID generation: PostgreSQL uses gen_random_uuid(),
-- SQLite uses a trigger or application-layer UUID.

CREATE TABLE observations (
    id            TEXT PRIMARY KEY,   -- UUID as string for SQLite compat
    filename      TEXT NOT NULL,
    uploaded_at   TEXT DEFAULT (datetime('now')),
    exposure_time REAL,
    obs_epoch     TEXT,               -- ISO8601
    fits_wcs_json TEXT,               -- JSON-serialized WCS params
    status        TEXT DEFAULT 'queued'
    -- status values: queued / processing / complete / failed
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

Use SQLAlchemy with async support:
- PostgreSQL: `asyncpg` driver
- SQLite (default): `aiosqlite` driver
- Connection string from env var `DATABASE_URL`
- Default: `sqlite+aiosqlite:///./streakmind.db`

### Test file: `tests/streakmind/test_db.py`
- [ ] Schema creates without error on SQLite
- [ ] Observation record can be inserted and queried by id
- [ ] Detection record references observation correctly
- [ ] Identification references detection correctly
- [ ] Status update (queued → processing → complete) works

---

## Phase 5 — API

### Gate condition
Database schema is created and CRUD operations pass tests.

### File: `streakmind/api/main.py`

```
POST /api/upload
  - Accept multipart file (FITS or PNG, max 100 MB)
  - Validate: file extension must be .fits, .fit, .fts, or .png
  - Validate magic bytes (FITS: starts with "SIMPLE  =")
  - Save to storage backend
  - Create observation record with status='queued'
  - Enqueue job
  - Return: {"job_id": "<uuid>", "status": "queued"}

GET /api/result/{job_id}
  - Return observation status + detections + identifications if complete
  - Response shape:
    {
      "job_id": str,
      "status": str,
      "filename": str,
      "obs_epoch": str | null,
      "detections": [{
        "id": str,
        "confidence": float,
        "bbox": [x1, y1, x2, y2],
        "obb": {"cx": float, "cy": float, "w": float, "h": float, "angle_deg": float},
        "streak_length_px": float,
        "ra_deg": float | null,
        "dec_deg": float | null,
        "identifications": [{
          "satellite_name": str,
          "norad_id": int,
          "confidence": float,
          "rank": int
        }]
      }]
    }

GET /api/image/{job_id}
  - Return processed PNG as image/png
  - Used by frontend canvas renderer

GET /health
  - Return {"status": "ok", "model_loaded": bool, "db_connected": bool}
```

### File: `streakmind/api/storage.py`

Abstract base class `StorageBackend` with two implementations:
- `LocalStorage` — reads/writes `./uploads/` directory
- `S3Storage` — reads/writes S3 bucket via boto3

Selection: `STORAGE_BACKEND=local` (default) or `s3`
S3 config from: `S3_BUCKET`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

### File: `streakmind/api/queue.py`

Abstract base class `JobQueue` with two implementations:
- `InMemoryQueue` — `asyncio.Queue`, runs worker in background task
- `SQSQueue` — boto3 SQS, polls in background task

Selection: `QUEUE_BACKEND=memory` (default) or `sqs`

Worker coroutine steps:
1. Dequeue job_id
2. Update DB status → 'processing'
3. Load image via storage backend
4. Run inference: `fits_loader → model → postprocess → crossid`
5. Write detections + identifications to DB
6. Update DB status → 'complete' (or 'failed' with error message)

**Design constraint:** `api/main.py` and `inference/pipeline.py` must
never import concrete storage or queue classes directly. Only a factory
function instantiates them based on env vars.

### Test file: `tests/streakmind/test_api.py`
- [ ] `POST /api/upload` with valid FITS returns 200 + job_id
- [ ] `POST /api/upload` with oversized file returns 413
- [ ] `POST /api/upload` with invalid extension returns 422
- [ ] `GET /api/result/{job_id}` for unknown id returns 404
- [ ] `GET /health` returns 200 with expected keys
- [ ] Full upload→poll→result cycle with a sample FITS (integration test)

---

## Phase 6 — Frontend

### Stack
React 18 + Vite + Tailwind CSS. No UI component library.

### File: `streakmind/frontend/src/components/UploadZone.jsx`

- Drag-and-drop zone accepting `.fits`, `.fit`, `.fts`, `.png`
- Display filename and size after selection
- Upload button POSTs to `POST /api/upload`
- On success, poll `GET /api/result/{job_id}` every 2 seconds
- Show animated status: Queued → Processing → Complete → (results appear)

### File: `streakmind/frontend/src/components/ResultViewer.jsx`

Receives `imageUrl: string` and `detections: array`.

Render on HTML `<canvas>`:
1. Draw image
2. For each detection, draw rotated bounding box:
   - Translate to (`obb.cx`, `obb.cy`)
   - Rotate canvas by `obb.angle_deg`
   - Draw rectangle `obb.w × obb.h`
   - Color: `#00DCFF` (cyan), opacity = `detection.confidence`
   - Line width: 2px
3. On hover over a box, show tooltip:
   ```
   Confidence: X%
   Length: Xpx
   RA: X.XX°  Dec: X.XX°
   Best match: SATELLITE NAME (X% conf)
   ```

### File: `streakmind/frontend/src/components/DetectionTable.jsx`

Table columns: `#`, `Confidence`, `Length (px)`, `RA`, `Dec`, `Best ID`, `ID Confidence`

Clicking a row highlights that OBB on canvas (stroke width 4px, color `#FF6B35`).

### Gate condition for Phase 7
Upload a FITS in browser, see annotated image with OBBs and detection table.

---

## Phase 7 — Docker & Deployment

### `streakmind/docker/Dockerfile.api`
```dockerfile
FROM python:3.11-slim
# Install requirements.txt (no torch or mmdet — those go in worker)
CMD ["uvicorn", "streakmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `streakmind/docker/Dockerfile.worker`
```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# Install requirements.txt (full, including mmdet, ultralytics)
# If WEIGHTS_URL env var set, download weights at build time
CMD ["python", "-m", "streakmind.inference.worker"]
```

### `streakmind/docker/Dockerfile.frontend`
Multi-stage:
- Stage 1: `node:20-alpine` → `npm run build`
- Stage 2: `nginx:alpine` → copy `dist/`, serve on port 80

### `docker-compose.yml` (repo root)

Services:
- `db` — postgres:16-alpine, init with `db/schema.sql`
- `api` — Dockerfile.api, port 8000, `STORAGE_BACKEND=local`, `QUEUE_BACKEND=memory`
- `worker` — Dockerfile.worker, GPU reservation (nvidia count=1), mounts `./uploads` and `./weights`
- `frontend` — Dockerfile.frontend, port 80

Also create `docker-compose.cloud.yml` as an override file for cloud deployment
(S3 storage, SQS queue, no local volume mounts).

### `.env.example`
```
DB_PASSWORD=changeme
STORAGE_BACKEND=local
QUEUE_BACKEND=memory
DATABASE_URL=sqlite+aiosqlite:///./streakmind.db
SPACETRACK_USER=
SPACETRACK_PASS=
S3_BUCKET=
AWS_REGION=
WEIGHTS_URL=
```

### Gate condition for Phase 8
`docker compose up` produces a working service at `http://localhost`.
Upload a FITS from a browser and receive annotated results.

---

## Phase 8 — Evaluation

### File: `streakmind/eval/metrics.py`

```python
def evaluate(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compute detection metrics.

    Returns:
      precision, recall, F1 at IoU threshold 0.5 — overall and per length band
      mAP at IoU 0.5 and 0.75
      mean_angle_error_deg — MAE between predicted and GT OBB angle
      per_band: {
        short:  {precision, recall, F1},   # streak length < 100px
        medium: {precision, recall, F1},   # 100–500px
        long:   {precision, recall, F1},   # > 500px
      }
    """
```

### File: `streakmind/eval/benchmark.py`

Run head-to-head comparison: Co-DINO vs YOLO11-OBB baseline.
- Same test split
- Same metrics from `metrics.py`
- Output markdown table to stdout
- Save per-image results to `streakmind/eval/results/`

Example output:
```
| Model     | mAP@0.5 | mAP@0.75 | Recall | Prec | F1   | Angle MAE |
|-----------|---------|----------|--------|------|------|-----------|
| Co-DINO   | 0.87    | 0.72     | 0.91   | 0.84 | 0.87 | 3.2°      |
| YOLO11-OBB| 0.79    | 0.61     | 0.85   | 0.74 | 0.79 | 6.8°      |
```

### File: `streakmind/eval/visualize.py`

Side-by-side prediction plots: Co-DINO vs YOLO on the same image.
Save to `streakmind/eval/results/viz/` as PNGs.

### Results to record in `results/phase2_baseline.json`

```json
{
  "phase": 2,
  "date_recorded": "",
  "model": "co_dino_swin_l",
  "images_tested": 0,
  "map_50": 0.0,
  "map_75": 0.0,
  "recall": 0.0,
  "precision": 0.0,
  "f1": 0.0,
  "mean_angle_error_deg": 0.0,
  "per_band": {
    "short":  {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "medium": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    "long":   {"precision": 0.0, "recall": 0.0, "f1": 0.0}
  },
  "yolo_baseline": {
    "map_50": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0
  },
  "notes": ""
}
```
