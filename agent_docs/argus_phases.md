# ARGUS Phases 2–8

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
| `training/train_dino.py` | Two-stage training, Stage2UnfreezeHook (epoch 21), CostGuardrailHook, --smoke-test, checkpoint/timebox overrides |
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
- **Swin-T/Swin-L weights**: `scripts/download_weights.py` downloads the
  model-size-appropriate DINO COCO pretrain checkpoint. Local fine-tuned
  Swin-T checkpoints can be selected with `MODEL_WEIGHTS`.
- **Training overrides**: use `--resume` for interrupted runs, `--load-from`
  to initialize a new run from a checkpoint, and `--max-epochs`,
  `--val-interval`, / `--checkpoint-interval` for timeboxed retraining.

### Gate ✅ cleared
- Both configs parse with mmengine: `DINO` model type, 1 class `streak`,
  Z-score mean/std, gradient checkpointing enabled
- `get_device()` returns `mps` on Mac; `get_device_config()` returns
  correct MPS-safe values
- `make_test_fits.py` generates valid FITS with WCS headers
- 206 tests passing (at time of Phase 2 completion)

---

## Phase 3 — Inference Pipeline  ✅ COMPLETE

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
        fast: If True, keep Radon refinement but skip crossid and DB write.
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

Radon-based angle refinement, endpoint tracing, detector-level NMS, and
streak-level grouping/fusion.

```python
def refine_angle(
    image_crop: np.ndarray,
    obb: dict,
    angle_search_range: float = 15.0,
) -> float:
    """Refine OBB angle using the Radon transform on the streak crop.

    Source: StreakMind — Radon angle refinement
    Ref: agent_docs/argus_phases.md

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

def group_detections(
    detections: list[dict],
    iou_threshold: float = 0.5,
    iom_threshold: float = 0.3,
) -> list[dict]:
    """Assign streak_id using rotated-IoU, IoMin, or collinear-fragment matching."""

def fuse_group_geometries(detections: list[dict]) -> list[dict]:
    """Fuse grouped fragments into one endpoint-spanning OBB per streak_id."""
```

#### `inference/crossid.py`

Satellite ephemeris cross-matching against the local `tle_catalog` table.
Inference does not query Space-Track directly; missing local coverage leaves
the object unidentified/unknown.

Does not require `SPACETRACK_USER` or `SPACETRACK_PASS` for inference.

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
    """Cross-match detections against the local TLE catalog.

    Queries the local tle_catalog table for the epoch window around obs_time,
    leaves detections unidentified when local coverage is missing, propagates
    candidates via SGP4, and scores candidates using Gaussian position
    confidence.
    """
```

TLE data source: local `tle_catalog` table in `argus.db` / PostgreSQL,
bootstrapped from Space-Track annual bundles. Space-Track GP/GP_History helpers
are reserved for explicit maintenance or diagnostics, not automatic inference
fallbacks.

### Tests to write

`tests/test_pipeline.py`:
- `run()` on a synthetic FITS completes without error in fast mode
- Returns list of dicts with required keys
- `FAST_MODE=true` keeps Radon refinement but skips cross-ID / DB writes

`tests/test_postprocess.py`:
- `refine_angle` on a synthetic streak returns angle within ±5° of ground truth
- `nms_detections` removes overlapping boxes, keeps highest confidence

`tests/test_crossid.py`:
- `cross_identify` with a known TLE returns top-3 candidates
- Candidate with lowest angular separation has highest confidence
- Missing sky coords → identifications empty list, no crash
- local TLE catalog queries preserve (name, line1, line2)
- missing local TLE coverage returns empty identifications without Space-Track calls

### Gate condition for Phase 4
`inference/pipeline.py --fast --image data/sample/synth_streak_000.fits`
completes in <60 seconds and returns at least one detection.

---

## Phase 4 — Database  ✅ COMPLETE

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

## Phase 5 — API  ✅ COMPLETE

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

## Phase 6 — Frontend  ✅ COMPLETE

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

### Gate condition for Phase 8
Upload FITS in browser, see annotated image with OBBs and table.

---

## Phase 8 — Evaluation  ✅ COMPLETE (local dev results recorded; cloud training pending)

### `eval/metrics.py`

```python
def evaluate(predictions, ground_truth) -> dict:
    """Returns: precision, recall, F1 @ IoU 0.5; mAP@0.5, mAP@0.75;
    mean_angle_error_deg; per_band (short/medium/long streaks)."""
```

IoU note: ground-truth streaks are ~3 px wide; DINO outputs axis-aligned bboxes.
`_obb_iou()` falls back to axis-aligned bbox IoU when GT height < 5 px so narrow
synthetic streaks score correctly against DINO predictions.

### `eval/benchmark.py`
Head-to-head: DINO vs YOLO11-OBB baseline on same test split.
Output markdown table + save per-image results to `eval/results/`.

**Batch inference API** — the benchmark loads the DINO model once before the
image loop (not once per image):

```python
from inference.pipeline import load_model, run as pipeline_run

dino_model, dino_device = load_model()          # one checkpoint load
for img_info in coco["images"]:
    dets = pipeline_run(
        fits_path=fits_path, fast=True,
        model=dino_model, inference_device=dino_device,  # reuse loaded model
    )
```

Target metrics (from StreakMind paper):
- DINO Swin-L: ≥94% precision, ≥97% recall
- YOLO baseline: reference comparison

### Local dev results: `results/phase8_benchmark.json`
Recorded 2026-05-05. Swin-T, 50 epochs, 50-image synthetic dev subset, 256×256px, CPU.

| Metric | DINO Swin-T | YOLO11-OBB | Target |
|--------|-------------|------------|--------|
| mAP@0.5 | 65.7% | 36.0% | — |
| Precision | 66.7% | 63.2% | ≥94% |
| Recall | 73.3% | 40.0% | ≥97% |
| F1 | 69.8% | 49.0% | — |
| Angle error | 29.6°* | 0.66° | — |

*DINO angle is Radon-refined in fast mode; cross-ID and DB writes are skipped.
YOLO angle is real (OBB corner-point output).

### Phase E results: `results/phase_e/phase_e_comparison_test.json`
Recorded 2026-05-15. Evaluation on held-out `test.json` (full merged dataset).

| Model | mAP | mAP@0.5 | mAP@0.75 | Notes |
|-------|-----|---------|---------|-------|
| Co-DINO Swin-T | 0.149 | 0.190 | 0.167 | full merged dataset |
| DINOv3 ViT-B (Phase C²) | 0.580 | **0.740** | 0.606 | frozen backbone, full dataset, 4 epochs |
| YOLO11n-OBB (full dataset) | 0.561† | 0.673† | 0.574† | 15 epochs, 3 023 images → 14 385 tiles, Mac M3 CPU; P=57% R=85% F1=68% |
| DINOv3 ViT-L (Phase D) | TBD | TBD | TBD | pending workstation run |

†YOLO metrics are from YOLO's native tiled val split (2 881 tiles, 604 source images),
not the full-image COCO test.json evaluated above.  The two evaluation protocols
are not directly comparable — tiled IoU matching inflates mAP relative to full-image GT.
`weights/run_full_yolo_obb/run/weights/best.pt` (5.4 MB), best at epoch 13/15.

DINOv3 ViT-B (frozen) outperforms Swin-T by +0.55 mAP@0.5 on a fair comparison (same data, test split).
Phase D (ViT-L, 50 epochs) is the definitive production run targeting ≥94% precision / ≥97% recall.

### Multi-method benchmark: `results/multi_method_benchmark.json`
Recorded 2026-05-16. 308-image SatStreaks test set, confidence threshold 0.05,
per-detector NMS IoU 0.5, cross-detector grouping IoU ≥ 0.5, IoMin ≥ 0.3,
or collinear-fragment match. Grouped fragments are fused to a single OBB
spanning their outer projected endpoints before WCS/cross-ID.
Confusion matrix PNGs in `results/confusion_matrices/`.

| Method | Precision | Recall | F1 | mAP@0.5 | mAP@0.75 | n preds |
|--------|----------:|-------:|---:|---------:|---------:|--------:|
| **Unified Confidence Score** (confidence floor + F-beta corroboration) | **29.9 %** | 72.1 % | **42.3 %** | 40.6 % | 31.8 % | 742 |
| DINOv3 ViT-B | 9.3 % | **89.3 %** | 16.8 % | **75.5 %** | **59.4 %** | 2 969 |
| OpenCV | 1.4 % | 1.0 % | 1.1 % | 0.01 % | 0.01 % | 223 |

Key finding: weighted fusion grouping collapses 3 192 individual predictions to 742,
raising precision from 9.3 % to 29.9 % while retaining 72 % recall (F1 42.3 %).
On long streaks (≥ 1 000 px, 92 % of test set) Unified F1 = 49 % vs 16.9 % for
DINOv3 alone.

The Unified Confidence Score is computed by `inference/confidence.py`.  ASTRiDE is
corroboration-only: groups where ASTRiDE is the only detector are lowered to a
conservative confidence in `inference/pipeline.py`, and corroborated ASTRiDE
detections can only add a small bounded score boost. The non-ASTRiDE fusion formula:

1. Cap each non-ASTRiDE detector's raw confidence at its optional
   `confidence_ceiling` before any weighting.
2. Use the best effective non-ASTRiDE confidence as the score floor, so a
   single detector is not penalized merely because no other model agreed.
3. Weight each additional detector's corroborating evidence by its F-0.5 score
   (`w = 1.25×P×R / (0.25×P + R)`).
4. Combine corroborating contributions into the remaining confidence mass,
   tempering only that boost when non-ASTRiDE detectors strongly disagree.
5. If ASTRiDE corroborates a non-ASTRiDE group, add only a small bounded boost
   (`0.04 × best_astride_conf`), never letting ASTRiDE lower the fused score.

This is *not* equal Noisy-OR.  **After each new training run, update `DETECTOR_PROFILES`
in `inference/confidence.py` with the measured precision and recall** — see the README
section "Updating Detector Profiles After Training".  If a newly added detector emits
unreliable confidence magnitudes, also set its `confidence_ceiling`. Do not convert
ASTRiDE back into a normal weighted detector.

### Ensemble v2 benchmark: `results/ensemble_benchmark_20260526/`
Recorded 2026-05-26. Full three-set benchmark after updating detector profiles,
adding per-band reliability weights, and preferring YOLO's tight OBBs in geometry
fusion. Script: `scripts/run_ensemble_benchmark.py`. IoU threshold 0.50.
P/R/F1 reported at conf ≥ 0.30. mAP computed on unfiltered predictions.

**SatStreaks test (308 images, in-distribution):**

| Method | mAP@0.50 | P @0.30 | R @0.30 | F1 @0.30 |
|--------|:--------:|:-------:|:-------:|:--------:|
| DINOv3 ViT-B Multisource | **0.091** | **26.5 %** | **26.6 %** | **26.6 %** |
| YOLO-OBB GTImages (single-pass) | 0.017 | 7.5 % | 10.4 % | 8.7 % |
| ASTRiDE | — (JPEG, skipped) | — | — | — |
| **Unified Ensemble v2** | 0.079 | 24.1 % | 26.6 % | 25.3 % |

**
| Method | mAP@0.50 | P @0.30 | R @0.30 | F1 @0.30 |
|--------|:--------:|:-------:|:-------:|:--------:|
| DINOv3 ViT-B Multisource | **0.083** | **25.4 %** | **24.9 %** | **25.2 %** |
| YOLO-OBB GTImages (single-pass) | 0.016 | 7.4 % | 9.6 % | 8.4 % |
| ASTRiDE | — (JPEG, skipped) | — | — | — |
| **Unified Ensemble v2** | 0.072 | 23.0 % | 24.9 % | 23.9 % |

**BrentImages (50-image sample, real FITS — Atwood observatory):**

| Method | mAP@0.50 | R @0.30 | Notes |
|--------|:--------:|:-------:|-------|
| DINOv3 ViT-B Multisource | 0.002 | 2.3 % | GT OBBs are tight (h≈16 px); IoU vs loose DINO boxes ≈ 0 |
| YOLO-OBB GTImages | 0.000 | 0.0 % | Single-pass 8192 px loses tile-scale detail |
| ASTRiDE | 0.001 | 2.3 % | 604 detections/image (capped at 50); essentially random recall |
| **Unified Ensemble v2** | 0.002 | 2.3 % | Same as DINO; ASTRiDE noise absorbed |

BrentImages metrics are not directly comparable with the other two sets.
The GT annotations use tight oriented bounding boxes (h ≈ 16 px) while DINO
outputs large axis-aligned boxes (~2 000 × 3 000 px). IoU between the two is
structurally near-zero regardless of correct localisation. A meaningful
evaluation on BrentImages requires IoM (intersection-over-minimum) or a
purpose-built overlap metric for thin-streak GT.

**Key findings:**

1. **DINOv3 is the strongest individual model** — 5–6× higher mAP than YOLO
   on both evaluatable sets; generalises well zero-shot to 2. **Unified Ensemble v2 is marginally worse than DINO alone** (mAP 0.079 vs
   0.091 on SatStreaks). The corroboration weighting is slightly downgrading
   valid DINO detections instead of boosting them.
3. **YOLO-OBB is weak at single-pass 8 192 px** — the model was trained on
   640 px tiles; running full-frame loses the scale at which it learned.
   Re-enabling tiled inference (STREAKMIND_YOLO_TILE_SIZE=640) should recover
   the ~80 % medium-streak recall seen in the per-detector analysis.
4. **ASTRiDE contributes nothing** — cannot run on JPEG inputs (SatStreaks,
      maximum confidence boost is capped at 4 %. **Disabled by default** as of
   2026-05-26 (`_ASTRIDE_ENABLED_BY_DEFAULT = False` in `inference/pipeline.py`).
   Re-enable with `ARGUS_ENABLE_ASTRIDE=1`.

**Future improvement — ensemble weight tuning:**

The per-band detector weights in `inference/confidence.py` (`DETECTOR_PROFILES`)
were set from a single-run analysis and have not been validated against held-out
data. The ensemble will not outperform the best individual model until the weights
are calibrated. Recommended next steps:

- Re-enable YOLO tiled inference and re-run the benchmark to get accurate
  per-band recall numbers for YOLO (especially medium streaks 150–400 px).
- Run a grid search or Bayesian optimisation over `band_weights` in
  `DETECTOR_PROFILES` targeting F1 on the validation split.
- Consider replacing the fixed F-beta weighting with a learned isotonic
  calibration layer trained on the validation set confidence distributions.
- Evaluate on BrentImages using IoM (add `iom_threshold` parameter to
  `_greedy_match` in `eval/metrics.py`) to get meaningful results on
  observatory-quality tight OBBs.
