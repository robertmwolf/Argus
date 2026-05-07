# ARGUS
### Automated Recognition and Grading of Unidentified Streaks

ARGUS is an academic research pipeline for automated satellite streak detection
and identification in FITS telescope images.  It detects satellite streaks using
a Co-DINO transformer model (Swin backbone), refines streak orientation via the
Radon transform, traces each streak to its true endpoints across the full image,
and cross-identifies detected objects against a local TLE catalog — sourced once
from Space-Track and stored in the ARGUS database — using SGP4 propagation and
multi-factor confidence scoring.  Results are served through a FastAPI backend
and React frontend.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 — Classical baseline | ASTRiDE detection, SGP4 matching | ✅ Complete |
| 1 — Data pipeline | FITS loader, COCO labels, augmentations | ✅ Complete |
| 2 — DINO model | device.py, MMDet configs, train script | ✅ Complete |
| 3 — Inference pipeline | orchestrator, Radon postprocess, crossid | ✅ Complete |
| 4 — Database | SQLAlchemy schema, async models | ✅ Complete |
| 5 — API | FastAPI upload / result endpoints | ✅ Complete |
| 6 — Frontend | React + Vite, canvas OBB rendering | ✅ Complete |
| 7 — Docker | docker-compose with GPU worker | ✅ Complete |
| 8 — Evaluation | mAP, angle error, DINO vs ASTRiDE | ✅ Complete |

---

## Name

**ARGUS** — **Automated Recognition and Grading of Unidentified Streaks**.

Also a reference to **Argus Panoptes** (Ἄργος Πανόπτης), the hundred-eyed
giant of Greek mythology — always vigilant, always watching.

---

## Research Context

This project builds on and cites the following prior works:

- **ASTRiDE** — Automated Streak Detection for Astronomical Images.
  Kim et al., *Astronomical Journal* (2017).
  https://github.com/dwkim78/ASTRiDE

- **StreakMind** — Co-DINO transformer-based satellite streak detection pipeline
  (prior work that ARGUS builds upon; cite per their published paper/repo).

- **Co-DINO** — Co-Deformable DETR object detection transformer.
  Zong et al., *arXiv* 2211.12860 (2023).
  https://arxiv.org/abs/2211.12860

- **Danarianto et al.** — Satellite identification prototype pipeline.
  Cite per published paper.

Code derived from or substantially adapting these works is annotated with
`# Source:` and `# Ref:` comments at the function/class level.

---

## Architecture

```
FITS Image
    │
    ├─ Phase 0/3: Classical path (src/)
    │   ├── ingest/fits_parser.py        — FITS → FITSImage dataclass
    │   ├── detection/classical_detector.py  — ASTRiDE streak extraction
    │   ├── astrometry/plate_solver.py   — pixel → RA/Dec (WCS)
    │   └── matching/                    — Space-Track → SGP4 → scorer → matcher
    │
    └─ Phase 2–8: ML path
        ├── inference/fits_loader.py     — FITS → normalised uint8 array + FITS/sidecar WCS
        ├── inference/device.py          — get_device(): CUDA → MPS → CPU
        ├── models/dino/                 — DINO Swin-T (dev) / Swin-L (cloud) configs
        ├── training/train_dino.py       — two-stage fine-tuning + cost guardrails
        ├── inference/pipeline.py        — end-to-end orchestrator
        ├── inference/postprocess.py     — Radon angle refinement + extent tracing + NMS
        ├── inference/crossid.py         — TLE cross-identification
        ├── api/main.py                  — FastAPI: upload / queue / result endpoints
        └── frontend/                    — React 18 + Vite: canvas OBB overlay
```

Full design: [`agent_docs/architecture.md`](agent_docs/architecture.md)

---

## End-to-End Pipeline

This describes exactly what happens from the moment a FITS file is uploaded
to the moment results appear in the browser.

### 1. Upload (Browser → API)

The user drags a `.fits` file onto the React frontend.  The browser POSTs it
to `POST /api/upload`, which:

1. Saves the file to the configured storage backend (local disk or S3).
2. Creates a job record in the SQLite/PostgreSQL database with status `queued`.
3. Returns a `job_id` UUID to the browser.
4. Enqueues the job to the background worker (in-memory queue or SQS).

### 2. FITS Loading (`inference/fits_loader.py`)

The background worker calls `inference.pipeline.run(fits_path)`.  The first
stage opens the FITS file with **astropy** and:

- Reads the primary HDU pixel data (any bit depth).
- Performs Z-score normalisation: subtract mean, divide by standard deviation,
  then rescale to uint8 [0, 255] clipped at ±3σ.
- Converts the single-channel science image to a 3-channel uint8 array
  `(H, W, 3)` so the DINO detector receives an RGB-like tensor.
- Extracts the WCS solution from the FITS header, or from a same-stem `.wcs`
  sidecar when the FITS header has no celestial WCS. GTImages/SkyTrack uploads
  rely on these sidecars for pixel → RA/Dec conversion.
- Records `wcs_source` as `fits`, `sidecar`, or `null`, and reads the
  observation timestamp (`DATE-OBS`) for SGP4 propagation.

Timing logged: `fits_load_ms`.

### 3. DINO Detection (`inference/pipeline.py` + MMDetection)

The normalised array is passed to a Co-DINO transformer model (Swin-T backbone
for local dev, Swin-L for cloud training).

- The image is rescaled so its longest edge equals `image_size` (400 px on Mac,
  256 px in fast mode).
- MMDetection's `inference_detector` runs a forward pass.
- Every predicted bounding box with score ≥ `CONFIDENCE_THRESHOLD` (default
  0.10 for locally-trained Swin-T; raise to 0.30 for cloud Swin-L) is kept.
- Bounding boxes are scaled back to original image pixel coordinates.

DINO can return multiple overlapping detections for the same physical streak
(common for long streaks that span multiple attention heads).  These are
deduplicated in stage 5.

Timing logged: `inference_ms`.

### 4. Radon Angle Refinement (`inference/postprocess.py`)

DINO produces axis-aligned bounding boxes — the streak angle is not directly
predicted.  For each raw detection:

**a) Seed angle from bbox geometry**

`_angle_from_bbox` uses `atan2(height, width)` to produce a rough initial
angle estimate.  This is always more accurate than snapping to 0° or 90°.

**b) Radon transform on the bbox crop**

The image region inside the DINO bounding box is cropped.  Before computing
the Radon transform:

- The crop is converted to float32 greyscale.
- The sky background (image median) is subtracted and negative values are
  clipped to zero.  Without this step the high DC sky level (~120 counts)
  dominates the Radon variance at all angles, pulling the estimate toward the
  axis aligned with the crop geometry.

The Radon sinogram is computed over a ±45° window around the seed angle.
The sinogram column with maximum variance corresponds to the projection where
the streak integrates to a single bright, narrow peak — i.e. the true streak
orientation.  The winning Radon angle is converted back to image streak angle:

```
φ_streak = 90° − θ_radon   (mod 180°)
```

This produces sub-degree angle precision without any GPU compute (scikit-image
Radon is CPU-only).

Timing is included in `postprocess_ms`.

### 5. OBB Construction and Streak Extent (`inference/postprocess.py`)

**a) Initial OBB from refined angle**

`bbox_to_obb` converts the DINO axis-aligned box and the Radon-refined angle
into an oriented bounding box `{cx, cy, w, h, angle_deg}` where `w` is always
the long axis.

**b) Full-image streak tracing (`extend_obb_to_streak_extent`)**

DINO bounding boxes frequently cover only a portion of a long streak.  This
function traces the streak axis across the entire image to find the true
endpoints:

1. A perpendicular strip of pixels (±3 px wide) is sampled at each integer
   position along the streak axis (parameterised as `t` px from the OBB
   centre, ranging from image edge to image edge).
2. Strip means above `background + 1.5σ` are marked as "bright".
3. Bright positions are grouped into contiguous runs (gap tolerance 5 px).
4. The run containing `t = 0` (the OBB centre, which DINO is guaranteed to
   have detected) is selected as the true streak.  Selecting by containment
   rather than by raw min/max prevents isolated noise spikes beyond the streak
   tip from inflating the endpoint position.
5. The OBB centre and long-axis length are updated to match the selected run.

**c) Rotated-IoU NMS**

All OBBs are converted to Shapely polygons.  A greedy NMS pass (sorted by
confidence descending) suppresses any detection whose rotated-IoU with a
higher-confidence kept detection exceeds 0.5.

### 6. WCS Coordinate Conversion (`inference/pipeline.py`)

The OBB centre pixel `(cx, cy)` is converted to equatorial coordinates using
astropy's `all_pix2world` with the FITS header WCS.  If no WCS is present
(e.g., synthetic test images) `ra_deg` and `dec_deg` are set to `null`.

### 7. Cross-Identification (`inference/crossid.py`)

*Skipped in fast mode (`FAST_MODE=true`).*

For each detection:

1. The local `tle_catalog` table in the ARGUS database is queried for all
   active satellites whose TLE epoch is within ±3 days of the observation time.
2. SGP4 propagation (via **sgp4** / **skyfield**) computes each candidate
   satellite's sky position at the observation epoch.
3. Angular separation and velocity-vector angle are compared against the
   detected streak's RA/Dec and `angle_deg`.
4. Up to 3 candidates are returned, ranked by a multi-factor confidence score.

Timing logged: `crossid_ms`.

### 8. Result Delivery (API → Browser)

The worker updates the job status to `complete` in the database and stores the
detection JSON.  The frontend polls `GET /api/result/{job_id}` until the job
is done. The result payload includes `image_width` and `image_height`, which
the canvas uses to scale detections in original source-image coordinates even
when the preview PNG has been resized. The browser then:

- Loads the original (unmodified) FITS preview image from `GET /api/preview/{job_id}`.
- Renders detection overlays on an HTML5 canvas:
  - Thin rotated bounding box outline at 55% opacity.
  - Dashed streak centreline from endpoint to endpoint.
  - Filled endpoint circles with dark inner rings for contrast.
  - Angle arc from horizontal with degree label.
  - Numbered badge above each detection.
- Hovering over a detection shows a tooltip: confidence, streak length, angle,
  RA/Dec, and the top cross-identification match if available.

---

## Project Structure

```
Argus/
├── AGENTS.md                  ← agent coding instructions
├── README.md
├── agent_docs/                ← read before writing any code
│   ├── architecture.md
│   ├── streakmind_phases.md   ← Phases 2–8 spec
│   ├── phase1_goals.md        ← Phase 1 (complete, reference only)
│   ├── datasets.md
│   ├── dependencies.md
│   ├── service_roadmap.md
│   ├── spacetrack.md
│   └── test_strategy.md
├── src/                       ← Phase 0: classical baseline (complete)
│   ├── ingest/fits_parser.py
│   ├── detection/classical_detector.py
│   ├── astrometry/plate_solver.py
│   └── matching/              ← scorer, propagator, spatial_filter, matcher, spacetrack_query, tle_store
├── inference/                 ← ML inference modules
│   ├── device.py              ← get_device() / get_device_config()
│   ├── fits_loader.py         ← FITS → tensor, normalisation + FITS/sidecar WCS
│   ├── pipeline.py            ← inference orchestrator
│   ├── postprocess.py         ← Radon angle refinement + streak extent + NMS
│   └── crossid.py             ← satellite cross-matching
├── training/
│   ├── convert_labels.py      ← YOLO OBB → COCO JSON
│   ├── dataset.py             ← FITSStreakDataset
│   ├── augmentations.py       ← albumentations + SyntheticStreakInject
│   ├── train_dino.py          ← DINO training script
│   └── train_baseline.py      ← YOLO11-OBB baseline
├── models/
│   └── dino/
│       ├── streak_codino_swin_t.py   ← Swin-T dev config
│       └── streak_codino_swin_l.py   ← Swin-L cloud config
├── scripts/
│   ├── make_test_fits.py           ← synthetic FITS generator
│   ├── download_weights.py         ← pretrained weight downloader
│   ├── bootstrap_tle_catalog.py    ← one-time TLE catalog setup
│   ├── update_tle_catalog.py       ← hourly TLE refresh (GP class)
│   ├── merge_annotations.py        ← SatStreaks mask + GTImages COCO split merger
│   └── prepare_cloud_training.py   ← go/no-go checklist before GPU rental
├── api/                       ← FastAPI application
├── frontend/                  ← React 18 + Vite + Tailwind
├── eval/                      ← metrics, benchmark, results
├── db/                        ← schema.sql, async ORM models
├── docker/                    ← docker-compose (deploy only)
├── tests/                     ← pytest — all offline, no credentials required
├── data/
│   ├── sample/                ← synthetic FITS for smoke-testing
│   ├── GTImages/              ← labeled satellite streak observations (gitignored)
│   ├── annotations/           ← COCO JSON label files
│   └── catalogs/              ← TLE catalog files
├── weights/                   ← model weights (gitignored)
└── results/                   ← baseline metrics JSON output
```

---

## Setup

```bash
# Create and activate the conda environment
conda create -n satid python=3.11
conda activate satid
pip install -r requirements.txt

# Required env vars for local dev
export MODEL_SIZE=tiny                          # tiny=Swin-T (Mac), large=Swin-L (A100)
export MODEL_WEIGHTS=weights/dino_tiny.pth      # path to trained checkpoint
export PYTORCH_ENABLE_MPS_FALLBACK=1            # required on Apple Silicon
export DATABASE_URL=sqlite+aiosqlite:///./argus.db

# Optional: lower the confidence threshold for locally-trained models
export CONFIDENCE_THRESHOLD=0.10               # default; raise to 0.30 after cloud training

# Match preprocessing to the loaded checkpoint:
export ARGUS_NORM=zscore                       # current local Swin-T weights
# export ARGUS_NORM=autostretch                # future autostretch-trained Swin-L weights

# Bootstrap the local TLE catalog (one-time per environment):
# 1. Download the 2025 annual bundle from Space-Track's cloud storage:
#    https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu
#    Place the file in data/tle_zips/
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2025

# Space-Track credentials (only needed for live TLE maintenance, not inference):
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword

# Update the catalog with the latest active satellites (≤ once/hour):
python scripts/update_tle_catalog.py
```

## Running Locally (Dev)

Run the API directly with the satid conda environment — Docker is reserved for
deployment.  The satid env has torch, mmdet, and all ML packages installed.

```bash
conda activate satid
export MODEL_SIZE=tiny
export MODEL_WEIGHTS=weights/dino_tiny.pth
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Start the API (port 8000)
uvicorn api.main:app --reload --port 8000

# Start the frontend dev server (port 5173)
cd frontend && npm run dev
```

Open `http://localhost:5173` in your browser and upload a FITS file.

## Running with Docker (Deploy)

Docker images should only be built for deployment — they do not include
torch/mmdet (GPU-intensive packages belong in the dedicated worker image).

```bash
# 1. Copy and edit credentials
cp .env.example .env

# 2. Start the full stack (db + api + frontend)
docker compose up --build

# 3. Open http://localhost in your browser

# Cloud deployment (S3 + SQS + GPU worker):
docker compose -f docker-compose.yml -f docker-compose.cloud.yml up
```

## Running Tests

```bash
conda activate satid
pytest tests/ -v
# All tests are offline (mocked) — no GPU, no Space-Track credentials required
```

## Generating Synthetic Test Data

No real FITS files are required to develop and test the pipeline:

```bash
python scripts/make_test_fits.py --small   # fast 512×512 images
python scripts/make_test_fits.py           # full 3096×2080 images
```

## Downloading Pretrained Weights

```bash
# Swin-T weights for local development (~160 MB):
MODEL_SIZE=tiny python scripts/download_weights.py

# Swin-L weights for cloud training (~828 MB, A100 only):
MODEL_SIZE=large python scripts/download_weights.py
```

## Local Training (Mac M3, no GPU required)

All phases 0–8 are code-complete with 325 passing tests.  To produce real
detection results without a cloud GPU, train the Swin-T model on the 50-image
dev subset:

```bash
# 1. Get annotated training data (~150 MB)
git clone https://github.com/jijup/SatStreaks data/satstreaks

# 2. Build the 50-image dev subset
python training/make_dev_subset.py

# 3. Smoke-test MPS training pipeline (~5 min)
PYTORCH_ENABLE_MPS_FALLBACK=1 MODEL_SIZE=tiny \
  python -m training.train_dino --smoke-test

# 4. Train YOLO11-OBB baseline (~30 min)
MODEL_SIZE=tiny python -m training.train_baseline

# 5. Train Swin-T DINO on dev subset (~1–2 hrs on MPS)
PYTORCH_ENABLE_MPS_FALLBACK=1 MODEL_SIZE=tiny USE_DEV_SUBSET=true \
  python -m training.train_dino --work-dir weights/local_run

# Optional: resume from an existing checkpoint or run a short timeboxed retrain
python -m training.train_dino \
  --work-dir weights/local_run \
  --load-from weights/local_run/best_coco_bbox_mAP_epoch_50.pth \
  --max-epochs 10 \
  --val-interval 2 \
  --checkpoint-interval 2

# 6. Run inference with trained weights
MODEL_WEIGHTS=weights/local_run/best_coco_bbox_mAP_epoch_50.pth \
  MODEL_SIZE=tiny PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python -m inference.pipeline --image data/sample/synth_streak_000.fits

# 7. Evaluate (loads model once for all images)
python -m eval.benchmark \
  --run-pipeline \
  --annotations data/annotations/dev_subset.json \
  --output results/phase8_benchmark.json
```

Recorded local results (50-image dev subset, Swin-T, CPU, 50 epochs):
- DINO Swin-T: mAP@0.5=65.7%, precision=66.7%, recall=73.3%, F1=69.8%
- YOLO11-OBB: mAP@0.5=36.0%, precision=63.2%, recall=40.0%, angle error=0.66°

Swin-L on the full dataset is needed for ≥94% precision / ≥97% recall (paper targets).

## Cloud Training (Lambda A100)

```bash
# Before renting GPU — verify all checks pass:
MODEL_SIZE=large python scripts/prepare_cloud_training.py

# On Lambda instance (run once):
bash scripts/cloud_setup.sh

# Full Swin-L training (~4–8 hrs on A100; longer on smaller CUDA cards):
MODEL_SIZE=large python -m training.train_dino --work-dir weights/run_001

# Fetch weights back to Mac:
bash scripts/fetch_weights.sh user@instance-ip
```

## Real Training Data

The model requires annotated FITS images.  Two sources:

| Dataset | Images | Format | Role |
|---------|--------|--------|------|
| **SatStreaks** | 3,073 annotated masks | PNG/JPEG + segmentation masks | Primary training corpus — [GitHub](https://github.com/jijup/SatStreaks) |
| **GTImages** | 759 FITS (593 labeled + 93 negatives) | FITS + `.strk` annotations | Validation, negative examples, cross-ID benchmark — `data/GTImages/` |

```bash
# Convert GTImages .strk annotations to COCO JSON:
python scripts/convert_gtimages.py \
    --strk-dir data/GTImages \
    --output data/annotations/gtimages.json \
    --negatives-output data/annotations/gtimages_negatives.json

# Merge SatStreaks and GTImages into train/val/test splits.
# SatStreaks segmentation masks are converted to real COCO bboxes at merge time.
python scripts/merge_annotations.py --seed 42 --val-fraction 0.2
```

For workstation handoff training, see
[`agent_docs/Training_Handoff.md`](agent_docs/Training_Handoff.md). The staged
data bundle should include `data/Manifest.txt` with dataset counts and the
expected results branch.

---

## Hardware

| Machine | Use | Config |
|---------|-----|--------|
| MacBook Air M3 (16 GB) | Development, testing | `MODEL_SIZE=tiny`, MPS |
| Lambda Labs A100 40 GB | Training | `MODEL_SIZE=large`, CUDA |

Never hardcode `torch.device("cuda")` — always use `get_device()` from
`inference/device.py`.
