# ARGUS
### Automated Recognition and Grading of Unidentified Streaks

ARGUS is an academic research pipeline for automated satellite streak detection
and identification in FITS telescope images.  It runs four independent detectors
in parallel — two ML-based (DINO-DETR with DINOv3 ViT-B backbone, YOLO11n-OBB)
and two classical (ASTRiDE-derived, OpenCV connected-components) — then merges
their results by grouping overlapping detections across methods rather than
suppressing them, so the UI can surface multi-method agreement.  Streak
orientation is refined via the Radon transform, each streak is traced to its
true endpoints across the full image, and detected objects are cross-identified
against a local TLE catalog using SGP4 propagation and multi-factor confidence
scoring.  Results are served through a FastAPI backend and React frontend.

> **Space-Track integration status:** Cross-identification reads exclusively from
> a local `tle_catalog` database table.  Live Space-Track API queries are
> **not currently active** — integration is pending evaluation for compliance
> with Space-Track's API terms of use.  The TLE catalog must be bootstrapped
> once from an offline bundle (see Setup below).


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

- **StreakMind** — DINO-DETR-based satellite streak detection pipeline
  (prior work that ARGUS builds upon; cite per their published paper/repo).

- **DINO-DETR** — End-to-end object detection with transformers (Detection
  Transformer with Improved deNoising anchOr boxes).
  Zhang et al., *arXiv* 2203.03605 (2022).
  https://arxiv.org/abs/2203.03605

- **Co-DINO** — Co-Deformable DETR object detection transformer.
  Zong et al., *arXiv* 2211.12860 (2023).
  https://arxiv.org/abs/2211.12860
  *Note: Co-DINO pretrained weights (`co_dino_swin_t_coco.pth`) were used to
  initialise the now-archived Swin-T backbone path (`models/dino/streak_codino_swin_t.py`).
  The active model (DINOv3 ViT-B) uses DINO-DETR directly without Co-DINO auxiliary heads.*

- **Danarianto et al.** — Satellite identification prototype pipeline.
  Cite per published paper.

- **DINOv3** — Meta AI self-supervised ViT foundation model (LVD-1689M pretraining).
  Used as a frozen backbone in the `feature/dinov3-backbone` integration.

Code derived from or substantially adapting these works is annotated with
`# Source:` and `# Ref:` comments at the function/class level.

---

## Architecture

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
  `(H, W, 3)` so the DINOv3 detector receives an RGB-like tensor.
- Extracts the WCS solution from the FITS header, or from a same-stem `.wcs`
  sidecar when the FITS header has no celestial WCS. GTImages/SkyTrack uploads
  rely on these sidecars for pixel → RA/Dec conversion.
- Records `wcs_source` as `fits`, `sidecar`, or `null`, and reads the
  observation timestamp (`DATE-OBS`) for SGP4 propagation.

Timing logged: `fits_load_ms`.

### 3. Multi-Method Detection (`inference/pipeline.py` + MMDetection)

Four independent detectors run on every image.  Their raw outputs are collected
into a single pool before downstream processing.

**ML detectors**

| Detector | Architecture | When active |
|----------|-------------|-------------|
| DINO-DETR + DINOv3 ViT-B/16 | DINO-DETR head, frozen ViT-B backbone (`MODEL_SIZE=dinov3_vitb`) | Always (primary) |
| YOLO11n-OBB | Tiled OBB detector, 256 px tiles (`weights/yolo_tiled/run/weights/best.pt`) | Always (when weights present) |

The DINO-DETR path:
- The normalised array is rescaled so its longest edge is ≥ 1280 px (256 px in
  fast mode), then passed to MMDetection's `inference_detector`.
- Every bounding box with score ≥ `CONFIDENCE_THRESHOLD` (default 0.05; lower
  for locally-trained checkpoints) is kept and scaled back to original pixel
  coordinates.
- When `TTA_ENABLED=true`, inference also runs on horizontal and vertical
  flips; bounding boxes are mapped back to original coordinates before merging.

**Classical detectors**

| Detector | Implementation | When active |
|----------|---------------|-------------|
| ASTRiDE | `src/detection/classical_detector.py` | Always |
| OpenCV connected-components | `_run_classical_detector()` in `pipeline.py` | Always |

The OpenCV detector thresholds the top 0.5 % of pixel values, closes short
gaps with a morphological kernel, then retains connected components whose
long-axis length ≥ 80 px and aspect ratio ≥ 5.  PCA on each component gives
the streak angle and endpoints.

ASTRiDE (Phase 0 baseline) uses sigma-thresholded contour detection on the raw
FITS data and is the most sensitive classical path for faint streaks.

**Merging**

After per-detector NMS (rotated-IoU threshold 0.5), all four detection lists
are combined:

```
combined = dino_dets + yolo_dets + classical_dets + astride_dets
```

Overlapping detections from *different* methods are **grouped** by
`streak_id` rather than suppressed — nothing is thrown away.  This lets the
frontend surface multi-method agreement ("3 of 4 detectors agree on this
streak") as a quality signal.

Timing logged: `inference_ms`.

### 4. Radon Angle Refinement (`inference/postprocess.py`)

DINOv3 produces axis-aligned bounding boxes — the streak angle is not directly
predicted.  For each raw detection:

**a) Seed angle from bbox geometry**

`_angle_from_bbox` uses `atan2(height, width)` to produce a rough initial
angle estimate.  This is always more accurate than snapping to 0° or 90°.

**b) Radon transform on the bbox crop**

The image region inside the DINOv3 bounding box is cropped.  Before computing
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

`bbox_to_obb` converts the DINOv3 axis-aligned box and the Radon-refined angle
into an oriented bounding box `{cx, cy, w, h, angle_deg}` where `w` is always
the long axis.

**b) Full-image streak tracing (`extend_obb_to_streak_extent`)**

DINOv3 bounding boxes frequently cover only a portion of a long streak.  This
function traces the streak axis across the entire image to find the true
endpoints:

1. A perpendicular strip of pixels (±3 px wide) is sampled at each integer
   position along the streak axis (parameterised as `t` px from the OBB
   centre, ranging from image edge to image edge).
2. Strip means above `background + 1.5σ` are marked as "bright".
3. Bright positions are grouped into contiguous runs (gap tolerance 5 px).
4. The run containing `t = 0` (the OBB centre, which DINOv3 is guaranteed to
   have detected) is selected as the true streak.  Selecting by containment
   rather than by raw min/max prevents isolated noise spikes beyond the streak
   tip from inflating the endpoint position.
5. The OBB centre and long-axis length are updated to match the selected run.

**c) Per-detector NMS, then cross-detector grouping**

Within each detector's output, OBBs are converted to Shapely polygons and a
greedy NMS pass (sorted by confidence descending) suppresses any detection
whose rotated-IoU with a higher-confidence kept detection exceeds 0.5.  This
collapses TTA's three passes and each classical detector's duplicate firings
down to one box per streak per method.

Detections from *different* methods are then **grouped** rather than
cross-suppressed: overlapping boxes across DINOv3, YOLO, ASTRiDE, and OpenCV
share a common `streak_id`.  All per-method detections are preserved so the
frontend can show multi-method agreement as a quality signal.

### 6. WCS Coordinate Conversion (`inference/pipeline.py`)

The OBB centre pixel `(cx, cy)` is converted to equatorial coordinates using
astropy's `all_pix2world` with the FITS header WCS.  If no WCS is present
(e.g., synthetic test images) `ra_deg` and `dec_deg` are set to `null`.

### 7. Cross-Identification (`inference/crossid.py`)

*Skipped in fast mode (`FAST_MODE=true`).*

For each detection:

1. The local `tle_catalog` table in the ARGUS database is queried for all
   satellites whose TLE epoch is within ±3 days of the observation time.
   Space-Track is **not queried at inference time** — the catalog must be
   pre-loaded (see Setup).
2. SGP4 propagation (via **sgp4** / **skyfield**) computes each candidate
   satellite's sky position at the observation epoch.
3. Angular separation and velocity-vector angle are compared against the
   detected streak's RA/Dec and `angle_deg`.
4. Up to 3 candidates are returned, ranked by a multi-factor confidence score.

If the local catalog has no coverage for the observation time window, ARGUS
leaves the object unidentified (`unknown`) rather than falling back to a live
Space-Track query.

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
├── AGENTS.md                  ← Codex entry point; points to assistant guide
├── CLAUDE.md                  ← Claude Code entry point; points to assistant guide
├── README.md
├── agent_docs/                ← shared assistant guide + reference docs
│   ├── assistant_guide.md     ← canonical assistant instructions
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
│       ├── streak_codino_swin_l.py   ← Swin-L cloud config
│       ├── streak_dinov3_vitb.py     ← DINOv3 ViT-B/16 dev config (Mac MPS)
│       ├── streak_dinov3_vitl.py     ← DINOv3 ViT-L/16 cloud config (GPU)
│       └── dinov3_adapter.py         ← PatchToPyramid adapter + MMDet backbone
├── scripts/
│   ├── make_test_fits.py           ← synthetic FITS generator
│   ├── download_weights.py         ← pretrained weight downloader
│   ├── bootstrap_tle_catalog.py    ← one-time TLE catalog setup
│   ├── update_tle_catalog.py       ← optional explicit GP-class maintenance
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
export MODEL_SIZE=tiny                          # tiny=Swin-T, large=Swin-L, dinov3_vitb, dinov3_vitl
export MODEL_WEIGHTS=weights/dino_tiny.pth      # path to trained checkpoint
export PYTORCH_ENABLE_MPS_FALLBACK=1            # required on Apple Silicon
export DATABASE_URL=sqlite+aiosqlite:///./argus.db

# Optional: lower the confidence threshold for locally-trained models
export CONFIDENCE_THRESHOLD=0.01               # local dev; raise after cloud calibration

# Match preprocessing to the loaded checkpoint:
export ARGUS_NORM=zscore                       # current local Swin-T weights
# export ARGUS_NORM=autostretch                # future autostretch-trained Swin-L weights

# Bootstrap the local TLE catalog (one-time per environment):
# Download a TLE bundle and place it in data/tle_zips/, then run:
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2025

# Space-Track credentials — NOT currently used.
# Live Space-Track API integration is pending evaluation for compliance with
# their terms of use.  The variables below are defined for future use only;
# do not set them in production until the integration is cleared.
# export SPACETRACK_USER=your@email.com
# export SPACETRACK_PASS=yourpassword
# export ARGUS_ENV=development
# export SPACETRACK_BASE_URL=https://for-testing-only.space-track.org/
```

Inference reads only from the local `tle_catalog`. If the catalog has no
coverage for an observation time window, ARGUS leaves the object unidentified
(`unknown`) rather than querying Space-Track.

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

See [Detection Performance](#detection-performance) for current per-method accuracy
on the full merged test split.  The archived Swin-T benchmark (`results/phase8_benchmark.json`)
covers the synthetic dev subset only and is not representative of real-data performance.

---

## Detection Performance

Results on the full merged test split — **308 images, 308 ground-truth streaks**
(SatStreaks dataset, JPEG exports of HST/archival FITS).  Eval metric: IoU ≥ 0.5
against COCO axis-aligned ground-truth bounding boxes, confidence threshold 0.05,
no TTA, per-detector NMS IoU 0.5.

### Per-method results

| Detector | Precision | Recall | F1 | TP | FP | FN |
|----------|----------:|-------:|---:|---:|---:|---:|
| **DINOv3 ViT-B** (DINO-DETR, augmented, epoch 10) | 1.0 % | **93.8 %** | 1.9 % | 289 | 29 552 | 19 |
| **YOLO11n-OBB** (tiled 256 px, epoch 17) | 24.1 % | 4.2 % | 7.2 % | 13 | 41 | 295 |
| **OpenCV** (connected-components) | 5.8 % | 4.2 % | 4.9 % | 13 | 210 | 295 |
| **ASTRiDE** (sigma-threshold) | — | — | — | — | — | — |

### Per-method confusion matrix (IoU ≥ 0.5)

```
DINOv3 ViT-B        Predicted +   Predicted −
  Actual +   TP =    289          FN =     19
  Actual −   FP = 29 552          TN =    n/a

YOLO11n-OBB         Predicted +   Predicted −
  Actual +   TP =     13          FN =    295
  Actual −   FP =     41          TN =    n/a

OpenCV              Predicted +   Predicted −
  Actual +   TP =     13          FN =    295
  Actual −   FP =    210          TN =    n/a
```

### Recall by streak length

| Detector | Short < 400 px (n=6) | Medium 400–999 px (n=18) | Long ≥ 1000 px (n=284) |
|----------|---------------------:|-------------------------:|-----------------------:|
| DINOv3 ViT-B | 50.0 % | 94.4 % | 94.7 % |
| YOLO11n-OBB | 16.7 % | 0.0 % | 4.2 % |
| OpenCV | 0.0 % | 0.0 % | 4.6 % |
| ASTRiDE | — | — | — |

### Training metrics (validation split)

| Detector | Checkpoint | Epochs | mAP | mAP@0.5 |
|----------|-----------|--------|----:|--------:|
| DINOv3 ViT-B | `weights/dinov3_vitb_augmented/best_coco_bbox_mAP_epoch_10.pth` | 10 (augmented, from ep 4) | 35.5 % | 53.2 % |
| YOLO11n-OBB | `weights/yolo_tiled/run/weights/best.pt` | 18 (tiled 256 px) | 27.4 % | 59.6 % |

### Interpretation

**DINOv3 ViT-B** achieves very high recall (93.8 %) — it misses only 19 of 308
ground-truth streaks — but produces a large number of false positives at the
default 0.05 confidence floor (~96 FP per image on average).  Raising the
confidence threshold via the filter UI significantly improves precision while
preserving recall for high-confidence detections.  Long streaks (≥ 1000 px)
are detected at 94.7 % recall, confirming the model generalises well to
full-resolution archival images.

**YOLO11n-OBB** is conservative — 24 % precision, 4 % recall on full images.
It was trained on 256 px tiles and evaluates on full images, so long streaks
(≥ 1000 px, 92 % of this test set) that span many tiles are under-detected.
YOLO's role in the ensemble is corroboration: when it agrees with ViT-B on a
streak, that agreement is a strong quality signal.

**OpenCV** (connected-components) provides marginal recall on this JPEG test set
but adds value on raw FITS images where the pixel distribution is well-suited to
the top-0.5 % brightness threshold.  It requires no learned weights and runs in
< 1 s per image.

**ASTRiDE** requires raw FITS pixel data (it thresholds against the background
sigma of the original science image).  The full SatStreaks test set is distributed
as JPEG exports, so ASTRiDE cannot be evaluated here.  On raw GTImages FITS
observations it detects faint streaks that escape the ML detectors, at the cost
of higher false-positive rates on noisy images.

> **Multi-method ensemble note:** The four detectors run independently and their
> outputs are grouped by overlap rather than suppressed.  A streak seen by
> multiple detectors gives a user higher confidence than one seen by a single method.
> The filter UI exposes per-method confidence sliders so the precision/recall
> tradeoff can be tuned interactively after results are returned.
> In the future, we may explore more sophisticated ensembling strategies (e.g., a learned
> meta-classifier that takes all four detectors' outputs as input and produces a final confidence score), but the current approach has the advantage of transparency and user control.

---

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
| MacBook Air M3 (16 GB) | Development, testing | `MODEL_SIZE=tiny` or `dinov3_vitb`, MPS |
| RTX 5070 Ti 16 GB (workstation) | DINOv3 ViT-L training (Phase D) | `MODEL_SIZE=dinov3_vitl`, CUDA |
| Lambda Labs A100 40 GB | Swin-L training or ViT-L Stage 2 unfreeze | `MODEL_SIZE=large`, CUDA |

Never hardcode `torch.device("cuda")` — always use `get_device()` from
`inference/device.py`.
