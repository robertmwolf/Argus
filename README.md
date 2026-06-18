# ARGUS
### Automated Recognition and Grading of Unidentified Streaks

ARGUS is a pipeline for automated satellite streak detection
and identification in FITS telescope images.  The primary detector is a
**DINOv3 ViT-S/16 heatmap model** (Run 15 — trained on Atwood Observatory FITS
at 400 px native tiles, z-score normalisation) that produces tiled probability
maps which are stitched into line-segment detections.  ASTRiDE provides a
classical corroboration signal on raw FITS input.

Detections are merged across tiles and detectors by grouping overlapping or
collinear segments and fusing them into a single streak-level line segment plus a
**Unified Confidence Score** that preserves a lone detector's confidence and uses
each detector's empirical precision and recall to boost corroborated detections.
ASTRiDE is treated as corroboration-only: ASTRiDE-only streak groups are kept at
a conservative display confidence, and corroborated ASTRiDE detections can only
add a small confidence boost.
Each detected streak is cross-identified against a local TLE catalog using SGP4
propagation and multi-factor confidence scoring.  Results are served through a
FastAPI backend and React frontend.
Detailed methodology and prior-work comparison: [METHODOLOGY.md](METHODOLOGY.md).

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

## Architecture

Full design: [`agent_docs/architecture.md`](agent_docs/architecture.md)

---

## End-to-End Pipeline

See [METHODOLOGY.md §§ 3–5](METHODOLOGY.md) for complete algorithmic documentation
including exact parameters, design rationale, and reproducible algorithm descriptions.

**Summary:** FITS → Z-score normalisation → tiled inference at 400 px (50 % overlap) →
ViT-S/16 heatmap probabilities stitched into full-image map → line-segment extraction
(threshold sweep, collinear stitch with growth-ratio guard) → optional ASTRiDE pass →
per-detector NMS → cross-detector grouping (rotated-IoU ≥ 0.5,
IoMin ≥ 0.3, or collinear-fragment match) → grouped-geometry fusion to outer
endpoints → ASTRiDE-only confidence lowering → Unified Confidence Score → SGP4
cross-identification → FastAPI + React canvas.

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
│   ├── argus_phases.md        ← Phases 2–8 spec
│   ├── phase1_goals.md        ← Phase 1 (complete, reference only)
│   ├── datasets.md
│   ├── dependencies.md
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
│   ├── tiled_pipeline.py      ← adaptive tiling + collinear stitch
│   ├── postprocess.py         ← Radon angle refinement + extent, NMS, grouping/fusion
│   ├── crossid.py             ← satellite cross-matching
│   ├── confidence.py          ← Unified Confidence Score (single-detector floor + F-beta corroboration)
│   └── vits_heatmap_detector.py ← DINOv3 ViT-S/16 heatmap detector (Run 15, primary)
├── training/
│   ├── convert_labels.py      ← OBB label format → COCO JSON
│   ├── dataset.py             ← FITSStreakDataset
│   ├── augmentations.py       ← albumentations + SyntheticStreakInject
│   └── train_dino.py          ← DINO training script
├── models/
│   └── dino/
│       ├── streak_codino_swin_t.py   ← Swin-T dev config
│       ├── streak_codino_swin_l.py   ← Swin-L cloud config
│       ├── streak_dinov3_vitb.py     ← DINOv3 ViT-B/16 dev config (Mac MPS)
│       ├── streak_dinov3_vitl.py     ← DINOv3 ViT-L/16 cloud config (GPU)
│       └── dinov3_adapter.py         ← PatchToPyramid adapter + MMDet backbone
├── scripts/
│   ├── README.md                 ← inventory: active/manual/provenance scripts
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

For GPU training handoff, use [agent_docs/Training_Handoff.md](agent_docs/Training_Handoff.md)
first, and complete the cloud readiness checklist in
[docs/cloud_training_preparation.md](docs/cloud_training_preparation.md) before
renting paid GPU time. On Windows hardware, that means WSL2 Ubuntu 22.04, not
native Windows. The short local setup below is for development and API/inference
work; the ML training stack has platform-specific PyTorch/MMDetection install
steps in [agent_docs/dependencies.md](agent_docs/dependencies.md).

```bash
# Create and activate the conda environment
conda create -n satid python=3.11
conda activate satid

# API-only environment, no ML stack:
pip install -r requirements-api.txt

# Local dev/inference/training environments:
# first install platform-specific torch/mmcv/mmdet from agent_docs/dependencies.md,
# then choose the narrowest lane that matches the work:
pip install -r requirements-inference.txt   # model-serving worker
pip install -r requirements-training.txt    # training/evaluation
pip install -r requirements-dev.txt         # tests on top of training
```

All runtime configuration lives in a `.env` file in the project root.
The API loads it automatically via `python-dotenv` on startup — no `export` needed.
Create `.env` from this template (`.env` is gitignored; never commit credentials):

```bash
# .env — ARGUS local dev configuration

# Space-Track credentials (required for TLE bootstrap; free account at space-track.org)
SPACETRACK_USER=your@email.com
SPACETRACK_PASS=yourpassword

# Multi-model DINO ensemble — set to [] to use ViT-S heatmap only (recommended).
# Each entry: id, size, weights, label, dataset, norm_mode.
ARGUS_MODEL_CONFIGS=[]

# ViT-S/16 heatmap detector (Run 15 — 400 px native tiles, z-score norm).
# Tile size MUST match the native_tile_size used when building the training cache.
# Run 15 was cached at 400 px; changing this will break detections.
VITS_HEATMAP_CHECKPOINT=weights/run15_vits/best.pt
VITS_HEATMAP_NORM=zscore
VITS_HEATMAP_NATIVE_TILE_SIZE=400
VITS_HEATMAP_TILE_OVERLAP=0.5
VITS_HEATMAP_THRESHOLD=0.85

PYTORCH_ENABLE_MPS_FALLBACK=1           # required on Apple Silicon
DATABASE_URL=sqlite+aiosqlite:///./argus.db
CONFIDENCE_THRESHOLD=0.05

# Fast mode: skips cross-ID (~25× faster on CPU).
FAST_MODE=false

# ASTRiDE classical detector — disabled by default.
# ARGUS_ENABLE_ASTRIDE=1

# ARGUS_ENV controls which Space-Track endpoint is used:
#   development (default) → https://for-testing-only.space-track.org/  (no real data)
#   production            → https://www.space-track.org/
# IMPORTANT: bootstrap scripts require production to fetch real TLE data.
ARGUS_ENV=production
```

### TLE Catalog Setup

ARGUS cross-identification reads exclusively from a local `tle_catalog` database.
Inference never calls Space-Track at runtime — it uses only what is already stored.

**One-time bootstrap (new install):**

```bash
# Step 1 — Load the 2025 annual bundle (covers Apr 2024–Dec 2025).
# Skip if data/tle_zips/data/exports/tle2025.txt is already loaded:
python scripts/bootstrap_tle_catalog.py \
    --zip data/tle_zips/data/exports/tle2025.txt

# Step 2 — Bootstrap the last 90 days from Space-Track GP_History (~4.5 min).
# Requires ARGUS_ENV=production in .env (the test site returns no data).
python scripts/bootstrap_recent_tles.py
```

Each day is fetched once and permanently cached; re-running is safe and idempotent.

**Daily keep-up (run once per day after 00:00 UTC):**

The same script fetches only the previous day's TLEs; all earlier days are
already cached and skipped automatically.

```bash
# Install the cron job — runs at 00:16 UTC daily (off-hour as required by Space-Track).
# Crontab requires a single line; no backslash continuations.
echo '16 0 * * * SPACETRACK_USER=your@email.com SPACETRACK_PASS=yourpassword ARGUS_ENV=production /Users/robert/miniconda3/envs/satid/bin/python /Users/robert/Argus/scripts/bootstrap_recent_tles.py >> /Users/robert/Argus/logs/tle_keepup.log 2>&1' | crontab -

# Verify it was installed:
crontab -l
```

> **Scheduling rule (Space-Track requirement):** Never schedule at :00 or :30
> past the hour.  Use :16 or :44.  The script warns if run at a busy time.

**To remove the cron job later:**
```bash
crontab -r          # removes all cron jobs (use if this is your only one)
# — or —
crontab -e          # opens vim; delete the line, then :wq to save
crontab -l          # verify it's gone
```

**Detecting and filling coverage gaps:**

Coverage is tracked per-day in `tle_catalog_coverage`.  Days that were queried
while `ARGUS_ENV=development` was active (the test site) will have
`record_count=0` even though Space-Track has real data for those dates.

```bash
# Find all zero-record gap days and re-fetch them from the production site:
python scripts/fill_tle_gaps.py
```

`fill_tle_gaps.py` queries `tle_catalog_coverage` for days with `record_count=0`,
forces a re-fetch of each from the production Space-Track API, and updates the
coverage record.  It reads credentials and `ARGUS_ENV=production` from `.env`
automatically.  Safe to re-run; already-covered days with real data are untouched.

If the catalog has no coverage for an observation time window, ARGUS leaves the
object unidentified (`unknown`) rather than querying Space-Track at runtime.

## Running Locally (Dev)

Run the API directly with the satid conda environment. The satid env has torch,
mmdet, and all ML packages installed.

### Detector inventory

ARGUS runs detectors in parallel. Each is activated as described:

| Detector | ID | How it activates | Weight path |
|---|---|---|---|
| DINOv3 ViT-S/16 HeatMap (Run 15) | `vits_heatmap` | always (primary ML detector) | `weights/run15_vits/best.pt` |
| ASTRiDE classical | `astride` | **opt-in**: `ARGUS_ENABLE_ASTRIDE=1` | — |
| ViT-S OBB Run 5 | `dinov3_vits_run5` | `ARGUS_MODEL_CONFIGS` only | `weights/run5_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth` |

The `vits_heatmap` detector is the primary production path.  It was trained on
Atwood Observatory FITS data (Runs 5–15, 400 px tile size, zscore normalisation)
and achieves F1 = 21.9 % (long-streak recall = 20.9 %, medium-streak recall = 8.3 %)
at threshold = 0.85 on the held-out Atwood validation set.  Using the
line-segment geometry metric (angle < 5°, perpendicular offset < 5 px,
length-IoU > 0.5), recall rises to 84.7 % before the collinear stitch step.

### Starting the dev servers

All configuration is read from `.env` — no environment variables need to be exported manually.

```bash
conda activate satid

# Start the API (port 8000) — reads .env automatically
uvicorn api.main:app --reload --port 8000

# In a second terminal — start the frontend dev server (port 5173)
cd frontend && npm run dev
```

To switch between single-model and full-ensemble mode, edit `ARGUS_MODEL_CONFIGS` in `.env`.
Remove the variable entirely to fall back to the single model set by `MODEL_SIZE`.

Open `http://localhost:5173`, upload a FITS file.  In the results canvas:
- **Cyan** lines = ViT-S heatmap detections (`vits_heatmap`)
- **Amber** lines = ASTRiDE classical detections
- Toggle the **Heatmap** overlay button to view the raw probability sidecar

Use the **Filters** panel to slide confidence thresholds per-method and isolate
each detector's output independently.

### Verifying which detectors are active

```bash
# Prints status for every detector without loading any model weights
python -c "
from inference.pipeline import get_detector_statuses
for d in get_detector_statuses():
    print(f\"{d['status']:12} {d['id']:30} {d['name']}\")
"
# status is one of: active | no_weights | unavailable
# 'no_weights' → the detector is implemented but the weight file is missing
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

The active production model is a ViT-S/16 heatmap trained on Atwood FITS data.
Training requires pre-built NPY tile caches.  Full training pipeline:

```bash
# 1. Build tiled annotation file (Atwood nights, 400px tiles, 50% overlap)
python scripts/build_tiled_brentimages_json.py \
  --src data/annotations/all_train_run5.json \
  --out /Volumes/External/TrainingData/annotations/atwood_train_tiled.json \
  --native-tile-size 400 --overlap 0.5 \
  --neg-tiles-per-image 50 --hard-neg-per-pos 5

# 2. Cache DINOv3 features → NPY (reads FITS, writes .npy files)
python scripts/cache_dinov3_heatmap_features.py \
  --annotations /Volumes/External/TrainingData/annotations/atwood_train_tiled.json \
  --output-dir /tmp/argus_run_cache \
  --backbone vits --image-size 400

# 3. Train ViT-S heatmap (40 epochs, cosine LR)
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python training/train_dinov3_heatmap_cached.py \
  --annotations /tmp/argus_run_cache/train.json \
  --val-annotations /tmp/argus_run_cache/val.json \
  --output-dir weights/run_local_vits \
  --backbone vits --epochs 40 \
  --pos-weight 20 --neg-ratio 0.38 \
  --lr 5e-4 --cosine-lr

# 4. Evaluate
python scripts/evaluate_dinov3_heatmap.py \
  --weights weights/run_local_vits/best.pt \
  --annotations data/annotations/val_atwood.json \
  --backbone vits --native-tile-size 400 --tile-overlap 0.5 \
  --threshold 0.85 --stitch --output results/run_local_vits/
```

See `docs/training_methods.md §3` for the full run history (Runs 1–15) and
`docs/training_methods.md §6` for evaluation policy.

### Historical OBB training (MMDetection DINO)

The OBB DINO-DETR path (ViT-S/B, Swin-T/L) is retained for reference.
To reproduce the Swin-T baseline:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 MODEL_SIZE=tiny \
  python -m training.train_dino --smoke-test
```

## Updating Detector Profiles After Training

Every time a detector is retrained or a new model is evaluated, update its entry
in `DETECTOR_PROFILES` inside [`inference/confidence.py`](inference/confidence.py).
The Unified Confidence Score preserves the best non-ASTRiDE detector confidence
as the score floor, then weights corroborating detector boosts by their F-0.5
score (`w = 1.25 × P × R / (0.25 × P + R)`). Stale values silently under- or
over-weight a detector's corroborating evidence. ASTRiDE is a special case: it
is corroboration-only and should not be tuned into a standalone confidence source.

### Step 1 — Run the benchmark to get per-method P/R

```bash
MODEL_WEIGHTS=weights/<run>/<best>.pth MODEL_SIZE=<size> USE_DEV_SUBSET=false \
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/test.json \
    --output results/<run>/phase8_benchmark.json
```

The output JSON contains `"precision"` and `"recall"` fields for the evaluated method.

### Step 2 — Edit `DETECTOR_PROFILES` in `inference/confidence.py`

Find the entry for the detector key (e.g. `"dinov3_vitb"`, `"tiny"`, `"astride"`) and
update `precision`, `recall`, and `notes`:

```python
"vits_heatmap": DetectorProfile(
    name="DINOv3 ViT-S HeatMap",
    precision=0.228,     # ← measured at t=0.85, val_atwood, Run 15
    recall=0.211,        # ← measured at t=0.85, val_atwood, Run 15
    notes="Run 15 results/run15_vits/threshold_sweep/threshold_sweep.json",
),
```

Detector keys must match the `method` string written by the inference pipeline.
Check `api/main.py` or the benchmark output for exact key names.

**Confidence ceiling** — if a new non-ASTRiDE detector is known to emit unreliably
high scores on false positives (its confidence magnitude is miscalibrated), set
`confidence_ceiling` to cap its effective contribution. Leave
`confidence_ceiling=None` for ML detectors with well-calibrated outputs. ASTRiDE
keeps its profile entry for diagnostics, but the pipeline lowers ASTRiDE-only
groups to conservative display confidence and the scorer excludes ASTRiDE from
non-ASTRiDE corroboration and divergence. When ASTRiDE corroborates another
detector, it can only add a small bounded boost; for example, an ML detector at
0.86 plus ASTRiDE 0.99 scores about 0.90.

### Step 3 — Verify

```bash
python -m inference.confidence      # prints example scores with updated weights
python -m pytest tests/test_confidence.py -v
```

The test `test_registered_profiles_have_valid_weights` will catch any weight
outside [0, 1].  Single-detector scores keep their detector confidence; review
multi-detector examples to confirm the corroboration boosts look sensible.

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

## Hardware

| Machine | Use | Config |
|---------|-----|--------|
| MacBook Air M3 (16 GB) | Development, API/inference | MPS, `VITS_HEATMAP_CHECKPOINT=weights/run15_vits/best.pt` |
| RTX 5070 Ti 16 GB (workstation) | ViT-S backbone unfreeze training (next step) | CUDA, partial unfreeze last 2–4 ViT blocks |
| Vast.ai / RunPod A10 (~$0.15/hr) | Cloud GPU for unfreeze or ViT-B experiments | CUDA |
| Lambda Labs A100 40 GB | ViT-L backbone unfreeze (if needed) | CUDA |

Never hardcode `torch.device("cuda")` — always use `get_device()` from
`inference/device.py`.
