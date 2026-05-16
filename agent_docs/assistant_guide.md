# ARGUS Assistant Guide

This is the canonical instruction file for coding assistants working on ARGUS.
Both Codex (`AGENTS.md`) and Claude Code (`CLAUDE.md`) should read this file.
Keep durable assistant instructions here; keep deeper reference material in the
topic-specific files under `agent_docs/`.

## ARGUS — Satellite Streak Detection & Identification Pipeline

## What This Is
An end-to-end pipeline that detects satellite streaks in FITS telescope images
using a Co-DINO transformer model (Swin-L backbone), refines streak angle via
the Radon transform, and cross-identifies detected objects against a local TLE
catalog (sourced once from Space-Track, stored in the ARGUS database) using SGP4
propagation and multi-factor confidence scoring. Results are served through a
FastAPI backend and React frontend.

## Current Phase
**ALL IMPLEMENTATION PHASES COMPLETE. DINOv3 ViT-B backbone (Phase C²) complete — awaiting Phase D (ViT-L workstation).**

Progress:
- ✅ Phase 0 (Classical baseline): `src/` — fits_parser, classical_detector, plate_solver, SGP4 matching
- ✅ Phase 1 (Data pipeline): `inference/fits_loader.py`, `training/convert_labels.py`, `training/dataset.py`, `training/augmentations.py`
- ✅ Phase 2 (DINO model): `inference/device.py`, `models/dino/` configs, `training/train_dino.py`, `scripts/download_weights.py`, `scripts/make_test_fits.py`
- ✅ Phase 3 (Inference pipeline): `inference/pipeline.py`, `inference/postprocess.py`, `inference/crossid.py`
- ✅ Phase 4 (Database): `db/schema.sql`, `db/models.py` — SQLAlchemy async ORM (SQLite + PostgreSQL)
- ✅ Phase 5 (API): `api/main.py`, `api/storage.py`, `api/queue.py`, `api/worker.py` — FastAPI + background worker
- ✅ Phase 6 (Frontend): `frontend/` — React 18 + Vite + Tailwind, canvas OBB rendering, detection table
- ✅ Phase 7 (Docker): `docker/`, `docker-compose.yml`, `docker-compose.cloud.yml` — full stack verified
- ✅ Phase 8 (Evaluation): `eval/metrics.py`, `eval/benchmark.py` — mAP, angle error, per-band, DINO vs YOLO
- ✅ Local training: Swin-T DINO (50 epochs, CPU) + YOLO11-OBB baseline — results in `results/phase8_benchmark.json`
- ✅ DINOv3 Phase A (feasibility probe): cosine dissimilarity = 0.095 — PASS
- ✅ DINOv3 Phase B (adapter + configs): `models/dino/dinov3_adapter.py`, `streak_dinov3_vitb.py`, `streak_dinov3_vitl.py` — smoke test PASS
- ✅ DINOv3 Phase C (frozen ViT-B dev subset): mAP@0.5=0.274 on dev_subset
- ✅ DINOv3 Phase C² (frozen ViT-B full dataset, 4 epochs): mAP@0.5=**0.74** on test.json — beats Swin-T (0.19) by +0.55
- ✅ DINOv3 Phase E (partial): Swin-T vs ViT-B comparison in `results/phase_e/` — ViT-B dominant
- ✅ YOLO11n-OBB full dataset: 15 epochs, 3 023 images → 14 385 tiles, mAP@0.5=67.3% P=57% R=85% F1=68% (tiled val); integrated as `yolo_full` 5th detector in `inference/pipeline.py`
- ⏳ DINOv3 Phase D: Frozen ViT-L, full dataset — RTX 5070 Ti workstation, see `agent_docs/Training_Handoff.md`

## Next Step: Phase D — DINOv3 ViT-L Workstation Training
Phase C² result: DINOv3 ViT-B frozen, full merged dataset, 4 epochs → mAP@0.5=0.74 (test.json).
This beats Swin-T (0.19) by a wide margin. Phase D trains ViT-L on the same data for the definitive result.
Phase 8 targets (≥94% precision, ≥97% recall) are expected to be met with ViT-L on the full dataset.
Follow `agent_docs/Training_Handoff.md` for the RTX 5070 Ti workstation handoff procedure.

## Hardware
- **Dev / CI:** MacBook Air M3 — CPU or MPS. Use `MODEL_SIZE=tiny` (Swin-T).
- **Training:** Lambda Labs A100 40 GB — CUDA. Use `MODEL_SIZE=large` (Swin-L).
- **Rule:** Never hardcode `torch.device("cuda")`. Always call `get_device()` from
  `inference/device.py`. Code must run on CPU, MPS, and CUDA without changes.

## Read These Before Starting Any Task
Always read the relevant agent_docs file before writing code:

- `agent_docs/architecture.md`       — full system design, component map, data flow
- `agent_docs/phase1_goals.md`       — Phase 1 data pipeline (complete — reference only)
- `agent_docs/streakmind_phases.md`  — ARGUS Phases 2–8: model through eval
- `agent_docs/dinov3_plan.md`        — DINOv3 backbone integration plan and phase status
- `agent_docs/datasets.md`           — where to get test data, download links
- `agent_docs/dependencies.md`       — exact packages, versions, install commands
- `agent_docs/service_roadmap.md`    — Docker, deployment, cloud scale path
- `agent_docs/test_strategy.md`      — how to measure and record baseline accuracy
- `agent_docs/spacetrack.md`         — Space-Track API policy, TLE catalog setup, rate limits
- `agent_docs/Training_Handoff.md`   — current RTX 5070 Ti DINOv3 ViT-L training handoff

## Stack
- Python 3.11, conda environment named `satid`
  (use `/Users/robert/miniconda3/envs/satid/bin/python`)
- Core ML: PyTorch ≥ 2.2, MMDetection ≥ 3.3 (Co-DINO), Ultralytics (YOLO baseline)
- Astronomy: astropy, astride, sgp4, skyfield, spacetrack
- Image: opencv-python-headless<4.10, scikit-image (Radon), albumentations, Shapely
- API: FastAPI, SQLAlchemy async, asyncpg/aiosqlite, Pydantic v2
- Frontend: React 18 + Vite + Tailwind CSS
- Testing: pytest
- numpy must stay < 2.0 (sep and astride are compiled against numpy 1.x)

## Project Structure
```
Argus/
├── AGENTS.md                ← Codex compatibility shim; points here
├── CLAUDE.md                ← Claude Code compatibility shim; points here
├── README.md
├── agent_docs/              ← canonical assistant guide + reference docs
├── src/                     ← Phase 0: classical baseline (complete, do not modify)
│   ├── ingest/fits_parser.py
│   ├── detection/classical_detector.py
│   ├── astrometry/plate_solver.py
│   └── matching/            ← scorer, spacetrack_query, tle_store, spatial_filter, propagator, matcher
├── inference/               ← ML inference modules
│   ├── fits_loader.py       ← FITS→tensor, normalisation + FITS/sidecar WCS (Phase 1 ✅)
│   ├── device.py            ← get_device() helper — CPU/MPS/CUDA (Phase 2, next)
│   ├── pipeline.py          ← main inference orchestrator (Phase 2)
│   ├── postprocess.py       ← Radon angle refinement + NMS (Phase 3)
│   └── crossid.py           ← satellite ephemeris cross-matching (Phase 3)
├── training/                ← training data and model training
│   ├── convert_labels.py    ← OBB YOLO labels → COCO JSON (Phase 1 ✅)
│   ├── dataset.py           ← FITSStreakDataset (Phase 1 ✅)
│   ├── augmentations.py     ← albumentations pipeline + SyntheticStreakInject (Phase 1 ✅)
│   ├── train_dino.py        ← Co-DINO training script + checkpoint/timebox CLI overrides (Phase 2)
│   └── train_baseline.py    ← YOLO11-OBB training script (Phase 2)
├── models/
│   ├── dino/                ← MMDetection configs: streak_codino_swin_t.py, _swin_l.py,
│   │                           streak_dinov3_vitb.py, streak_dinov3_vitl.py,
│   │                           dinov3_adapter.py (PatchToPyramid + DINOv3Backbone)
│   └── baselines/           ← YOLO11-OBB config
├── api/                     ← FastAPI application (Phase 5)
│   ├── main.py
│   ├── models.py
│   ├── storage.py           ← local / S3 swappable backend
│   └── queue.py             ← in-memory / SQS swappable backend
├── frontend/                ← React + Vite (Phase 6)
├── eval/                    ← metrics, benchmark, visualise (Phase 8)
├── db/                      ← schema.sql, migrations (Phase 4)
├── docker/                  ← Dockerfiles + docker-compose (Phase 7)
├── data/
│   ├── raw/                 ← original FITS files (gitignored)
│   ├── processed/           ← converted PNGs (gitignored)
│   ├── catalogs/            ← TLE catalog files
│   ├── tle_zips/            ← Space-Track annual TLE bundles (gitignored, one-time setup)
│   └── annotations/         ← COCO-format JSON label files
├── tests/                   ← pytest (mirrors src/ and top-level module layout)
├── results/                 ← baseline metrics JSON output
└── weights/                 ← model weights (gitignored)
```

## Academic Research Context
This project is academic research software. It builds on the following prior works:

- **ASTRiDE** — Automated Streak Detection for Astronomical Images
  (Kim et al., https://github.com/dwkim78/ASTRiDE)
- **StreakMind** — Co-DINO transformer-based satellite streak detection pipeline
  (prior work that ARGUS builds upon; cite per their published paper/repo)
- **Co-DINO** — Co-Deformable DETR object detection
  (Zong et al., 2023, https://arxiv.org/abs/2211.12860)
- **Danarianto et al. Prototype** — satellite identification prototype pipeline
  (Danarianto et al., cite per their published paper)

### Source Citation Rules
Whenever code directly implements, adapts, or is substantially derived from one
of the above works, add an inline citation comment at the function, class, or
code block level. Use this format:

```python
# Source: <AuthorOrProject> — <brief description of what was adapted>
# Ref: <DOI, URL, or "unpublished manuscript" as appropriate>
```

Apply citations to:
- Any algorithm, formula, or threshold copied or adapted from a prior work
- Any preprocessing step, model architecture, or scoring logic derived from a source
- Helper functions whose design is traceable to a specific paper or repo

Do **not** cite:
- Standard library usage or generic Python idioms
- astropy/sgp4/skyfield API calls that follow their own documentation
- Logic that is entirely original to this project

When in doubt, cite. Over-attribution is preferable to under-attribution in
academic research code.

## Code Standards
- Type hints on every function signature
- Google-style docstrings on every public function and class
- Every module has a `if __name__ == "__main__":` block for standalone testing
- Never hardcode credentials — use environment variables only
- Never hardcode `torch.device("cuda")` — use `get_device()` from `inference/device.py`
- All file paths via `pathlib.Path`, never raw strings
- Log with `logging` module, not `print()` (except __main__ blocks)
- Log all inference timings at DEBUG level:
  `fits_load_ms`, `inference_ms`, `postprocess_ms`, `crossid_ms`, `db_write_ms`

## Environment Variables Required
```bash
export DATABASE_URL=sqlite+aiosqlite:///./argus.db   # default
export MODEL_SIZE=tiny     # tiny=Swin-T (dev/MPS), large=Swin-L (A100),
                           # dinov3_vitb=ViT-B/16 (Mac MPS), dinov3_vitl=ViT-L/16 (GPU)
export ARGUS_NORM=zscore   # zscore (Swin-T/L) or autostretch (DINOv3 ViT-B/L)
# Optional for cloud deployment:
export STORAGE_BACKEND=local   # or s3
export QUEUE_BACKEND=memory    # or sqs
export S3_BUCKET=
export AWS_REGION=

# Space-Track credentials — only needed for explicit catalog maintenance or
# diagnostics. Day-to-day inference never queries Space-Track directly.
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
# Dev/local maintenance calls use Space-Track's test site by default.
# Production uses the official site when ARGUS_ENV=production.
export ARGUS_ENV=development
export SPACETRACK_BASE_URL=https://for-testing-only.space-track.org/
```

## TLE Catalog Setup (one-time per environment)

The cross-identification pipeline reads TLEs from a local database table
(`tle_catalog` in `argus.db`) rather than querying Space-Track at inference
time.  This must be bootstrapped once per environment.

```bash
# 1. Download annual TLE bundle(s) from Space-Track's cloud storage:
#    https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu
#    Place file(s) in data/tle_zips/  (any .zip or .txt format is accepted)

# 2. Load into the database (idempotent — safe to re-run):
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2025
```

If `tle_catalog` does not contain records for an observation's time window,
ARGUS skips cross-identification and leaves the object unknown. It does not
fall back to broad `gp_history` or hourly `gp` calls during inference.

See `agent_docs/spacetrack.md` for full API policy details.

## Running Tests
```bash
conda activate satid
pytest tests/ -v
# All tests are offline (mocked) — no Space-Track credentials required
```

## Workflow Rules
- Complete and test each phase before starting the next
- Phase 1 gate is already cleared — valid COCO JSON ✅, FITSStreakDataset iterates ✅
- Write pytest tests alongside each module, not after
- Run pytest after every module is complete — fix failures before continuing
- Ask for a plan before writing code for any module over 100 lines
- Storage and queue backends must be swappable via env var with zero changes to
  `api/main.py` or `inference/pipeline.py`
- numpy must stay pinned < 2.0; do not upgrade albumentations or opencv past versions
  that require numpy ≥ 2.0
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in your shell when running on Mac

## Device Abstraction (`inference/device.py`)

`inference/device.py` must exist before any other ML code. It provides:

```python
def get_device() -> torch.device:
    """Priority: CUDA → MPS → CPU. Never hardcode 'cuda' anywhere."""

def get_device_config() -> dict:
    """Returns device-appropriate hyperparameters."""
```

Config values by device:

| Key | CUDA | MPS | CPU |
|-----|------|-----|-----|
| `batch_size` | 2 | 1 | 1 |
| `num_workers` | 4 | 0 | 2 |
| `pin_memory` | True | False | False |
| `image_size` | 800 | 400 | 400 |
| `mixed_precision` | True | False | False |
| `gradient_checkpointing` | True | True | True |

**MPS-specific rules** (apply everywhere):
1. `num_workers` must be 0 — always use `get_device_config()["num_workers"]`
2. Wrap AMP: `use_amp = device.type == "cuda"` then `torch.autocast(device_type=device.type, enabled=use_amp)`
3. NMS/roi_align fallback: if `device.type == "mps"`, call `.cpu()` before torchvision op then `.to(device)` after
4. `pin_memory=True` crashes on MPS — always use `get_device_config()["pin_memory"]`
5. `skimage.transform.radon` and Shapely are CPU-only — expected, do not move to MPS

## Model Size Selection

Two MMDetection configs must always exist:
- `models/dino/streak_codino_swin_t.py` — development, Mac-safe (Swin-T, ~340MB weights)
- `models/dino/streak_codino_swin_l.py` — production, cloud GPU only (Swin-L, ~2.4GB weights)

`MODEL_SIZE=large` must raise `EnvironmentError` if `device.type != "cuda"`.

`scripts/download_weights.py` — downloads Swin-T (default) or Swin-L weights based on `MODEL_SIZE`; skips if file exists. Add `weights/` to `.gitignore`.

## Dev Subset Tool (`training/make_dev_subset.py`)

50-image reproducible subset for fast local iteration:
- 20 images with no streaks, 20 with short streaks (<269px), 10 with long streaks (≥269px)
- `USE_DEV_SUBSET=true` (default) → loads `data/annotations/dev_subset.json`
- `USE_DEV_SUBSET=false` → loads full annotation file (cloud training only)

## Fast Iteration Mode

`FAST_MODE=true` or `pipeline.run(image, fast=True)`:
- Skips Radon angle refinement, cross-ID, DB write
- Forces `image_size=256`
- Target: <60 seconds wall time per image on Mac

## Phase Sequencing (Hardware-Aware)

| Phase | Where | Gate |
|-------|-------|------|
| 1 — Data pipeline | Mac CPU | COCO JSON valid, Dataset iterates |
| 2 — Model config | Mac (no GPU) | Both configs pass mmdet check; Swin-T weights downloaded |
| 3 — Augmentation | Mac CPU | `augmentations.py --visualize` runs clean |
| 4 — Integration | Mac MPS, tiny | `pipeline.py --fast` <60s |
| 5 — API + Frontend | Mac CPU | docker-compose up, upload curl works |
| 6 — Cloud handoff | Mac | `prepare_cloud_training.py` all checks pass |
| 7 — Cloud training | Lambda A100 | val mAP >90%, fetch weights |
| 8 — Evaluation | Mac MPS | ≥94% precision, ≥97% recall |
| DINOv3 A — Probe | Mac CPU | Cosine dissimilarity > 0.05 ✅ (0.095) |
| DINOv3 B — Adapter | Mac | MMDet configs parse, pipeline smoke test ✅ |
| DINOv3 C² — ViT-B full | Mac MPS | mAP@0.5 > Swin-T baseline ✅ (0.74 vs 0.19) |
| DINOv3 D — ViT-L full | RTX 5070 Ti | mAP@0.5 ≥ 0.70, see Training_Handoff.md ⏳ |
| DINOv3 E — Comparison | Mac | ViT-L vs ViT-B vs Swin-T table |

## DINOv3 Training (model size `dinov3_vitb` / `dinov3_vitl`)

```bash
# Mac MPS — ViT-B frozen, dev subset (smoke test):
MODEL_SIZE=dinov3_vitb USE_DEV_SUBSET=true ARGUS_NORM=autostretch \
  python -m training.train_dino --backbone dinov3_vitb --work-dir weights/dinov3_vitb_dev

# Workstation — ViT-L frozen, full dataset (Phase D):
MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false ARGUS_NORM=autostretch \
  python -m training.train_dino --backbone dinov3_vitl --work-dir weights/run_5070ti_dinov3_vitl
```

DINOv3 weights are not downloaded by `download_weights.py`.
Copy from Mac (`weights/dinov3_vitb16_pretrain_lvd1689m*.pth` or `dinov3_vitl16_lvd1689m.pth`),
or see `scripts/download_dinov3_weights.py`.

`MODEL_SIZE=dinov3_vitl` raises `EnvironmentError` if `device.type != "cuda"`.

## Cloud Training Scripts

`scripts/prepare_cloud_training.py` — validates all checklist items (annotations, dataset, configs, augmentations, pipeline fast-mode, API, docker, requirements.txt pinned) before GPU rental; exits 1 on any failure.

`scripts/cloud_setup.sh` — run once on Lambda instance: installs deps, downloads Swin-L weights, verifies CUDA, flips `.env` to `MODEL_SIZE=large` and `USE_DEV_SUBSET=false`.

`scripts/fetch_weights.sh <user@ip>` — rsync `weights/best.pth` and training logs back to Mac after training.

## Cost Guardrails (`training/train_dino.py`)

After epoch 1 completes, print estimated total time and Lambda cost ($1.29/hr), then `sleep(30)` before epoch 2. Ctrl+C during that window aborts the run without further charges.

## Deferred Work (stub with `raise NotImplementedError`)

Do not implement until Phase 7 weights exist:
- Multi-frame tracklet association (DB schema only)
- Swin-L → Swin-T weight distillation
