# ARGUS / StreakMind — Satellite Streak Detection & Identification Pipeline

## What This Is
An end-to-end pipeline that detects satellite streaks in FITS telescope images
using a Co-DINO transformer model (Swin-L backbone), refines streak angle via
the Radon transform, and cross-identifies detected objects against TLE orbital
data from Space-Track's GP_History API using SGP4 propagation and multi-factor
confidence scoring. Results are served through a FastAPI backend and React frontend.

## Current Phase
**PHASE 2 — Co-DINO model** (next to build).

Progress:
- ✅ Phase 0 (Classical baseline): `src/` — fits_parser, classical_detector, plate_solver, SGP4 matching
- ✅ Phase 1 (Data pipeline): `inference/fits_loader.py`, `training/convert_labels.py`, `training/dataset.py`, `training/augmentations.py`
- ⬜ Phase 2 (Co-DINO model): `inference/device.py`, MMDetection configs, training script
- ⬜ Phase 3 (Cross-identification): `inference/crossid.py`, `inference/postprocess.py`
- ⬜ Phase 4 (Database): `db/schema.sql`, SQLAlchemy async models
- ⬜ Phase 5 (API): `api/` — FastAPI upload/result/image endpoints
- ⬜ Phase 6 (Frontend): `frontend/` — React + Vite + Tailwind, canvas OBB rendering
- ⬜ Phase 7 (Docker): `docker/` — docker-compose with GPU worker
- ⬜ Phase 8 (Evaluation): `eval/` — mAP, angle error, DINO vs YOLO benchmark

## Hardware
- **Dev / CI:** MacBook Air M3 — CPU or MPS. Use `MODEL_SIZE=tiny` (Swin-T).
- **Training:** Lambda Labs A100 40 GB — CUDA. Use `MODEL_SIZE=large` (Swin-L).
- **Rule:** Never hardcode `torch.device("cuda")`. Always call `get_device()` from
  `inference/device.py`. Code must run on CPU, MPS, and CUDA without changes.

## Read These Before Starting Any Task
Always read the relevant agent_docs file before writing code:

- `agent_docs/architecture.md`       — full system design, component map, data flow
- `agent_docs/phase1_goals.md`       — Phase 1 data pipeline (complete — reference only)
- `agent_docs/streakmind_phases.md`  — Phases 2–8: model through eval
- `agent_docs/datasets.md`           — where to get test data, download links
- `agent_docs/dependencies.md`       — exact packages, versions, install commands
- `agent_docs/service_roadmap.md`    — Docker, deployment, cloud scale path
- `agent_docs/test_strategy.md`      — how to measure and record baseline accuracy
- `agent_docs/spacetrack.md`         — Space-Track API usage, rate limits, caching rules

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
├── CLAUDE.md
├── README.md
├── agent_docs/              ← read before coding
├── src/                     ← Phase 0: classical baseline (complete, do not modify)
│   ├── ingest/fits_parser.py
│   ├── detection/classical_detector.py
│   ├── astrometry/plate_solver.py
│   └── matching/            ← scorer, spacetrack_query, spatial_filter, propagator, matcher
├── inference/               ← ML inference modules
│   ├── fits_loader.py       ← FITS→tensor, Z-score normalisation (Phase 1 ✅)
│   ├── device.py            ← get_device() helper — CPU/MPS/CUDA (Phase 2, next)
│   ├── pipeline.py          ← main inference orchestrator (Phase 2)
│   ├── postprocess.py       ← Radon angle refinement + NMS (Phase 3)
│   └── crossid.py           ← satellite ephemeris cross-matching (Phase 3)
├── training/                ← training data and model training
│   ├── convert_labels.py    ← OBB YOLO labels → COCO JSON (Phase 1 ✅)
│   ├── dataset.py           ← FITSStreakDataset (Phase 1 ✅)
│   ├── augmentations.py     ← albumentations pipeline + SyntheticStreakInject (Phase 1 ✅)
│   ├── train_dino.py        ← Co-DINO training script (Phase 2)
│   └── train_baseline.py    ← YOLO11-OBB training script (Phase 2)
├── models/
│   ├── dino/                ← MMDetection configs: streak_codino_swin_t.py, _swin_l.py
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
│   └── annotations/         ← COCO-format JSON label files
├── tests/                   ← pytest (mirrors src/ and top-level module layout)
├── results/                 ← baseline metrics JSON output
└── weights/                 ← model weights (gitignored)
```

## Academic Research Context
This project is academic research software. It builds on the following prior works:

- **ASTRiDE** — Automated Streak Detection for Astronomical Images
  (Kim et al., https://github.com/dwkim78/ASTRiDE)
- **StreakMind** — Co-DINO transformer-based satellite streak detection
  (StreakMind project, cite per their published paper/repo)
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
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
export DATABASE_URL=sqlite+aiosqlite:///./argus.db   # default
export MODEL_SIZE=tiny     # tiny=Swin-T (dev/MPS), large=Swin-L (A100)
# Optional for cloud deployment:
export STORAGE_BACKEND=local   # or s3
export QUEUE_BACKEND=memory    # or sqs
export S3_BUCKET=
export AWS_REGION=
```

## Running Tests
```bash
conda activate satid
pytest tests/ -v
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
