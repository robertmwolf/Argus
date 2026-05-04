# ARGUS
### Automated Recognition and Grading of Unidentified Streaks

ARGUS is an academic research pipeline for automated satellite streak detection
and identification in FITS telescope images.  It detects satellite streaks using
a DINO transformer model (Swin backbone), refines streak orientation via the
Radon transform, and cross-identifies detected objects against historical
Two-Line Element (TLE) orbital data from the US Space Force Space-Track catalog
using SGP4 propagation and multi-factor confidence scoring.  Results are served
through a FastAPI backend and React frontend.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 — Classical baseline | ASTRiDE detection, SGP4 matching | ✅ Complete |
| 1 — Data pipeline | FITS loader, COCO labels, augmentations | ✅ Complete |
| 2 — DINO model | device.py, MMDet configs, train script | ✅ Complete |
| 3 — Cross-identification | inference pipeline, Radon, crossid | ⬜ Next |
| 4 — Database | SQLAlchemy schema, async models | ⬜ Planned |
| 5 — API | FastAPI upload / result endpoints | ⬜ Planned |
| 6 — Frontend | React + Vite, canvas OBB rendering | ⬜ Planned |
| 7 — Docker | docker-compose with GPU worker | ⬜ Planned |
| 8 — Evaluation | mAP, angle error, DINO vs ASTRiDE | ⬜ Planned |

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
    ├─ Phase 0/3: Classical path
    │   ├── src/ingest/fits_parser.py       — FITS → FITSImage dataclass
    │   ├── src/detection/classical_detector.py  — ASTRiDE streak extraction
    │   ├── src/astrometry/plate_solver.py  — pixel → RA/Dec (WCS)
    │   └── src/matching/                   — Space-Track query → SGP4 → scorer → matcher
    │
    └─ Phase 2/3: ML path
        ├── inference/fits_loader.py        — FITS → normalised tensor
        ├── inference/device.py             — get_device() / get_device_config()
        ├── models/dino/                    — DINO Swin-T (dev) / Swin-L (cloud) configs
        ├── training/train_dino.py          — two-stage fine-tuning + cost guardrails
        ├── inference/pipeline.py           — end-to-end orchestrator  [Phase 3]
        ├── inference/postprocess.py        — Radon angle refinement   [Phase 3]
        └── inference/crossid.py            — TLE cross-identification  [Phase 3]
```

Full design: [`agent_docs/architecture.md`](agent_docs/architecture.md)

---

## Project Structure

```
Argus/
├── CLAUDE.md                  ← agent coding instructions
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
│   └── matching/              ← scorer, propagator, spatial_filter, matcher, spacetrack_query
├── inference/                 ← ML inference modules
│   ├── device.py              ← get_device() / get_device_config()  ✅
│   ├── fits_loader.py         ← FITS → tensor, Z-score normalisation  ✅
│   ├── pipeline.py            ← inference orchestrator  [Phase 3]
│   ├── postprocess.py         ← Radon angle refinement  [Phase 3]
│   └── crossid.py             ← satellite cross-matching  [Phase 3]
├── training/
│   ├── convert_labels.py      ← YOLO OBB → COCO JSON  ✅
│   ├── dataset.py             ← FITSStreakDataset  ✅
│   ├── augmentations.py       ← albumentations + SyntheticStreakInject  ✅
│   ├── train_dino.py          ← DINO training script  ✅
│   └── train_baseline.py      ← YOLO11-OBB baseline  [Phase 2]
├── models/
│   └── dino/
│       ├── streak_codino_swin_t.py   ← Swin-T dev config  ✅
│       └── streak_codino_swin_l.py   ← Swin-L cloud config  ✅
├── scripts/
│   ├── make_test_fits.py      ← synthetic FITS generator  ✅
│   ├── download_weights.py    ← pretrained weight downloader  ✅
│   └── prepare_cloud_training.py  ← go/no-go checklist  [Phase 6]
├── api/                       ← FastAPI application  [Phase 5]
├── frontend/                  ← React + Vite  [Phase 6]
├── eval/                      ← metrics, benchmark  [Phase 8]
├── db/                        ← schema.sql, migrations  [Phase 4]
├── docker/                    ← docker-compose  [Phase 7]
├── tests/                     ← 228 tests, all passing
├── data/
│   ├── sample/                ← synthetic FITS for smoke-testing  ✅
│   ├── raw/                   ← MILAN FITS (gitignored, download separately)
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

# Set Space-Track credentials (required for matching)
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword

# Set model size (tiny=Mac dev, large=cloud A100)
export MODEL_SIZE=tiny
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Running Tests

```bash
conda activate satid
pytest tests/ -v
# 228 passed, 1 skipped (CUDA test, skipped on Mac)
```

## Generating Synthetic Test Data

No real FITS files are required to develop and test the pipeline:

```bash
python scripts/make_test_fits.py --small   # fast 512×512 images
python scripts/make_test_fits.py           # full 3096×2080 MILAN resolution
```

## Downloading Pretrained Weights

```bash
# Swin-T weights for local development (~160 MB):
MODEL_SIZE=tiny python scripts/download_weights.py

# Swin-L weights for cloud training (~828 MB, A100 only):
MODEL_SIZE=large python scripts/download_weights.py
```

## Training

```bash
# Local Mac dev (Swin-T, 50-image dev subset):
MODEL_SIZE=tiny python -m training.train_dino

# Verify cloud setup before full run:
MODEL_SIZE=large python -m training.train_dino --smoke-test

# Full cloud training (A100):
MODEL_SIZE=large python -m training.train_dino --work-dir weights/run_001
```

## Real Training Data

The model requires annotated FITS images.  Two sources:

| Dataset | Images | Format | Access |
|---------|--------|--------|--------|
| **SatStreaks** | 3,073 annotated | PNG + YOLO OBB labels | [GitHub](https://github.com/jijup/SatStreaks) — free |
| **MILAN Sky Survey** | 50,068 raw FITS | FITS (needs annotation) | [Zenodo](https://zenodo.org/records/7049839) — free |

```bash
# Download one month of MILAN (2–5 GB):
pip install zenodo_get
zenodo_get 7049839 -o data/raw/milan_2022-08/
```

---

## Hardware

| Machine | Use | Config |
|---------|-----|--------|
| MacBook Air M3 (16 GB) | Development, testing | `MODEL_SIZE=tiny`, MPS |
| Lambda Labs A100 40 GB | Training | `MODEL_SIZE=large`, CUDA |

Never hardcode `torch.device("cuda")` — always use `get_device()` from
`inference/device.py`.
