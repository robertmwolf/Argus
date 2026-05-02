# ARGUS
### Automated Recognition and Grading of Unidentified Streaks

ARGUS is an academic research pipeline for automated satellite identification
in FITS telescope images. It detects satellite streaks using classical image
processing techniques, then identifies the satellite by matching the observed
trajectory against historical Two-Line Element (TLE) orbital data from the
US Space Force Space-Track catalog, using SGP4 propagation and multi-factor
confidence scoring.

---

## Name

**ARGUS** stands for **Automated Recognition and Grading of Unidentified Streaks**.

The name also references **Argus Panoptes** (Ἄργος Πανόπτης), the many-eyed
giant of Greek mythology — son of Gaia, set by Hera to watch over Io with his
hundred eyes, each pair taking turns to sleep so that he was always vigilant.
Like its namesake, this system watches the sky continuously, cataloguing every
streak it finds.

---

## Research Context

This project builds on and cites the following prior works:

- **ASTRiDE** — Automated Streak Detection for Astronomical Images.
  Kim et al. *Astronomical Journal* (2017).
  https://github.com/dwkim78/ASTRiDE

- **StreakMind** — YOLO-OBB model for satellite streak detection.
  Cite per published paper/repo.

- **Danarianto et al.** — Satellite identification prototype pipeline.
  Cite per published paper.

Code that is directly derived from or substantially adapts these works is
annotated with `# Source:` and `# Ref:` comments at the relevant function,
class, or block level, in accordance with academic attribution standards.

---

## Pipeline Overview

```
FITS Image → Streak Detection (ASTRiDE) → Plate Solve (WCS)
          → Space-Track GP_History Query → SGP4 Propagation
          → Multi-Factor Matching → Ranked Candidate List
```

Full architecture: [`agent_docs/architecture.md`](agent_docs/architecture.md)

---

## Phase Roadmap

| Phase | Weeks | Detector | Status |
|-------|-------|----------|--------|
| 1 — Classical baseline | 1–4 | ASTRiDE + SGP4 weighted scoring | In progress |
| 2 — YOLO-OBB integration | 5–10 | YOLO-OBB primary + ASTRiDE validator | Planned |
| 3 — Hybrid consensus | 11–14 | Consensus layer + DINOv3 anomaly classifier | Planned |

---

## Tech Stack

The pipeline is built in phases. Phase 1 establishes a classical, deterministic baseline
before any ML is introduced. Later phases layer in neural models whose gains can be
measured against that baseline.

**Phase 1 — Classical Baseline (Weeks 1–4)** ✓ current

| Layer | Tool | Purpose |
|-------|------|---------|
| Runtime | Python 3.11 (conda) | Core language |
| Astronomy I/O | [astropy](https://www.astropy.org/) | FITS parsing, WCS astrometry, coordinate transforms |
| Streak detection | [ASTRiDE](https://github.com/dwkim78/ASTRiDE) | Classical contour-based streak detection |
| Orbit propagation | [sgp4](https://github.com/brandon-rhodes/python-sgp4) + [skyfield](https://rhodesmill.org/skyfield/) | SGP4 TLE propagation, satellite position/velocity |
| Catalog access | [spacetrack](https://pypi.org/project/spacetrack/) | Space-Track GP_History API client |
| Image processing | opencv-python, scipy | Preprocessing, morphological operations |
| Testing | pytest | Unit and integration tests |

**Phase 2 — YOLO-OBB Integration (Weeks 5–10)** *(planned)*

| Layer | Tool | Purpose |
|-------|------|---------|
| Streak detection (primary) | [StreakMind](https://github.com/StreakMind) YOLO-OBB | Neural oriented bounding-box streak detector |
| Streak detection (validator) | ASTRiDE | Classical cross-check of YOLO detections |

**Phase 3 — Hybrid Consensus (Weeks 11–14)** *(planned)*

| Layer | Tool | Purpose |
|-------|------|---------|
| Consensus layer | custom | Merges YOLO-OBB + ASTRiDE detections |
| Anomaly classification | DINOv3 | Flags unknown/anomalous objects not in TLE catalog |

---

## Setup

```bash
# Create and activate the conda environment
conda create -n argus python=3.11
conda activate argus
pip install -r requirements.txt

# Set Space-Track credentials (required for Week 3+)
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
```

## Running Tests

```bash
conda activate argus
pytest tests/ -v
```

## Data

Real FITS images from the [MILAN Sky Survey](https://zenodo.org/records/7049839)
(Parisot et al., 2023) are used for testing. Download to `data/milan/`:

```bash
pip install zenodo_get
zenodo_get 7049839 -o data/milan/2022-08/
```

Data files are excluded from version control (see `.gitignore`).

---

## Project Structure

```
Argus/
├── CLAUDE.md              ← agent instructions
├── README.md
├── requirements.txt
├── agent_docs/            ← design docs (read before coding)
├── src/
│   ├── ingest/            ← FITS parsing
│   ├── detection/         ← ASTRiDE streak detection
│   ├── astrometry/        ← WCS plate solving
│   └── matching/          ← Space-Track query, SGP4, scoring
├── tests/
├── results/               ← baseline metrics JSON output
└── data/                  ← not checked in
```
