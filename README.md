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
