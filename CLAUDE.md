# ARGUS — Automated Recognition and Grading of Unidentified Streaks

## What This Is
An automated pipeline that identifies satellites in FITS telescope images
by detecting streaks and matching them against TLE orbital data from
Space-Track's GP_History API using SGP4 propagation and multi-factor
confidence scoring.

## Current Phase
**ML PIPELINE — Co-DINO streak detection (Phases 1–8).**
Phase 0 (classical ASTRiDE baseline) is complete and lives in `src/`.
Active work is now the Co-DINO ML pipeline. See `agent_docs/ml_pipeline.md`.

Development machine: MacBook Air M3 (CPU/MPS only).
Training machine: Lambda Labs A100 (rented when code is ready).

## Read These Before Starting Any Task
Always read the relevant agent_docs file before writing code:

- `agent_docs/architecture.md`    — full system design, component map, data flow
- `agent_docs/ml_pipeline.md`     — Co-DINO pipeline, hardware constraints, 8-phase sequence
- `agent_docs/phase1_goals.md`    — Phase 0 classical baseline reference (ASTRiDE)
- `agent_docs/datasets.md`        — where to get test data, download links
- `agent_docs/dependencies.md`    — exact packages, versions, install commands (includes ML stack)
- `agent_docs/service_roadmap.md` — FastAPI service, Docker, Cloudflare Tunnel, scale path
- `agent_docs/test_strategy.md`   — how to measure and record baseline accuracy
- `agent_docs/spacetrack.md`      — Space-Track API usage, rate limits, caching rules

## Stack
- Python 3.11, conda environment named `satid`
- Phase 0 libs: astropy, astride, sgp4, skyfield, spacetrack, opencv-python, scipy
- ML libs: torch, torchvision, mmengine, mmcv, mmdet (Co-DINO), albumentations
- Testing: pytest

## Project Structure
```
Argus/
├── CLAUDE.md
├── README.md
├── agent_docs/          ← read before coding
├── src/
│   ├── ingest/          ← FITS parsing
│   ├── detection/       ← ASTRiDE streak detection
│   ├── astrometry/      ← WCS plate solving, pixel→sky coords
│   └── matching/        ← Space-Track query, SGP4, scoring
├── tests/               ← pytest test files
├── results/             ← baseline metrics JSON output
└── data/                ← FITS data (not checked in)
    ├── milan/           ← MILAN sky survey FITS files
    └── sample/          ← small sample files for quick testing
```

## Academic Research Context
This project is academic research software. It builds on the following prior works:

- **ASTRiDE** — Automated Streak Detection for Astronomical Images
  (Kim et al., https://github.com/dwkim78/ASTRiDE)
- **StreakMind** — YOLO-OBB satellite streak detection model
  (StreakMind project, cite per their published paper/repo)
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
- All file paths via `pathlib.Path`, never raw strings
- Log with `logging` module, not `print()` (except __main__ blocks)

## Environment Variables Required
```bash
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
```

## Running Tests
```bash
conda activate argus
pytest tests/ -v
```

## Workflow Rules
- Build one week's tasks completely before moving to the next
- Write pytest tests alongside each module, not after
- Run pytest after every module is complete — fix failures before continuing
- Write baseline metrics to results/phase1_baseline.json at end of Week 4
- Ask for a plan before writing code for any module over 100 lines
