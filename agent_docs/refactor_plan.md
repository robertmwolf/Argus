# Structural Refactor Plan

Recorded 2026-05-04 after a full codebase audit.  All implementation phases
(0–8) are complete.  This document captures the structural work needed before
the project is extended or handed off.  Nothing here is blocking; the pipeline
runs end-to-end.  Address in priority order.

---

## The Core Problem

A future maintainer cloning this repo sees the following at the root:

```
api/
agent_docs/
data/
db/
docker/
eval/
frontend/
inference/
models/
results/
scripts/
src/
tests/
training/
```

This layout is unreadable without prior context:

- **`src/` alongside `api/`, `inference/`, `training/`** — `src/` is not a
  meaningful name when everything in the repo is source code.  A maintainer
  cannot tell whether `src/` is the canonical implementation or an auxiliary
  one.  It is actually the *classical baseline*, a frozen research reference
  that should never be modified.  The name hides this completely.

- **`agent_docs/`** — named after the AI tooling that generated it, not after
  its content.  A maintainer unfamiliar with the development process will not
  know what to look in here for.  It contains architecture, dataset, deployment,
  and API documentation that belongs in `docs/`.

- **`models/`** — contains only MMDetection Python config files, no Python
  model classes.  A Python developer expects `models/` to be a package with
  class definitions.  The actual content is detector configurations.

- **No `pyproject.toml`** — the project is not installable as a package.  All
  imports work only because the repo root is on `sys.path`.  There is no
  machine-readable declaration of what this project is, what its entry points
  are, or what Python version it requires.

- **`requirements.txt` header reads "Phase 1 — Classical Baseline"** — stale
  comment from early development; the file now covers the full stack.

The refactor below fixes all of these in one coordinated pass.

---

## Before and After

**Today — what a new maintainer sees:**
```
src/                   ← frozen classical ASTRiDE pipeline (not obvious)
inference/             ← ML inference modules
training/              ← training scripts and data tools
api/                   ← FastAPI application
frontend/              ← React UI
eval/                  ← metrics and benchmark
db/                    ← schema and ORM
agent_docs/            ← documentation (name is an internal artifact)
models/                ← MMDetection config files (not Python models)
tests/                 ← mirrors src/ and top-level packages
```

**After refactor — what a new maintainer sees:**
```
classical/             ← frozen ASTRiDE baseline (Phase 0 reference, read-only)
common/                ← shared utilities used by both classical/ and inference/
inference/             ← ML inference modules (unchanged)
training/              ← training scripts and data tools (unchanged)
api/                   ← FastAPI application (unchanged)
frontend/              ← React UI (unchanged)
eval/                  ← metrics and benchmark (unchanged)
db/                    ← schema and ORM (unchanged)
docs/                  ← architecture, deployment, API, dataset documentation
configs/               ← MMDetection training configs (Swin-T and Swin-L)
tests/                 ← mirrors package layout
pyproject.toml         ← project metadata, entry points, dependency declaration
```

Every top-level directory now has a self-documenting name.  A new maintainer
can understand the structure without reading any documentation first.

---

## Issue 1 — `src/` should be renamed `classical/`

**Why it matters:**
`src/` is the Phase 0 classical ASTRiDE + SGP4 pipeline. The shared assistant
guide says it is "complete, do not modify." The name `src/` communicates nothing about this.
Renaming to `classical/` makes the purpose and status of the package
immediately legible — it is the classical detection baseline, frozen as a
research reference.

**Changes required:**
- Rename directory: `src/` → `classical/`
- Rename Python package init if present
- Update all imports: `from src.matching...` → `from classical.matching...`
  (only two locations: `src/matching/matcher.py` and `inference/crossid.py`)
- Rename test files: `tests/test_classical_detector.py` etc. already use
  descriptive names; only the `from src.` import lines need updating
- Update `agent_docs/assistant_guide.md`: replace all `src/` references with `classical/`
- Update `agent_docs/architecture.md` (will become `docs/architecture.md`)

**Effort:** 30 minutes.  Pure rename — zero logic changes.

---

## Issue 2 — `agent_docs/` should be renamed `docs/`

**Why it matters:**
`agent_docs/` is named after the AI agent that wrote the files during
development.  This is an internal implementation detail that has no meaning
to a future maintainer.  The directory contains project architecture,
deployment runbooks, dataset provenance, and API documentation — it is the
project's documentation root and should be called `docs/`.

**Changes required:**
- Rename directory: `agent_docs/` → `docs/`
- Update `AGENTS.md`, `CLAUDE.md`, and `agent_docs/assistant_guide.md`: replace `agent_docs/` references with `docs/`
- Update any script or test that references the path directly

**Effort:** 15 minutes.

---

## Issue 3 — `models/` should be renamed `configs/`

**Why it matters:**
`models/` contains MMDetection Python config files
(`streak_codino_swin_t.py`, `streak_codino_swin_l.py`).  A Python developer
expects `models/` to be a package containing model class definitions.  The
actual content is training configuration — hyperparameters, data pipelines,
schedules.  `configs/` is the standard name for this in the MMDetection
ecosystem.

**Changes required:**
- Rename directory: `models/` → `configs/`
- Rename subdirectory: `models/dino/` → `configs/dino/`
- Update `training/train_dino.py`: `_CONFIG_MAP` paths
  (`models/dino/...` → `configs/dino/...`)
- Update `models/dino/streak_codino_swin_t.py` `custom_imports` path if it
  references a path rather than a module
- Update `agent_docs/assistant_guide.md` and docs references

**Effort:** 20 minutes.

---

## Issue 4 — Create `common/` for shared infrastructure

**Why it matters:**
Two separate code trees (`classical/` and `inference/`) need the same
building blocks: Space-Track queries, angular separation math, Gaussian
scoring, and SGP4 propagation.  Currently this shared code is either
duplicated or the ML code imports directly from the "frozen" classical
package.  Both patterns create maintenance debt.

**Sub-issue 4a — `spacetrack_query.py` is shared infrastructure in a frozen package**

`inference/crossid.py` imports:
```python
from src.matching.spacetrack_query import query_gp_history
```
This is the only import from `src/` outside `src/` itself.  Moving to
`common/` breaks the frozen-package dependency and lets `classical/` be
archived independently if needed.

**Sub-issue 4b — Three separate `angular_separation` implementations**

| Location | Function | Units |
|----------|----------|-------|
| `classical/matching/spatial_filter.py:18` | `_angular_separation` | degrees |
| `classical/matching/matcher.py:76` | `_angular_separation_deg` | degrees |
| `inference/crossid.py:173` | `_angular_separation_arcsec` | arcseconds |

All three are the same haversine formula.

**Sub-issue 4c — Two separate `gaussian_score` implementations**

| Location | Function |
|----------|----------|
| `classical/matching/scorer.py:21` | `gaussian_score(delta, sigma)` |
| `inference/crossid.py:199` | `_gaussian_score(delta, sigma)` |

Identical formula: `exp(-0.5 * (delta/sigma)^2)`.

**Sub-issue 4d — Two separate SGP4 propagators**

| Location | Returns |
|----------|---------|
| `classical/matching/propagator.py:32` | `PropagationResult` (RA/Dec + velocity + direction) |
| `inference/crossid.py:84` | `(ra, dec)` tuple only |

Both use skyfield's `EarthSatellite`.  `inference/crossid.py` even comments:
*"Reuses the same skyfield EarthSatellite approach as src/matching/propagator.py"*

**Proposed `common/` layout:**
```
common/
  __init__.py
  spacetrack_query.py    ← moved from classical/matching/
  astro_utils.py         ← angular_separation_deg(), angular_separation_arcsec(),
                            gaussian_score()
  propagator.py          ← shared SGP4 propagation; classical/matching/propagator.py
                            can import from here and add its PropagationResult wrapper
```

**Callers to update after creating `common/`:**
- `classical/matching/spacetrack_query.py` → delete (file moves, not copies)
- `classical/matching/matcher.py` → `from common.spacetrack_query import ...`
- `classical/matching/spatial_filter.py` → `from common.astro_utils import angular_separation_deg`
- `classical/matching/scorer.py` → `from common.astro_utils import gaussian_score`
- `inference/crossid.py` → `from common.spacetrack_query import ...`,
  `from common.astro_utils import ...`, drop local `_propagate_to_radec`
- `tests/test_spacetrack_query.py` → update import path

**Effort:** ~3 hours total for all four sub-issues.  Low logic-change risk —
the functions themselves do not change.

---

## Issue 5 — Add `pyproject.toml`

**Why it matters:**
There is no machine-readable project declaration.  The project only works when
run from the repo root because all packages rely on the implicit root-level
`sys.path` entry.  A `pyproject.toml` enables:

- `pip install -e .` for development installs (no path hacks)
- Entry points (`argus-api`, `argus-pipeline`, `argus-eval`) for clean CLI
  invocation without `python -m`
- Formal declaration of Python version requirement (3.11)
- IDE tooling (Pyright, Ruff) picks up package roots automatically

**Minimum viable `pyproject.toml`:**
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "argus"
version = "0.1.0"
requires-python = ">=3.11"
description = "Satellite streak detection and identification pipeline"

[project.scripts]
argus-api      = "api.main:main"
argus-pipeline = "inference.pipeline:main"
argus-eval     = "eval.benchmark:main"
argus-classical = "scripts.run_classical:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["api*", "classical*", "common*", "db*", "eval*", "inference*", "training*"]
```

**Effort:** 1 hour including testing `pip install -e .` in the satid environment.

---

## Issue 6 — Classical path is disconnected from API and database

The classical pipeline (`classical/`) produces `FITSImage`, `StreakDetection`,
and `CandidateMatch` dataclasses that are never written to the database or
returned by any API endpoint.  The API uses only `inference/pipeline.py`.

**Decision: Option A — classical path as CLI-only benchmark tool (recommended)**

Accept that `classical/` is a research baseline, not a service.  Add a
`scripts/run_classical.py` CLI entry point for benchmarking.  Document the
boundary explicitly in `docs/architecture.md`.

Option B (add `DETECTOR=classical|dino` env var to the API) is ~4 hours and
only needed if side-by-side comparison through the UI is specifically required.
The eval benchmark already handles DINO vs YOLO comparison; the classical path
is covered by `eval/benchmark.py` through its direct API.

**Effort for Option A:** 30 minutes.

---

## Issue 7 — `docs/architecture.md` references a missing file

`docs/architecture.md` (line 22–26) describes `src/matching/search_window.py`
which does not exist.  The search-window logic was folded inline into
`spacetrack_query.py`.

**Fix:** Remove the `search_window.py` box from the diagram; note that
search-window computation lives in `spacetrack_query.py` (will be
`common/spacetrack_query.py` after Issue 4).

**Effort:** 5 minutes.

---

## Issue 8 — `requirements.txt` header is stale

The file header reads "Phase 1 — Classical Baseline" — a relic from early
development.  The file now covers the full stack (ML, API, frontend build,
database, evaluation).

**Fix:** Replace the header comment with a neutral description and a note
about the pinned numpy constraint.

**Effort:** 2 minutes.

---

## Recommended Execution Order

Execute as a **single coordinated commit** once training results are in hand
and the codebase is stable.  Doing this piecemeal risks a window where imports
are broken across commits.

1. **Issue 7** — Fix the architecture doc stale reference (5 min, no code)
2. **Issue 8** — Fix `requirements.txt` header (2 min)
3. **Issues 1–3** — Directory renames: `src/` → `classical/`, `agent_docs/` → `docs/`, `models/` → `configs/`
   - Update all internal references in the same commit
   - Update `AGENTS.md`, `CLAUDE.md`, and `agent_docs/assistant_guide.md` in the same commit
4. **Issue 4** — Create `common/`, move shared modules, update all callers
5. **Issue 5** — Add `pyproject.toml`, verify `pip install -e .` works
6. **Issue 6** — Add `scripts/run_classical.py` CLI entry point

**Total effort:** approximately 6–8 hours of focused work.

A future maintainer arriving after this refactor is complete can:
- Read the directory tree and immediately understand what each package does
- Run `pip install -e .` to get a working development install
- Find all project documentation in `docs/`
- Know that `classical/` is the frozen baseline and everything else is active
- Find shared math utilities in `common/` rather than searching three packages

---

## What Does NOT Need to Change

- The module boundary between `inference/` and `api/` — clean already
- The abstract storage/queue backends in `api/` — correct design
- The `training/` ↔ `inference/` dependency (training imports fits_loader) — intentional
- The lazy imports in `inference/pipeline.py` — needed for MPS/CPU flexibility
- The `db/` schema — no structural issues
- The `frontend/` layout — standard Vite project structure
- The `eval/` module — clean interface
- The `docker/` layout — correct separation of dev and cloud compose files
