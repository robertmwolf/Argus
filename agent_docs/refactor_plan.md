# Structural Refactor Plan

Recorded 2026-05-04 after a full codebase audit.  All implementation phases
(0–8) are complete.  This document captures structural debt to address before
the project is extended further.  None of these are blocking; the pipeline
runs end-to-end.  Address in priority order.

---

## Background

The project grew in two distinct waves:

**Wave 1 (Phase 0):** A classical ASTRiDE + SGP4 pipeline written entirely
inside `src/`.  It is self-contained: `fits_parser → classical_detector →
plate_solver → spacetrack_query → spatial_filter → propagator → scorer →
matcher`.  These modules were designed to be frozen once the ML path was built.

**Wave 2 (Phases 1–8):** The ML pipeline was built in top-level packages
(`inference/`, `training/`, `api/`, `eval/`, `db/`).  Some Phase 0 logic was
needed here too, so `inference/crossid.py` imports directly from `src/`, and
several algorithms from `src/` were re-implemented locally in `inference/`
rather than shared.

The result is two separate implementations of the same maths, and a hard
dependency from the "new" code into the "frozen" baseline.

---

## Issue 1 — `src/matching/spacetrack_query.py` is shared infrastructure locked inside a frozen package

**What happened:**
`inference/crossid.py` does:
```python
from src.matching.spacetrack_query import query_gp_history
```
This is the *only* import from `src/` outside of `src/` itself.  It exists
because both the classical and ML cross-ID paths need to query Space-Track,
and the caching logic in `spacetrack_query.py` is non-trivial.

**The problem:**
`src/` is supposed to be frozen (CLAUDE.md: "Phase 0 classical baseline —
complete, do not modify").  Having `inference/` depend on it means any
refactor of `src/matching/spacetrack_query.py` must be co-ordinated with the
ML pipeline.  It also means the "frozen baseline" can never truly be archived
or removed while the ML path is live.

**Long-term fix:**
Move `src/matching/spacetrack_query.py` to a new `common/` package:
```
common/
  __init__.py
  spacetrack_query.py   ← moved from src/matching/
```
Update the two callers:
- `src/matching/matcher.py` → `from common.spacetrack_query import ...`
- `inference/crossid.py` → `from common.spacetrack_query import ...`

`src/` can then be treated as a true read-only reference implementation.

**Effort:** ~1 hour.  No logic changes — file move + import updates + test
update for `tests/test_spacetrack_query.py`.

---

## Issue 2 — Three separate `angular_separation` implementations

The same haversine formula exists in three places with different unit
conventions:

| Location | Function | Units |
|----------|----------|-------|
| `src/matching/spatial_filter.py:18` | `_angular_separation` | degrees |
| `src/matching/matcher.py:76` | `_angular_separation_deg` | degrees |
| `inference/crossid.py:173` | `_angular_separation_arcsec` | arcseconds |

**Long-term fix:**
Once Issue 1 is addressed and `common/` exists, add:
```python
# common/astro_utils.py
def angular_separation_deg(ra1, dec1, ra2, dec2) -> float: ...
def angular_separation_arcsec(ra1, dec1, ra2, dec2) -> float: ...
```
Remove the three local copies and import from `common/`.

**Effort:** ~1 hour including test updates.

---

## Issue 3 — Two separate `gaussian_score` implementations

| Location | Function |
|----------|----------|
| `src/matching/scorer.py:21` | `gaussian_score(delta, sigma)` |
| `inference/crossid.py:199` | `_gaussian_score(delta, sigma)` |

Identical formula: `exp(-0.5 * (delta/sigma)^2)`.
The `src/` version also includes `tle_age_penalty()` and `aggregate_score()`
which have no equivalent in `inference/crossid.py` (the ML cross-ID uses a
simpler single-factor score).

**Long-term fix:**
Move the formula to `common/astro_utils.py` (see Issue 2) and import it in
both places.  Keep `src/matching/scorer.py` intact as the classical scorer
but have it import the formula from `common/`.

**Effort:** ~30 min, low risk.

---

## Issue 4 — Two separate SGP4 propagators

| Location | Function |
|----------|----------|
| `src/matching/propagator.py:32` | `propagate(tle_name, line1, line2, obs_time, lat, lon, alt)` |
| `inference/crossid.py:84` | `_propagate_to_radec(name, line1, line2, obs_time, lat, lon, alt)` |

Both use skyfield's `EarthSatellite`.  The crossid version even comments:
> *"Reuses the same skyfield EarthSatellite approach as src/matching/propagator.py"*

The classical propagator returns a `PropagationResult` dataclass with
velocity and direction; the ML version returns only RA/Dec.

**Long-term fix:**
Extend `src/matching/propagator.py`'s `propagate()` to optionally return
ra/dec-only mode (or add a thin wrapper).  Move it to `common/` as part of
Issue 1.  The ML crossid drops its local `_propagate_to_radec` and calls the
shared version.

**Effort:** ~2 hours.  Requires care around the return-type change.

---

## Issue 5 — Classical path (`src/`) is disconnected from the API and database

The classical pipeline produces `FITSImage`, `StreakDetection`, and
`CandidateMatch` dataclasses.  None of these are written to the database or
returned by any API endpoint.  The API uses only `inference/pipeline.py`.

The classical path can be run via `tests/test_end_to_end.py` and standalone
scripts, but there is no way to upload a FITS file to the web UI and get
classical detections back.

**Decision required (choose one):**

**Option A — Classical path as CLI-only benchmark tool (recommended)**
Accept that `src/` is a research baseline, not a service.  Document this
explicitly.  Add a `scripts/run_classical.py` CLI entry point so it remains
usable for benchmarking.  Update `README.md` to make the boundary clear.
Cost: ~30 min.

**Option B — Integrate classical detector as an alternative API backend**
Add a `DETECTOR=classical|dino` env var.  When `classical`, `api/main.py`
calls `src/` instead of `inference/pipeline.py`.  Requires an adapter layer
converting `CandidateMatch` → detection dict format (the JSON the frontend
expects).
Cost: ~4 hours.  Only worth doing if there is a specific need for
side-by-side comparison through the UI.

*Current recommendation: Option A.  The purpose of `src/` is to establish
baseline metrics (eval/benchmark.py already handles this comparison), not to
be a production detector.*

---

## Issue 6 — `architecture.md` references a missing file

`agent_docs/architecture.md` (line 22–26) describes:
```
[3. Search Window Computation]
  src/matching/search_window.py
```
This file does not exist.  The search-window logic was folded into
`src/matching/spacetrack_query.py` (the epoch/cone filter parameters are
computed inline there).

**Fix:** Remove the `search_window.py` box from the architecture diagram and
note inline that search-window computation lives in `spacetrack_query.py`.

**Effort:** 5 minutes.

---

## Recommended execution order

1. **Issue 6** — Fix the architecture doc (5 min, no code risk)
2. **Issue 1** — Create `common/` and move `spacetrack_query.py` (1 hr)
3. **Issues 2 + 3** — Consolidate angular_separation and gaussian_score into `common/astro_utils.py` while `common/` is open (1.5 hr)
4. **Issue 4** — Consolidate propagators (2 hr)
5. **Issue 5** — Decide on classical path strategy, add CLI or adapter (0.5–4 hr)

Total: ~5–9 hours of focused refactor work.  No new features required.
All existing tests should pass without modification after each step; the
only expected changes are import lines.

---

## What does NOT need to change

- The module boundary between `inference/` and `api/` — clean already
- The abstract storage/queue backends in `api/` — correct design
- The `training/` ↔ `inference/` dependency (training imports fits_loader) — intentional
- The lazy imports in `inference/pipeline.py` — needed for MPS/CPU flexibility
- The `db/` schema — no structural issues
- The test directory structure — mirrors source layout correctly
