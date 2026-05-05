# System Architecture

## Pipeline Overview

```
[FITS Image Input]
       ↓
[1. FITS Ingest & Header Parser]
   src/ingest/fits_parser.py
   → Extracts: DATE-OBS, RA, DEC, FOV, observer position, image data
       ↓
       ├──────────────────────────────┐
       ↓                              ↓
[2a. Streak Characterization]   [2b. Astrometric Plate Solve]
  src/detection/                  src/astrometry/
  classical_detector.py           plate_solver.py
  → velocity, direction,          → pixel coords → RA/Dec WCS
    magnitude, length             → angular separation calculations
       ↓                              ↓
       └──────────────────────────────┘
                      ↓
           [3. Search Window Computation]
             src/matching/search_window.py
             → epoch range (T_obs ± 3 days)
             → sky cone (FOV center + radius)
             → observer ECEF coords
                      ↓
           [4. Local TLE Catalog Query (DB-first, API fallback)]
             src/matching/tle_store.py          — primary: SQLite tle_catalog table
             src/matching/spacetrack_query.py   — fallback: GP class (live) or GP_History (archive)
             → filtered by epoch range (obs_time ± epoch_window_days)
             → DB hit returns instantly; API results stored immediately for future queries
             → returns ~5,000–15,000 candidate TLEs
                      ↓
           [5. Spatial Cone Filter]
             src/matching/spatial_filter.py
             → lightweight SGP4 prescreen
             → FOV intersection test
             → returns ~5–50 objects
                      ↓
           [6. SGP4 Propagation]
             src/matching/propagator.py
             → exact position at T_obs per candidate
             → velocity vector at T_obs
             → TLE age penalty computation
                      ↓
           [7. Multi-Factor Matching Engine]
             src/matching/matcher.py
             → Angular position score  (weight: 35%)
             → Angular velocity score  (weight: 30%)
             → Direction/PA score      (weight: 25%)
             → Brightness/mag score   (weight: 10%)
             → TLE age penalty applied to position score
                      ↓
           [8. Confidence Scorer]
             src/matching/scorer.py
             → Weighted score aggregation per candidate
             → Ambiguity detection (top-2 scores within 0.05)
             → Returns ranked CandidateMatch list
                      ↓
           [Output: Ranked candidates with confidence scores]
```

## Key Data Structures

### FITSImage (dataclass)
```python
@dataclass
class FITSImage:
    filepath: Path
    obs_time: datetime          # from DATE-OBS, UTC
    ra_center: float | None     # degrees, FOV center
    dec_center: float | None    # degrees, FOV center
    width_px: int               # NAXIS1
    height_px: int              # NAXIS2
    pixscale_arcsec: float | None  # arcsec/pixel
    exptime_sec: float | None   # exposure duration
    sitelat: float | None       # observer latitude, degrees
    sitelong: float | None      # observer longitude, degrees
    siteelev: float | None      # observer elevation, meters
    data: np.ndarray            # image pixel array
    header: fits.Header         # full FITS header
```

### StreakDetection (dataclass)
```python
@dataclass
class StreakDetection:
    x_start: float              # pixel
    y_start: float              # pixel
    x_end: float                # pixel
    y_end: float                # pixel
    x_center: float             # pixel, midpoint
    y_center: float             # pixel, midpoint
    angle_deg: float            # position angle, degrees
    length_px: float            # streak length in pixels
    width_px: float             # estimated streak width
    shape_factor: float         # ASTRiDE elongation metric
    area_px: float              # area in pixels
    ra_start: float | None      # sky coords (set after plate solve)
    dec_start: float | None
    ra_end: float | None
    dec_end: float | None
    ra_center: float | None
    dec_center: float | None
    angular_velocity_arcsec_s: float | None  # set after plate solve + exptime
    position_angle_deg: float | None         # celestial PA
```

### CandidateMatch (dataclass)
```python
@dataclass
class CandidateMatch:
    norad_id: int
    object_name: str
    tle_epoch: datetime
    tle_age_hours: float        # hours from tle_epoch to obs_time
    predicted_ra: float         # degrees at obs_time
    predicted_dec: float        # degrees at obs_time
    predicted_velocity_arcsec_s: float
    predicted_direction_deg: float
    predicted_magnitude: float | None
    angular_sep_arcsec: float   # observed vs predicted position
    velocity_delta_pct: float   # observed vs predicted speed
    direction_delta_deg: float  # observed vs predicted PA
    magnitude_delta: float | None
    position_score: float       # 0.0–1.0
    velocity_score: float       # 0.0–1.0
    direction_score: float      # 0.0–1.0
    magnitude_score: float      # 0.0–1.0
    weighted_score: float       # final aggregated score
    ambiguous: bool             # True if next candidate within 0.05
```

## Design Notes

### Streak Direction Ambiguity (two-solution problem)
ASTRiDE detects streaks as contours and assigns `x_start`/`x_end` arbitrarily — it has
no knowledge of which end the satellite occupied first. This means `angle_deg` and
`position_angle_deg` carry an inherent 180° ambiguity.

This matters for the direction score in the matcher (weight: 25%), which compares observed
PA against SGP4-predicted PA. Assuming the wrong end is "start" flips the comparison by
180°, producing a near-zero direction score for the correct match.

**Resolution rule (implemented in `matcher.py`):**
For each TLE candidate, compute `direction_delta_deg` against both the stored PA and
`PA + 180° (mod 360)`, and take the smaller of the two deltas. The SGP4-predicted
position angle is the disambiguator — the correct direction is whichever orientation
aligns better with the predicted trajectory. Do **not** resolve this ambiguity in
`classical_detector.py` or `plate_solver.py`; pass through both possibilities implicitly
by letting the matcher test both orientations.

---

## Design Principles

### Classical-Only in Phase 1
No ML, no neural networks. Every decision in Phase 1 is deterministic
and explainable. This is intentional — you need a baseline to know
what ML actually improves.

### TLE Age Awareness
Every candidate carries its TLE epoch and age relative to obs_time.
Position scores are penalized for stale TLEs using a Gaussian decay:
- < 6 hours: full score
- 6–24 hours: ~80% score
- 24–72 hours: ~50% score
- > 72 hours: ~20% score (flagged in output)

### TLE Catalog Strategy
TLE data is stored permanently in the local `argus.db` SQLite database (`tle_catalog` table).
The lookup path in `crossid.py` is:
1. **DB first**: `query_tles_for_window()` — epoch-range filter, returns in milliseconds
2. **API fallback**: only if DB returns nothing
   - obs_time < 2 hours old → `query_gp_current()` (GP class, ≤ once/hour rate limit)
   - obs_time ≥ 2 hours old → `query_gp_history()` (GP_History, use sparingly)
3. **Store immediately**: API results are written to DB via `upsert_tles()` before returning

Bootstrap the DB once per environment with `scripts/bootstrap_tle_catalog.py`. Keep it
current with `scripts/update_tle_catalog.py` (hourly cron or manual).

### Historical Image Support
The pipeline is identical for current and archival images. For a 6-month-old image,
`query_tles_for_window()` filters the local DB by the obs_time epoch from the FITS header.
No API call is needed once historical data is bootstrapped. If the local DB lacks coverage
for that period, `query_gp_history()` is called as a one-time fallback and the results
are stored locally for all future queries.

### Spatial Filtering
Never run full SGP4 on the entire TLE window. Always:
1. Epoch-range filter in the DB (obs_time ± epoch_window_days) — reduces 16M → 5–15k
2. Spatial cone (local SGP4 prescreen in `spatial_filter.py`) — reduces 5–15k → 5–50
3. Only then run full SGP4 propagation for the small surviving set

## Scoring Formulas

### Per-factor scoring (Gaussian falloff)
```python
def gaussian_score(delta: float, sigma: float) -> float:
    """Score from 0–1 using Gaussian falloff from zero error."""
    return math.exp(-0.5 * (delta / sigma) ** 2)

# Position: sigma = 0.25 degrees (half the 0.5° threshold)
position_score = gaussian_score(angular_sep_deg, sigma=0.25)

# Velocity: sigma = 5% (half the 10% threshold)
velocity_score = gaussian_score(velocity_delta_pct, sigma=5.0)

# Direction: sigma = 2.5 degrees (half the 5° threshold)
direction_score = gaussian_score(direction_delta_deg, sigma=2.5)

# Magnitude: sigma = 0.75 mag (half the 1.5 mag threshold)
magnitude_score = gaussian_score(magnitude_delta, sigma=0.75)
```

### Weighted aggregation
```python
weighted_score = (
    0.35 * position_score  * tle_age_penalty +
    0.30 * velocity_score  +
    0.25 * direction_score +
    0.10 * magnitude_score
)
```

## Phase Roadmap

| Phase | Detector | Notes |
|-------|----------|-------|
| 0 — Classical baseline | ASTRiDE (Hough) | ✅ Complete. `src/` directory. Baseline metrics. |
| 1 — Data pipeline | — | FITS loader, COCO conversion, augmentations (`training/`) |
| 2 — DINO model | DINO + Swin-L backbone | MMDetection config + training script (adapts StreakMind architecture) |
| 3 — Cross-identification | SGP4 ephemeris matching | Reuses Phase 0 matching logic, adapted for ARGUS inference |
| 4 — Database | PostgreSQL/SQLite schema | SQLAlchemy async, normalized schema |
| 5 — API | FastAPI | Upload / result / image endpoints |
| 6 — Frontend | React + Vite + Tailwind | Canvas OBB renderer |
| 7 — Docker | docker-compose | Separate API, worker (GPU), frontend containers |
| 8 — Evaluation | mAP, angle error | DINO vs YOLO head-to-head benchmark |

## ARGUS ML Pipeline (Phases 1–8)

```
[FITS or PNG Upload]
        ↓
[inference/fits_loader.py — FITSLoader]
   Z-score normalize → 3-channel uint8 array
   Extract WCS metadata
        ↓
[Co-DINO inference — inference/pipeline.py]
   pipeline.load_model()  — load once for batch eval
   pipeline.run(fits_path, model=model)  — reuse preloaded model
   models/dino/streak_codino_swin_l.py  — MMDetection config
   → axis-aligned bboxes + confidence scores
        ↓
[inference/postprocess.py — AngleRefinement + StreakNMS]
   Radon transform on each crop → refined angle
   Reconstruct OBB (cx, cy, w, h, angle_deg)
   Rotated IoU NMS via Shapely
        ↓
[inference/crossid.py — SatelliteCrossIdentifier]
   WCS → RA/Dec for streak midpoints
   DB-first TLE lookup (src/matching/tle_store.py), API fallback
   SGP4 propagate candidate TLEs to obs epoch
   Gaussian confidence score per candidate
   Top-3 identifications per detection
        ↓
[Database write — SQLAlchemy async]
   observations, detections, identifications tables
        ↓
[API response / Frontend display]
   Canvas OBB rendering, detection table
```

## Service Deployment Roadmap

See `agent_docs/service_roadmap.md` for the full plan. Summary:

| Phase | What |
|-------|------|
| S1 | Standalone CLI pipeline (complete Phases 1–3 first) |
| S2–S5 | FastAPI + React frontend + Docker + Cloudflare Tunnel |
| S6 | Cloud scale: Redis/SQS, S3, separate GPU worker containers |
