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
           [4. Space-Track GP_History Query]
             src/matching/spacetrack_query.py
             → filtered by epoch range + optional object type
             → cached to disk keyed by (obs_hour, ra, dec)
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

### Surgical Space-Track Queries
Never pull the full catalog. Always filter by:
1. Epoch range (± 3 days of obs_time) — reduces 50k → 5–15k objects
2. Spatial cone (local SGP4 prescreen) — reduces 5–15k → 5–50 objects
3. Only then run full SGP4 for the small surviving set

### Historical Image Support
The pipeline is identical for current and archival images.
For a 6-month-old image, the Space-Track query targets GP_History
using the obs_time from the FITS header — not today's catalog.
This is handled transparently in spacetrack_query.py.

### Caching Strategy
Cache Space-Track query results to disk keyed by:
  (obs_time rounded to nearest hour, ra_center rounded to 1°, dec_center rounded to 1°)
TTL: 24 hours for historical queries (data won't change),
     2 hours for recent queries (catalog updates frequently).

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

| Phase | Weeks | Detector | Matching |
|-------|-------|----------|---------|
| 1 — Classical baseline | 1–4 | ASTRiDE (Hough) | SGP4 + weighted scoring |
| 2 — YOLO-OBB integration | 5–10 | YOLO-OBB primary + ASTRiDE validator | Same |
| 3 — Hybrid consensus | 11–14 | Consensus layer | + DINOv3 anomaly classifier |

## Service Deployment Roadmap

See `agent_docs/service_roadmap.md` for the full plan. Summary:

| Phase | What |
|-------|------|
| S1 | Standalone CLI pipeline (prerequisite — complete Phases 1–3 first) |
| S2 | FastAPI wrapper, local file storage, in-memory job tracking |
| S3 | Frontend with canvas OBB rendering, served from FastAPI |
| S4 | Dockerize API + frontend, `docker-compose` with volume-mounted data |
| S5 | Cloudflare Tunnel for self-hosted public HTTPS (optional) |
| S6 | Scale path: Redis/SQS jobs, S3/Blob storage, cloud container deploy |
