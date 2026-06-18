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
           [4. Local TLE Catalog Query (DB-only)]
             src/matching/tle_store.py          — primary: SQLite tle_catalog table
             → filtered by epoch range (obs_time ± epoch_window_days)
             → DB hit returns instantly
             → DB miss leaves object unidentified/unknown; no Space-Track call
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
2. **No inference API fallback**: if DB returns nothing, detections remain
   unidentified/unknown for that observation
3. **Explicit maintenance only**: Space-Track GP/GP_History helpers exist for
   operator-run catalog maintenance or diagnostics, not automatic inference

Bootstrap the DB once per environment with `scripts/bootstrap_tle_catalog.py`.
Current/live Space-Track integration is intentionally not automatic.

### External Orbital Lookup Services
ARGUS intentionally keeps the local TLE catalog as the primary orbital-object
lookup source.  A direct runtime replacement, such as a field-of-view ephemeris
API, has two practical problems for this project:

1. **Runtime reliability and reproducibility** — external services can be slow,
   unavailable, rate-limited, or change their backing catalog.  Cross-ID results
   need to be repeatable for research runs, regression tests, and historical
   images.
2. **Input readiness** — field-of-view lookup APIs require reliable sky
   geometry: observation time, observer location, and solved RA/Dec endpoints or
   a trustworthy field center/radius.  Many uploaded real streak FITS files have
   useful DATE-OBS/site/target hints but no celestial WCS until ASTAP or a
   sidecar solve is available.

For these reasons, external orbital-object APIs should be treated as research
probes or offline comparison tools only.  They should not replace the local
`tle_catalog` inference path unless they can demonstrate low latency, high
availability, historical coverage, expected-NORAD recovery, and stable
provenance across the ARGUS real-image benchmark set.

### ML FITS Loading and WCS Sidecars
`inference/fits_loader.py` normalises FITS image data for the DINO path and
loads celestial WCS in this order:

1. FITS header WCS, when `astropy.wcs.WCS(header).has_celestial` is true.
2. Same-stem `.wcs` or `.WCS` sidecar, for ASTAP/SkyTrack solves such as the
   GTImages dataset.
3. `None`, leaving RA/Dec outputs null while preserving pixel-space detections.

The loader returns `wcs_source` as `fits`, `sidecar`, or `None`. API uploads are
processed from temporary files, so `api/main.py` copies a matching sidecar next
to the temporary FITS before calling the inference pipeline. The lookup checks
local upload storage plus common ARGUS data locations (`data/GTImages/`,
`data/raw/`, and `data/sample/`).

### API Result Coordinate Contract
`GET /api/result/{job_id}` returns source image dimensions as `image_width` and
`image_height` when they can be recovered from the uploaded FITS or PNG. The
frontend treats all detection coordinates as original source-image pixels and
uses those dimensions to scale overlays onto the rendered preview image.

### Historical Image Support
The pipeline is identical for current and archival images. For a 6-month-old image,
`query_tles_for_window()` filters the local DB by the obs_time epoch from the FITS header.
No API call is needed once historical data is bootstrapped. If the local DB lacks coverage
for that period, cross-identification is skipped and the detection is reported as
unknown. Broad `gp_history` calls are never made from inference.

### Spatial Filtering
Never run full SGP4 on the entire TLE window. Always:
1. Epoch-range filter in the DB (obs_time ± epoch_window_days) — reduces 16M → 5–15k
2. Spatial cone (local SGP4 prescreen in `spatial_filter.py`) — reduces 5–15k → 5–50
3. Only then run full SGP4 propagation for the small surviving set

### Cross-ID Performance Backlog
Current status: `inference/crossid.py` still scores each selected detection
against the local TLE window by propagating one TLE per catalogued object. After
deduplicating TLE epochs this is usually tens of thousands of SGP4 calls per
detection, so full cross-identification can take tens of seconds on a large
catalog. Keep `CROSSID_MAX_DETECTIONS` small for interactive API use until this
is improved.

Suggested future improvements:
- Add the spatial cone prescreen from the architecture path to the inference
  cross-ID flow before full scoring. Use the FITS WCS-derived field center,
  field radius, observer position, and obs_time to reduce candidates before
  midpoint/length scoring.
- Cache propagated topocentric positions for `(obs_time, observer, catalog
  epoch window)` once per uploaded image, then score every detection against the
  cached RA/Dec positions. This avoids repeating the same SGP4 propagation for
  multiple detections in the same frame.
- Add coarse DB-side orbit filtering where safe: mean motion/orbit class,
  declination/RA bins from a precomputed ephemeris cache, or known NORAD/object
  hints from GTImages-style filenames and FITS `OBJECT` headers.
- Consider a nightly ephemeris index for bootstrapped catalog days: precompute
  coarse sky positions on a time grid per observing site and query nearby bins
  during inference, then run exact SGP4 only for the survivors.
- Add performance tests that assert candidate-count reductions and wall-clock
  budgets for representative GEO and LEO images, including `AMAZONAS 5`
  (`Streak_42934_222307.fits`) as a regression case.

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
| 0 — Classical baseline | ASTRiDE | ✅ Complete. `src/` directory. Baseline metrics. |
| 1 — Data pipeline | — | ✅ FITS loader, COCO conversion, augmentations (`training/`) |
| 2 — DINO model | DINO + ViT-S backbone | ✅ MMDetection configs + training script |
| 3 — Cross-identification | SGP4 ephemeris matching | ✅ Reuses Phase 0 matching logic |
| 4 — Database | SQLite schema | ✅ SQLAlchemy async, normalized schema |
| 5 — API | FastAPI | ✅ Upload / result / image endpoints |
| 6 — Frontend | React + Vite + Tailwind | ✅ Line-segment canvas renderer; heatmap overlay toggle |
| 8 — Evaluation | F1, threshold sweep | ✅ `eval/benchmark.py`, `scripts/run_posthoc_threshold_analysis.py` |
| ML R1–R15 | ViT-S/ConvNeXt heatmap | ✅ 15 training runs; Run 15 (400 px / zscore) is active |
| ML R17 — Partial unfreeze | ViT-S last 2–4 blocks | ⏳ Next step; requires GPU |

## ARGUS ML Pipeline (Phases 1–8, current state)

```
[FITS or PNG Upload]
        ↓
[inference/fits_loader.py — FITSLoader]
   Z-score normalisation (zscore, 3σ clip → uint8)
   Extract FITS-header or sidecar WCS metadata
        ↓
[inference/tiled_pipeline.py — adaptive tiler]
   400 px tiles, 50% overlap → ~650 tiles per 6248×4176 Atwood frame
        ↓
[inference/vits_heatmap_detector.py — primary ML detector]
   DINOv3 ViT-S/16 (frozen) + MLP head → per-tile (25×25) probability map
   Bilinear upsample → (400, 400) per tile
   Stitch tiles → full-image heatmap (max-pool overlapping regions)
   Threshold at VITS_HEATMAP_THRESHOLD (default 0.85)
   Connected-component extraction → candidate line segments
   stitch_collinear_fragments (max_growth_ratio=3.0) → merged segments
   → list of {method: "vits_heatmap", seg: {x1,y1,x2,y2}, confidence} dicts
        ↓
[Optional parallel detectors — inference/pipeline.py]
   ASTRiDE (ARGUS_ENABLE_ASTRIDE=1): classical σ-threshold contour detection on raw FITS
        ↓
[inference/postprocess.py — StreakGrouping + UCS]
   Per-detector NMS (rotated-IoU)
   Cross-detector grouping: rotated-IoU ≥ 0.5, IoMin ≥ 0.3, or collinear-fragment match
   Fuse grouped geometry to outer endpoints → single line segment per physical streak
   ASTRiDE-only confidence lowering
   Unified Confidence Score (F-0.5 weighted corroboration)
        ↓
[inference/crossid.py — SatelliteCrossIdentifier]
   WCS → RA/Dec for streak midpoints
   Local TLE lookup (src/matching/tle_store.py); missing coverage → unknown
   SGP4 propagate candidate TLEs to obs epoch
   Gaussian confidence score per candidate
   Top-3 identifications per detection
        ↓
[Database write — SQLAlchemy async]
   observations, detections, identifications tables
        ↓
[API response / Frontend display]
   Line-segment canvas rendering; heatmap overlay toggle; detection table
   Cyan = vits_heatmap; Amber = ASTRiDE
```

## Service Deployment Roadmap

| Phase | What |
|-------|------|
| S1 | Standalone CLI pipeline (complete Phases 1–3 first) |
| S2–S5 | FastAPI + React frontend + Cloudflare Tunnel |
| S6 | Cloud scale: Redis/SQS, S3, separate GPU workers |
