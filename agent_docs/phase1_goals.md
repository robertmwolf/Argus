# Phase 1 Goals — Classical Baseline (Weeks 1–4)

## Guiding Principle
Build one week completely before moving to the next.
Each week ends with a passing pytest suite and a clear metric recorded.
The goal is not perfection — it is a working end-to-end pipeline
with documented baseline numbers to compare against Phase 2.

---

## Week 1: FITS Ingest & Header Parser

### File to build
`src/ingest/fits_parser.py`

### What it must do
- Accept a file path to any `.fits` or `.fit` file
- Open with `astropy.io.fits`
- Extract all fields into a `FITSImage` dataclass (see architecture.md)
- Raise `ValueError` with a clear message if `DATE-OBS` or `NAXIS1/2` are missing
- Handle both DATE-OBS formats: `2024-04-02T02:55:24.38` and `2024-04-02 02:55:24`
- Parse `DATE-OBS` into a timezone-aware `datetime` (UTC)
- Gracefully handle missing optional fields (RA, DEC, PIXSCALE, SITELAT etc.)
  by setting them to `None` — do not crash
- Normalize image data to float32

### Standalone __main__
Running `python src/ingest/fits_parser.py path/to/image.fits` should:
- Print all extracted header fields
- Print image shape and dtype
- Print any missing optional fields as warnings

### Test file
`tests/test_fits_parser.py`

Tests must cover:
- [ ] Parse a real FITS file from data/sample/ successfully
- [ ] Correct datetime parsing including timezone awareness
- [ ] Missing DATE-OBS raises ValueError with clear message
- [ ] Missing optional fields (RA, DEC etc.) return None, not crash
- [ ] Image data is float32 numpy array

### Success Metric — Week 1
Run parser on first 50 MILAN FITS files.
**Target: 100% parse without crash, 0 unhandled exceptions.**
Record result in `results/week1_ingest.json`.

---

## Week 2: ASTRiDE Classical Streak Detector

### File to build
`src/detection/classical_detector.py`

### What it must do
- Accept a `FITSImage` (from week 1 parser)
- Run ASTRiDE `Streak` detection on the image data
- Return a list of `StreakDetection` dataclasses (see architecture.md)
- Compute x_center, y_center as midpoint of start/end
- Compute angle_deg from (x_start,y_start) → (x_end,y_end)
  (0° = east, 90° = north, standard position angle convention)
- Compute length_px as Euclidean distance start→end
- Set all sky coord fields (ra_*, dec_*) to None — plate solve not yet wired
- Accept `contour_threshold` as a parameter (default 3.0)
  so sensitivity can be tuned without code changes
- Accept `min_length_px` as a parameter (default 20)
  to filter out very short detections

### Preprocessing before ASTRiDE
Before calling ASTRiDE:
1. Subtract median background using `sep` library
2. Clip extreme pixel values at 99.9th percentile
3. Convert to 16-bit unsigned int (ASTRiDE expects this)

### Standalone __main__
Running `python src/detection/classical_detector.py path/to/image.fits` should:
- Run detection
- Print number of streaks found and their properties
- Save an annotated PNG to the same directory showing bounding boxes

### Test file
`tests/test_classical_detector.py`

Tests must cover:
- [ ] Returns empty list on image with no streaks (don't crash)
- [ ] Returns StreakDetection objects with correct types
- [ ] angle_deg is in range [-180, 180]
- [ ] length_px > 0 for all detections
- [ ] Tunable parameters (contour_threshold, min_length_px) change results

### Success Metric — Week 2
Run detector on 20 MILAN images known to contain Starlink passes.
**Target: detect visible streak in ≥ 70% of images.**
Record per-image results in `results/week2_detection.json`.

---

## Week 3: Astrometry + Space-Track Query

### Files to build

#### `src/astrometry/plate_solver.py`
- Accept a `FITSImage` and a `StreakDetection`
- Extract WCS from FITS header using `astropy.wcs.WCS`
- Convert pixel coordinates (x, y) to sky coordinates (RA, Dec)
  using `wcs.all_pix2world()`
- Populate `ra_start`, `dec_start`, `ra_end`, `dec_end`, `ra_center`, `dec_center`
  on the `StreakDetection` in place
- Compute `position_angle_deg` (celestial, from North through East)
- Compute `angular_velocity_arcsec_s` from streak length in arcsec / exptime
- If FITS has no valid WCS, log a warning and return with sky fields as None
  (do not crash — the pipeline continues, just without sky coords)

#### `src/matching/spacetrack_query.py`
- Accept `obs_time: datetime`, `epoch_window_days: int = 3`
- Query Space-Track `GP_History` class
- Filter: epoch within `(obs_time - window, obs_time)`
- Sort: `orderby='epoch desc'` to prefer most recent TLE per object
- Cache results to disk at `data/cache/` using pickle
  Cache key: `f"{obs_time.strftime('%Y%m%d%H')}_{epoch_window_days}d"`
  Cache TTL: 48 hours for historical queries (obs_time > 7 days ago)
             2 hours for recent queries
- Rate limiting: never make more than 1 request per 3 seconds
- Return: list of raw TLE dicts from Space-Track JSON response
- Read credentials from environment variables `SPACETRACK_USER`, `SPACETRACK_PASS`
  Raise clear error if not set

### Standalone __main__ for spacetrack_query.py
Running `python src/matching/spacetrack_query.py 2024-04-02T02:55:24` should:
- Query GP_History for that date ± 3 days
- Print count of TLEs returned
- Print first 5 TLE object names as a sanity check

### Test files
`tests/test_plate_solver.py`
- [ ] Pixel (0,0) maps to a reasonable sky coord given known header
- [ ] Center pixel maps to CRVAL1/CRVAL2 from header
- [ ] angular_velocity is positive and in reasonable range for LEO (0.1–2.0 deg/s)

`tests/test_spacetrack_query.py`
- [ ] Returns non-empty list for a known historical date
- [ ] Caches result and does not re-hit Space-Track on second call
- [ ] Missing env vars raises clear error

### Success Metric — Week 3
For one confirmed Starlink pass in MILAN data (manually identified):
**Target: correct NORAD ID appears in candidate list from GP_History query.**
Record in `results/week3_query.json`.

---

## Week 4: SGP4 Matching + End-to-End Metrics

### Files to build

#### `src/matching/spatial_filter.py`
- Accept: list of TLE dicts, obs_time, observer lat/lon/alt, ra_center, dec_center, fov_radius_deg
- For each TLE: run SGP4 to get position at obs_time
- Convert TEME position to topocentric RA/Dec for observer
- Return only TLEs whose predicted position falls within fov_radius_deg of (ra_center, dec_center)
- This is the pre-screen — lightweight, fast, reduces 5–15k → 5–50 candidates

#### `src/matching/propagator.py`
- Accept: single TLE dict, obs_time, observer lat/lon/alt
- Return: (predicted_ra, predicted_dec, velocity_arcsec_s, direction_deg, tle_age_hours)
- Use `sgp4` library `Satrec.twoline2rv()` and `sat.sgp4(jd, fr)`
- Convert TEME frame to topocentric using `skyfield` for observer-relative coords
- Compute TLE age penalty using Gaussian decay (see architecture.md)

#### `src/matching/matcher.py`
- Accept: `StreakDetection`, list of propagated candidates, `FITSImage`
- For each candidate:
  - Compute angular_sep_arcsec (observed vs predicted position)
  - Compute velocity_delta_pct (observed vs predicted speed)
  - Compute direction_delta_deg (observed vs predicted PA)
  - Compute magnitude_delta if magnitudes available
  - Score each factor using gaussian_score() (see architecture.md)
  - Apply TLE age penalty to position_score
  - Compute weighted_score
- Sort by weighted_score descending
- Flag ambiguous=True if top-2 scores within 0.05 of each other
- Return list of `CandidateMatch` objects

#### `src/matching/scorer.py`
- Contains `gaussian_score(delta, sigma)` utility function
- Contains `tle_age_penalty(age_hours, orbit_type='LEO')` function
- Contains `aggregate_score(position, velocity, direction, magnitude, age_hours)` function

#### `tests/test_end_to_end.py`
Full pipeline test. For each of 5 test FITS images:
1. Parse FITS → FITSImage
2. Detect streaks → list of StreakDetection
3. Plate solve → populate sky coords
4. Query GP_History → TLE candidates
5. Spatial filter → short candidate list
6. Propagate + match → CandidateMatch list
7. Assert: correct NORAD ID in top-3 candidates for confirmed passes

### Success Metrics — Week 4 (record all in results/phase1_baseline.json)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Detection recall | ≥ 60% | % of confirmed passes ASTRiDE found |
| False positive rate | ≤ 30% | % of detections with no matching TLE |
| Match rate | ≥ 70% | % of detections correctly IDed (correct NORAD in top-3) |
| Position residual | log arcsec | median angular sep observed vs predicted |
| Ambiguous rate | log % | % of matches flagged ambiguous |
| Processing time | log seconds | wall clock per image, end-to-end |

### results/phase1_baseline.json format
```json
{
  "phase": 1,
  "date_recorded": "2024-XX-XX",
  "images_tested": 5,
  "detection_recall": 0.0,
  "false_positive_rate": 0.0,
  "match_rate": 0.0,
  "median_position_residual_arcsec": 0.0,
  "ambiguous_rate": 0.0,
  "mean_processing_time_sec": 0.0,
  "notes": ""
}
```
