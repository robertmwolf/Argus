# Test Strategy

## Philosophy
Every module has a pytest test file. Tests run after every module is built.
Baseline metrics are recorded as JSON — not printed and forgotten.
The numbers from Phase 1 become the comparison target for Phase 2.

---

## Test Structure

```
tests/
├── conftest.py                  ← shared fixtures; auto-skip integration/real_data markers
├── test_fits_parser.py          ← Phase 0: FITS parsing
├── test_classical_detector.py   ← Phase 0: ASTRiDE detection
├── test_plate_solver.py         ← Phase 0: WCS plate solving
├── test_spacetrack_query.py     ← Phase 0/3: Space-Track query + caching
├── test_matcher.py              ← Phase 0: SGP4 matcher
├── test_scorer.py               ← Phase 0: multi-factor scorer
├── test_end_to_end.py           ← Phase 0: classical pipeline end-to-end
├── test_fits_loader.py          ← Phase 1: FITS→tensor loader
├── test_convert_labels.py       ← Phase 1: COCO label conversion
├── test_dataset.py              ← Phase 1: FITSStreakDataset
├── test_augmentations.py        ← Phase 1: albumentations pipeline
├── test_device.py               ← Phase 2: device abstraction
├── test_model_configs.py        ← Phase 2: MMDet config validation
├── test_train_dino.py           ← Phase 2: training script
├── test_pipeline.py             ← Phase 3: inference pipeline
├── test_postprocess.py          ← Phase 3: Radon refinement, NMS, grouping/fusion
├── test_crossid.py              ← Phase 3: TLE cross-identification
├── test_db.py                   ← Phase 4: async ORM models
├── test_api.py                  ← Phase 5: FastAPI endpoints
│                                  includes source image dimensions and WCS sidecar copy
├── test_eval.py                 ← Phase 8: metrics + benchmark
└── test_real_images.py          ← real FITS images (auto-skipped when dir empty)
```

---

## conftest.py — Shared Fixtures

**Claude Code: create tests/conftest.py with these fixtures:**

```python
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from astropy.io import fits

SAMPLE_DIR = Path('data/sample')

@pytest.fixture(scope='session')
def sample_fits_with_streak(tmp_path_factory):
    """Synthetic FITS file with one known streak."""
    # Import and run the synthetic generator from datasets.md
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from make_test_fits import make_test_fits
    path = tmp_path_factory.mktemp('fits') / 'test_streak.fits'
    make_test_fits(str(path), with_streak=True)
    return path

@pytest.fixture(scope='session')
def sample_fits_no_streak(tmp_path_factory):
    """Synthetic FITS file with no streak."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from make_test_fits import make_test_fits
    path = tmp_path_factory.mktemp('fits') / 'test_no_streak.fits'
    make_test_fits(str(path), with_streak=False)
    return path

@pytest.fixture
def mock_tle_dict():
    """Real Starlink TLE for testing propagation."""
    return {
        'OBJECT_NAME': 'STARLINK-2183',
        'NORAD_CAT_ID': '48274',
        'TLE_LINE1': '1 48274U 21044AP  24093.10416667  .00005123  00000-0  35291-3 0  9999',
        'TLE_LINE2': '2 48274  53.0538 142.5671 0001423  89.4284 270.6936 15.06389548169829',
        'EPOCH': '2024-04-02T02:30:00',
    }

@pytest.fixture
def known_obs_time():
    """A specific obs_time for deterministic tests."""
    return datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)

@pytest.fixture
def luxembourg_observer():
    """Stellina telescope location (Luxembourg)."""
    return {'lat': 49.61, 'lon': 6.13, 'elev_m': 280.0}
```

---

## Running Tests

```bash
# All tests (325 passing, 15 integration tests auto-skipped)
pytest tests/ -v

# Single module
pytest tests/test_fits_parser.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Stop on first failure (useful during development)
pytest tests/ -x

# Run only tests matching a keyword
pytest tests/ -k "streak" -v

# Live Space-Track API tests (requires SPACETRACK_USER + SPACETRACK_PASS):
pytest tests/ -m integration -v
```

Recent API/FITS loader coverage to preserve:

- Same-stem `.wcs` sidecars are loaded when the FITS header lacks celestial WCS.
- API upload processing copies matching sidecars from local upload storage or
  known data locations before running the pipeline.
- `/api/result/{job_id}` includes `image_width` and `image_height` so frontend
  overlays can scale original pixel coordinates onto preview images.

---

## Baseline Metrics Collection — Week 4

The end-to-end test in `tests/test_end_to_end.py` must:
1. Run on all images in `results/confirmed_passes.json`
2. Record per-image results
3. Compute summary statistics
4. Write to `results/phase1_baseline.json`

### What "correct identification" means
- Correct NORAD ID appears in top-3 `CandidateMatch` results
- OR correct NORAD ID appears anywhere with `weighted_score > 0.3`

### Metrics Definitions

**Detection recall:**
```
recall = (images where ASTRiDE found ≥ 1 streak) / (total images with confirmed pass)
```

**False positive rate:**
```
fpr = (detections with no TLE match in top-3) / (total detections)
```

**Match rate:**
```
match_rate = (detections where correct NORAD in top-3) / (total detections)
```

**Position residual:**
```
For each correct match: angular_sep_arcsec between observed streak center and predicted position
Report: median, mean, 90th percentile
```

**Processing time:**
```
Wall clock from FITS open to final ranked candidates
Report: mean per image, total for batch
```

---

## Test Data Management

### Never commit large FITS files to git
Add to `.gitignore`:
```
data/GTImages/
data/frigate/
data/cache/
data/sample/*.fits
results/*.json
```

### What IS committed to git
```
data/sample/test_with_streak.fits.md5   ← checksum for verification
results/confirmed_passes.json            ← ground truth, small, commit this
results/phase1_baseline.json             ← once recorded, commit this
```

### Integration and real-data tests

`conftest.py` registers two markers and auto-skips them in normal `pytest` runs:

- `@pytest.mark.integration` — tests that call the live Space-Track API.
  Auto-skipped unless `-m integration` is passed.  Require `SPACETRACK_USER`
  and `SPACETRACK_PASS` in the environment (or a `.env` file at the project root).
- `@pytest.mark.real_data` — tests that read real FITS images from `tests/data/test/`.
  Auto-skipped unless `-m real_data` is passed.

```bash
# Normal run — no network calls, no real FITS needed:
pytest tests/ -v

# Live Space-Track API tests (requires credentials):
pytest tests/ -m integration -v

# Real-image tests (drop .fits files into tests/data/test/ first):
pytest tests/ -m real_data -v
```

The auto-skip is implemented as a `pytest_collection_modifyitems` hook in
`conftest.py` — no `skipif` decorators needed on individual tests.

---

## Logging Standard

All modules use the standard logging module. Set level in __main__ blocks:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)
```

Log levels:
- `DEBUG`: per-candidate scoring details, intermediate pixel coords
- `INFO`: per-image summary (N candidates found, best match, processing time)
- `WARNING`: missing optional FITS fields, TLE age > 48 hours, no match found
- `ERROR`: Space-Track query failure, SGP4 error code != 0, file not found
