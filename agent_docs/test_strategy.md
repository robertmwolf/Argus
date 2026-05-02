# Test Strategy

## Philosophy
Every module has a pytest test file. Tests run after every module is built.
Baseline metrics are recorded as JSON — not printed and forgotten.
The numbers from Phase 1 become the comparison target for Phase 2.

---

## Test Structure

```
tests/
├── conftest.py              ← shared fixtures (sample FITS, mock TLEs)
├── test_fits_parser.py      ← Week 1
├── test_classical_detector.py  ← Week 2
├── test_plate_solver.py     ← Week 3
├── test_spacetrack_query.py ← Week 3
├── test_propagator.py       ← Week 4
├── test_matcher.py          ← Week 4
├── test_scorer.py           ← Week 4
└── test_end_to_end.py       ← Week 4 (integration)
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
# All tests
pytest tests/ -v

# Single module
pytest tests/test_fits_parser.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Stop on first failure (useful during development)
pytest tests/ -x

# Run only tests matching a keyword
pytest tests/ -k "streak" -v
```

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
data/milan/
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

### Generating sample FITS for CI
In CI environments without real FITS files, use synthetic fixtures only.
The conftest.py fixtures generate synthetic FITS in tmp dirs — no real data needed for unit tests.
Only `test_end_to_end.py` needs real FITS files and real Space-Track access.
Mark those tests to skip in CI:

```python
import pytest
import os

requires_real_data = pytest.mark.skipif(
    not os.path.exists('data/milan'),
    reason="Real FITS data not available"
)

requires_spacetrack = pytest.mark.skipif(
    'SPACETRACK_USER' not in os.environ,
    reason="Space-Track credentials not set"
)

@requires_real_data
@requires_spacetrack
def test_full_pipeline_on_real_image():
    ...
```

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
