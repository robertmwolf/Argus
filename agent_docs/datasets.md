# Datasets

## Overview
Three data sources support the ARGUS pipeline. GTImages is the primary
validation and negative-example source — it is already present in `data/GTImages/`.

---

## 1. GTImages — Primary Validation & Negative Source (Available Locally)

**What it is:** 759 FITS images of intentional satellite streak observations
captured by a fixed ground station (43.67°N, 81.02°W — Ontario, Canada) using
SkyTrack 1.9.8. Images are 6248×4176 pixels, 16-bit, 0.5 s exposures, GAIN=300,
with ASTAP plate solutions provided as `.wcs` / `.ini` sidecar files.

**Annotations:** 68 `.strk` files (one per tracked NORAD ID) containing
pixel-precise start/end coordinates of every streak, peak/mean SNR, streak
length, full TLE elements for the tracked satellite, and a reject flag.

**Key statistics:**
- 593 usable labeled streak images (reject=0)
- 93 real no-streak images (reject=−1) — valuable as negative training examples
- 68 unique NORAD IDs (79% Starlink; also Meteor-M2, Yaogan, Cosmos, Iridium)
- Streak lengths: median 624 px, p10=373 px, p90=1003 px (mostly long streaks)
- Single night (2026-04-27), single site — no sky-background diversity

**Role in ARGUS:**
- **Negative examples:** 93 no-streak images fill the gap in SatStreaks
- **Cross-ID benchmark:** every image has a known NORAD ID — run pipeline and
  check whether crossid.py recovers the correct satellite
- **Supplemental training:** fold 593 labeled images into training alongside
  SatStreaks, but do not replace SatStreaks (GTImages lacks short streaks and
  scene diversity)

**Convert to COCO JSON:**
```bash
python scripts/convert_gtimages.py \
    --strk-dir data/GTImages \
    --output data/annotations/gtimages.json \
    --negatives-output data/annotations/gtimages_negatives.json
```

**Location:** `data/GTImages/` — already present, no download needed.

---

## 2. Frigate Dataset (Staged — Partially Annotated)

**What it is:** Wide-field FITS images of LEO specifically collected for
satellite detection by ExoAnalytic Solutions, using a QHY600M camera
at 9600×6422 pixels, 0.5-second exposures. Raw and pre-processed versions
both released. This is purpose-built for exactly this pipeline.

**GitHub:** https://github.com/DanSRoll/frigate
**Paper:** https://www.nature.com/articles/s41597-025-06220-0

**Location:** `/Volumes/External/frigate/` (raw FITS + processed PNGs at 2325×1555)

**Annotation status — ✅ ready for training inclusion:**
- 350 frames manually reviewed via `scripts/annotate_frigate_streaks.py`
- 191 frames contain satellite streaks (377 OBBs total) — all **very short streaks**
  (~20–80 px), filling a morphology gap absent from GTImages and SatStreaks
- 159 frames confirmed streak-free (explicit negatives)
- Frames are scattered across the full observation sequence (priority-list ordering),
  providing diversity across sky conditions throughout the night
- Annotations: `data/annotations/frigate_streaks.json`
- Remaining 1,630 frames: unreviewed — do NOT include in training

**To include in a training run:**
```bash
python scripts/merge_annotations.py --seed 42 --val-fraction 0.2 --include-frigate
```

**Streak characteristics:**
- Short streaks only (~20–80 px at 2325×1555 px scale)
- Single site, single night — limited sky-background diversity
- QHY600M camera (different sensor/optics from GTImages Stellina and HST SatStreaks)

---

## 3. Space-Track TLE Catalog (Required — Register Now)

**What it is:** Authoritative US Space Force satellite catalog.
Over 220 million historical TLE sets. Free public access.
This is the source of all TLE data for matching.

**Register:** https://www.space-track.org/auth/createAccount

**Important:** Read and agree to the terms of service.
Do not redistribute raw data. Do not make automated requests
more than once per 3 seconds (enforced rate limit).

**Set credentials as environment variables — never hardcode:**
```bash
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
export ARGUS_ENV=production   # use real site; omit = test site
```

**Bootstrap historical coverage (recommended path):**
```bash
# Download the annual bundle covering the last 3 months:
python scripts/download_tle_bundle.py

# See what's available first:
python scripts/download_tle_bundle.py --list

# Specific year (e.g. current partial year):
python scripts/download_tle_bundle.py --year 2026
```

`download_tle_bundle.py` uses the Space-Track **fileshare** API to discover and
stream-download annual zip bundles, then ingests them via `bootstrap_tle_catalog`.
Each annual bundle is ~2.7 GB for a complete past year; the current partial year
is smaller and grows over time. The script is idempotent.

**Manual fallback** if the fileshare API is unavailable:
```
https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu
```
Download to `data/tle_zips/` then:
```bash
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2026
```

**GP_History API — NOT used for bootstrap:**
Space-Track prohibits broad `gp_history` queries (no NORAD filter + large date range).
Use the annual zip bundles instead. See `agent_docs/spacetrack.md` for the full policy.

**Test your access:**
```python
from spacetrack import SpaceTrackClient
import os

st = SpaceTrackClient(
    identity=os.environ['SPACETRACK_USER'],
    password=os.environ['SPACETRACK_PASS']
)
result = st.gp(norad_cat_id=25544, format='json')  # ISS
print(result[0]['OBJECT_NAME'])  # Should print: ISS (ZARYA)
```

---

## 4. SatStreaks Dataset (Annotated, for DINO/YOLO Training)

**What it is:** 3,073 densely annotated real images of satellite streaks
from Hubble Space Telescope (114,607 images scanned via citizen science)
and NASA Satellite Streak Watcher project (233 ground-based images).
Includes processed PNG/JPEG images and segmentation masks.

**GitHub:** https://github.com/jijup/SatStreaks
**Paper:** CRV 2024 — "SatStreaks: Towards Supervised Learning for
Delineating Satellite Streaks from Astronomical Images"

**Note:** These are not FITS files — they are processed PNG/JPEG.
Use for DINO/YOLO training only, not Phase 1 FITS parsing.

ARGUS converts SatStreaks masks into detector annotations during split merge:

```bash
python scripts/merge_annotations.py --seed 42 --val-fraction 0.2
```

The merge script reads each image size, computes the foreground mask bounding
box and pixel area, and writes real COCO `bbox` values into
`data/annotations/train.json`, `val.json`, and `test.json`. Entries with
missing or empty masks are skipped. GTImages labeled and negative examples are
merged into the train/validation pool unless `--satstreaks-only` is supplied.

---

## Combined Training Splits

Current training split generation uses:

```bash
python scripts/convert_gtimages.py \
    --strk-dir data/GTImages \
    --output data/annotations/gtimages.json \
    --negatives-output data/annotations/gtimages_negatives.json

python scripts/merge_annotations.py --seed 42 --val-fraction 0.2
```

Outputs:

```text
data/annotations/train.json
data/annotations/val.json
data/annotations/test.json
```

SatStreaks keeps its provided train/val/test split. GTImages labeled and
negative images are shuffled with the configured seed and split according to
`--val-fraction`; GTImages does not enter the held-out SatStreaks test split.
The optional handoff manifest at `data/Manifest.txt` records the staged dataset
counts used for external workstation training.

---

## Sample Data for Testing

Use synthetic FITS files to test the parser without real images:

**Any public FITS file from NASA/STScI works for parser testing:**
```bash
# Download a small sample FITS from the HST archive (no account needed)
wget -O data/sample/test_hst.fits \
  "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/j8pu0y010_drz.fits"

# Or generate a synthetic FITS with known properties:
python scripts/make_test_fits.py  # (Claude Code: create this script)
```

**Synthetic FITS generator (Claude Code: create this as scripts/make_test_fits.py):**
```python
"""Generate minimal synthetic FITS files for parser testing."""
import numpy as np
from astropy.io import fits
from datetime import datetime, timezone
from pathlib import Path

def make_test_fits(output_path: str, with_streak: bool = True):
    # Create realistic background (stars + noise)
    rng = np.random.default_rng(42)
    image = rng.poisson(100, size=(2080, 3096)).astype(np.uint16)

    # Add fake stars
    for _ in range(200):
        x, y = rng.integers(10, 3086), rng.integers(10, 2070)
        brightness = rng.integers(500, 5000)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                r2 = dx**2 + dy**2
                image[y+dy, x+dx] += int(brightness * np.exp(-r2/2))

    # Add fake streak if requested
    if with_streak:
        for i in range(800):
            x = 200 + i
            y = 500 + int(i * 0.3)
            if 0 <= x < 3096 and 0 <= y < 2080:
                image[y, x] = 8000
                if y+1 < 2080: image[y+1, x] = 4000

    hdu = fits.PrimaryHDU(image)
    hdu.header['DATE-OBS'] = '2024-04-02T02:55:24.383'
    hdu.header['NAXIS1']   = 3096
    hdu.header['NAXIS2']   = 2080
    hdu.header['EXPTIME']  = 10.0
    hdu.header['PIXSCALE'] = 1.36      # arcsec/pixel (Stellina)
    hdu.header['SITELAT']  = 49.61
    hdu.header['SITELONG'] = 6.13
    hdu.header['SITEELEV'] = 280.0
    hdu.header['CRVAL1']   = 83.82     # RA  (Orion region)
    hdu.header['CRVAL2']   = -5.39     # Dec
    hdu.header['CRPIX1']   = 1548.0
    hdu.header['CRPIX2']   = 1040.0
    hdu.header['CD1_1']    = -0.000378  # ~1.36 arcsec/px in degrees
    hdu.header['CD1_2']    = 0.0
    hdu.header['CD2_1']    = 0.0
    hdu.header['CD2_2']    = 0.000378
    hdu.header['CTYPE1']   = 'RA---TAN'
    hdu.header['CTYPE2']   = 'DEC--TAN'

    hdu.writeto(output_path, overwrite=True)
    print(f"Written: {output_path}")

if __name__ == '__main__':
    Path('data/sample').mkdir(parents=True, exist_ok=True)
    make_test_fits('data/sample/test_with_streak.fits', with_streak=True)
    make_test_fits('data/sample/test_no_streak.fits',   with_streak=False)
```

---

## Cross-ID Ground Truth (GTImages)

GTImages provides ready-made cross-ID ground truth — every image has a known
NORAD ID and embedded TLE.  The converter (`scripts/convert_gtimages.py`) writes
`data/annotations/gtimages.json` which the eval benchmark can use directly.

To run a cross-ID accuracy check against GTImages:
```bash
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/gtimages.json \
    --output results/gtimages_crossid.json
```

The benchmark reports whether the correct NORAD ID appears in the top-3
`CandidateMatch` results for each image.
