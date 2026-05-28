# Datasets

## Overview
Four data sources support the ARGUS pipeline. BrentImages (which subsumes the
original GTImages night) is the primary ground-truth and ongoing capture source.
All BrentImages data lives on the external drive under
`/Volumes/External/TrainingData/raw/BrentImages/`.

---

## 1. BrentImages — Ongoing Ground-Truth Capture Source

**What it is:** FITS images of intentional satellite streak passes captured from
a fixed private observatory at Atwood, Ontario (43.6735556°N, 81.0204722°W,
365 m). Camera: ZWO ASI2600MM Pro, 6248×4176 pixels, 16-bit, 0.5 s exposures,
Lum filter, dark + flat calibrated. SkyTrack 1.9.8 schedules passes and writes
rich FITS headers including NORAD ID, full TLE elements, object name, and
atmospheric data. New capture nights are added as additional subdirectories.

**Naming convention:** Each night is a subdirectory named `Img_YYYYMMDD_Atwood/`
containing `Streak_NORADID_HHMMSS.fits` files. Unlike the original GTImages
night, BrentImages FITS files do **not** include ASTAP `.wcs`/`.ini` plate
solution sidecars.

**Location:** `/Volumes/External/TrainingData/raw/BrentImages/`

### Current nights

| Directory | Date | Frames | NORAD IDs | Annotation status |
|---|---|---|---|---|
| `Img_20260412_Atwood/` | 2026-04-12 | 759 | 68 | ✅ Fully annotated (.strk + .ini/.wcs sidecars, GAIN=300) |
| `Img_20260515_Atwood/` | 2026-05-15 | 300 | 39 | ✅ Annotated — 204 streak images + 27 negatives (`brentimages_20260515.json`, `brentimages_20260515_negatives.json`) |

The `Img_20260412_Atwood/` night is the original GTImages dataset — it has
ASTAP plate solutions and fully pixel-annotated `.strk` files from SkyTrack.
Subsequent nights (starting with `Img_20260515_Atwood/`) have rich FITS
headers but no plate solutions; `.strk` stubs are generated from headers and
pixel coordinates are filled in via the manual annotation workflow.

**Key statistics (Img_20260412_Atwood — fully annotated):**
- 593 usable labeled streak images (reject=0)
- 93 real no-streak images (reject=−1) — valuable negative training examples
- 68 unique NORAD IDs (79% Starlink; also Meteor-M2, Yaogan, Cosmos, Iridium)
- Streak lengths: median 624 px, p10=373 px, p90=1003 px (mostly long streaks)

**Key statistics (Img_20260515_Atwood — fully annotated):**
- 204 streak images, 204 annotations (`brentimages_20260515.json`)
- 27 confirmed no-streak images (`brentimages_20260515_negatives.json`)
- 39 unique NORAD IDs
- Streak lengths: mean 725 px, median 687 px, min 215 px, max 1404 px (long + medium)
- **Not yet merged into any training split** — available as a clean holdout for out-of-distribution evaluation or future training expansion

**Role in ARGUS:**
- **Negative examples:** no-streak images fill the negative-example gap in SatStreaks
- **Cross-ID benchmark:** every annotated image has a known NORAD ID
- **Supplemental training:** fold labeled images alongside SatStreaks; BrentImages
  alone lacks short streaks and scene diversity

### Workflow for a new Atwood capture night (same scope)

```bash
# 1. Generate .strk stubs from FITS headers (one per NORAD ID, Reject=2 pending):
python scripts/generate_brentimages_strk.py \
    --night-dir /Volumes/External/TrainingData/raw/BrentImages/Img_YYYYMMDD_Atwood

# 2. Manually annotate pixel coordinates using the annotation tool, which flips
#    Reject to 0 (streak present) or -1 (no streak).

# 3. Convert annotated night to COCO JSON:
python scripts/convert_gtimages.py \
    --strk-dir /Volumes/External/TrainingData/raw/BrentImages/Img_YYYYMMDD_Atwood \
    --output data/annotations/brentimages_YYYYMMDD.json \
    --negatives-output data/annotations/brentimages_YYYYMMDD_negatives.json

# 4. Add the new session to the manifest:
#    Edit data/sessions/manifest.yaml — add an entry with split: train
#    (copy the atwood_20260515 entry as a template)

# 5. Rebuild the canonical training JSON:
python scripts/build_training_json.py \
    --output data/annotations/all_train_nodm_vN.json
```

**Notes on `generate_brentimages_strk.py`:**
- Reads NORAD ID, TLE elements, site info, and DATE-OBS directly from FITS headers
- GPBSTAR header uses a non-standard value format; the script calls
  `hdul.verify('silentfix')` to handle this transparently
- All OBS rows are written with Reject=2 so `convert_gtimages.py` skips them
  until pixel coordinates are annotated
- Re-running with `--force` overwrites existing stubs

**Convert fully-annotated night to COCO JSON** (same command for all nights):
```bash
python scripts/convert_gtimages.py \
    --strk-dir /Volumes/External/TrainingData/raw/BrentImages/Img_20260412_Atwood \
    --output data/annotations/gtimages.json \
    --negatives-output data/annotations/gtimages_negatives.json
```

---

## 2. Frigate Dataset (Staged — Partially Annotated)

**What it is:** Wide-field FITS images of LEO specifically collected for
satellite detection by ExoAnalytic Solutions, using a QHY600M camera
at 9600×6422 pixels, 0.5-second exposures. Raw and pre-processed versions
both released. This is purpose-built for exactly this pipeline.

**GitHub:** https://github.com/DanSRoll/frigate
**Paper:** https://www.nature.com/articles/s41597-025-06220-0

**Location:** `/Volumes/External/TrainingData/raw/frigate/` (raw FITS + processed PNGs at 2325×1555)

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

### Canonical training set — `all_train_nodm.json`

Built by `scripts/build_all_train_json.py`. Contains no DarkMatters data.

| Source | Images | Annotations | Notes |
|---|---|---|---|
| SatStreaks | 2,488 | 2,488 | JPEG/PNG, space + ground scopes |
| BrentImages Night 1 (Img_20260412_Atwood) | ~535 | ~469 | FITS, Atwood Night 1 |
| BrentImages Night 2 (Img_20260515_Atwood) | 231 | 204 | FITS, Atwood Night 2 |
| Frigate tiled crops | ~717 | ~655 | Very short streaks |
| **Total** | **~3,971** | **~3,816** | — |

Val split: `val.json` (SatStreaks + BrentImages Night 1)
Test split: `test.json` (308 images — SatStreaks + BrentImages Night 1 only, held-out)

### Base split — `train.json` / `val.json` / `test.json`

Original SatStreaks + BrentImages Night 1 split, used as the evaluation
reference and as a component of `all_train_nodm.json`:

```text
train.json  — 3,023 images, 2,957 annotations
val.json    —   411 images,   386 annotations
test.json   —   308 images,   308 annotations
```

Rebuild with:

```bash
# Convert the fully-annotated BrentImages Night 1:
python scripts/convert_gtimages.py \
    --strk-dir /Volumes/External/TrainingData/raw/BrentImages/Img_20260412_Atwood \
    --output data/annotations/gtimages.json \
    --negatives-output data/annotations/gtimages_negatives.json

python scripts/merge_annotations.py --seed 42 --val-fraction 0.2
```

### Available but not yet in any split

| Dataset | File(s) | Images | Annotations | Notes |
|---|---|---|---|---|
| BrentImages Night 2 | `brentimages_20260515.json` + `_negatives.json` | 231 | 204 streaks | Clean holdout; long + medium streaks |
| Frigate (streaks) | `frigate_streaks.json` | 350 | 377 | Very short ~20–80 px streaks (only source) |
| Frigate (negatives) | `frigate_negatives.json` | 300 | 0 | — |

To include Frigate in a training run:
```bash
python scripts/merge_annotations.py --seed 42 --val-fraction 0.2 --include-frigate
```

SatStreaks keeps its provided train/val/test split. GTImages/BrentImages labeled
and negative images are shuffled with the configured seed and split according to
`--val-fraction`; BrentImages does not enter the held-out SatStreaks test split.
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

## Cross-ID Ground Truth (BrentImages)

Every fully-annotated BrentImages frame has a known NORAD ID embedded in the
FITS header and confirmed by the `.strk` annotation.  The converter writes
`data/annotations/gtimages.json` which the eval benchmark uses directly.

To run a cross-ID accuracy check:
```bash
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/gtimages.json \
    --output results/gtimages_crossid.json
```

The benchmark reports whether the correct NORAD ID appears in the top-3
`CandidateMatch` results for each image.  As additional BrentImages nights are
annotated and merged, re-run the benchmark against the updated annotation file.

---

## Adding a New Telescope

When images arrive from a scope other than Atwood (different sensor, optics, or
site), follow this workflow.  The key rule: **Night 1 from any new scope is
always evaluated zero-shot before any training decision is made.**

### Step 1 — Annotate Night 1

Use the same pipeline as a new Atwood night:

```bash
# Generate .strk stubs (if FITS headers contain NORAD IDs):
python scripts/generate_brentimages_strk.py \
    --night-dir /path/to/new_scope/Img_YYYYMMDD

# Or annotate from scratch via the annotation tool.

# Convert to COCO JSON:
python scripts/convert_gtimages.py \
    --strk-dir /path/to/new_scope/Img_YYYYMMDD \
    --output data/annotations/newscope_YYYYMMDD.json \
    --negatives-output data/annotations/newscope_YYYYMMDD_negatives.json
```

### Step 2 — Zero-shot evaluation (DO THIS BEFORE TRAINING)

Run the current production model against Night 1 without any fine-tuning:

```bash
python scripts/zero_shot_eval.py \
    --annotation data/annotations/newscope_YYYYMMDD.json \
    --negatives data/annotations/newscope_YYYYMMDD_negatives.json \
    --raw-dir /path/to/new_scope/Img_YYYYMMDD \
    --scope newscope \
    --label "New Scope Night 1 (zero-shot)"
```

The script prints a **RECOMMENDATION** based on long-band recall:

| Long-band recall | Recommendation |
|---|---|
| ≥ 80% | Fine-tuning optional — fold into next scheduled retrain |
| 60–80% | Fine-tune advised — see Step 3B |
| < 60% | Investigate domain shift before training — see Step 3C |

Results are saved to `results/zero_shot_newscope_YYYYMMDD_HHMMSS/`.

### Step 3A — No fine-tuning needed (recall ≥ 80%)

Add the scope to the manifest with `split: holdout` for Night 1 (eval only).
As additional nights accumulate, promote them to `split: train` and rebuild
the training JSON for the next full retrain:

```bash
# Edit data/sessions/manifest.yaml: set split: train for Night 2+
python scripts/build_training_json.py \
    --output data/annotations/all_train_nodm_vN.json
```

### Step 3B — Fine-tune (recall 60–80%)

```bash
# 1. Annotate Night 2+ (target: ≥200 images).
# 2. Add to manifest (split: train) with an appropriate mix_weight.
#    A 1:1 ratio with existing Atwood data is a safe starting point.
#    Example: if Atwood has ~4000 images and new scope has 200,
#    set mix_weight: 20.0 to reach parity (200 × 20 = 4000).

# 3. Build the fine-tune training JSON:
python scripts/build_training_json.py \
    --mix-ratio newscope:20.0 \
    --output data/annotations/all_train_ft_newscope.json

# 4. Run fine-tune (~8 epochs, ~3–4h on Mac M3):
PYTORCH_ENABLE_MPS_FALLBACK=1 \
USE_DEV_SUBSET=false \
TRAIN_ANN_FILE=annotations/all_train_ft_newscope.json \
VAL_ANN_FILE=annotations/val.json \
ARGUS_NORM=autostretch \
caffeinate -i \
python -m training.train_dino \
    --config models/dino/streak_dinov3_vitb_400px_ft.py \
    --work-dir weights/run_ft_newscope \
    --max-epochs 8 \
    --val-interval 1 \
    --checkpoint-interval 1

# 5. Evaluate on BOTH new scope Night 1 AND standard test set:
python scripts/evaluate_comprehensive.py \
    --checkpoint weights/run_ft_newscope/best.pth \
    --config models/dino/streak_dinov3_vitb_400px_ft.py \
    --sets test_standard

python scripts/zero_shot_eval.py \
    --annotation data/annotations/newscope_YYYYMMDD.json \
    --raw-dir /path/to/new_scope/Img_YYYYMMDD \
    --scope newscope \
    --checkpoint weights/run_ft_newscope/best.pth \
    --config models/dino/streak_dinov3_vitb_400px_ft.py
```

**Accept the fine-tune only if:**
- New scope long-band recall improves by ≥ 5pp vs zero-shot baseline, AND
- Standard test set recall does **not** drop by more than 2pp vs Run 3 baseline (83.8%)

### Step 3C — Investigate domain shift (recall < 60%)

Before fine-tuning, diagnose the cause of the gap:

1. **Pixel scale**: compute `206265 × pixel_size_mm / focal_length_mm` (arcsec/px).
   Atwood baseline is 1.27 arcsec/px.  A significantly different scale means
   streak diagonal lengths will fall in different bands — verify that your GT
   annotations span the expected diagonal range at 400px tile resolution.

2. **Normalisation**: check that `apply_norm()` (in `inference/fits_loader.py`)
   stretches the new scope's images to the same 0–255 range as training data.
   Inspect a sample tile: if it appears very dark or very bright after
   normalisation, the auto-stretch parameters may need tuning.

3. **Anchor coverage**: DINO uses 300 learned queries, not hand-crafted anchors,
   so scale mismatch is less critical than in classic detectors — but if streaks
   at 400px tile resolution are systematically shorter or longer than in the
   training set, adding a data augmentation step (resize + crop) may help.

4. Once the root cause is understood, consider a full retrain with a new manifest
   version rather than fine-tuning.

### Session manifest entry for a new scope

Add to `data/sessions/manifest.yaml`:

```yaml
- session_id: newscope_YYYYMMDD
  scope_id: newscope               # short stable ID for --mix-ratio flags
  source_type: brentimages         # or public_dataset / synthetic
  date: "YYYY-MM-DD"
  split: holdout                   # Night 1: zero-shot eval only
  mix_weight: 1.0                  # will be set when promoted to train
  annotation_file: "data/annotations/newscope_YYYYMMDD.json"
  negatives_file: "data/annotations/newscope_YYYYMMDD_negatives.json"
  raw_dir: "/path/to/new_scope/Img_YYYYMMDD"
  n_images: 0                      # fill in after annotation
  n_annotations: 0
  camera: "Sensor @ focal_length mm"
  pixel_scale_arcsec: 0.0          # 206265 * pixel_size_mm / focal_length_mm
  focal_length_mm: 0
  exposure_s: 0.5
  filter: "Lum"
  norad_ids: 0
  notes: "Night 1 — run zero_shot_eval.py before promoting to train"
```

Promote to `split: train` and set `mix_weight` once the zero-shot evaluation
and annotation of Night 2+ are complete.
