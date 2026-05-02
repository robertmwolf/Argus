# Datasets

## Overview
Three data sources are needed for Phase 1. Start with MILAN — it is
available immediately with no account required.

---

## 1. MILAN Sky Survey — Start Here (Available Now)

**What it is:** 50,068 raw FITS images from a Stellina automated telescope,
captured over 12 months (April 2022 – early 2023) during deep sky sessions.
Satellite streaks appear incidentally during observations.
Each image is 10-second exposure, ~3096×2080 pixels, 16-bit.

**Why it's good for Phase 1:** Real noise, real PSF, real artefacts.
The FITS headers contain valid DATE-OBS timestamps which are essential
for Space-Track queries. Many images contain Starlink passes that can
be confirmed against GP_History.

**Download (one month at a time — start with August 2022):**
```
August 2022:    https://zenodo.org/records/7049839
September 2022: https://zenodo.org/records/7115518
October 2022:   https://zenodo.org/records/7399412   (actually November)
November 2022:  https://zenodo.org/records/7399412
```
Each monthly archive is a ZIP of FITS files, roughly 2–5 GB per month.

**How to download programmatically:**
```bash
# Install zenodo_get for easy Zenodo downloads
pip install zenodo_get

# Download August 2022 (record 7049839)
zenodo_get 7049839 -o data/milan/2022-08/

# Or direct wget (find URLs on the Zenodo page)
wget https://zenodo.org/records/7049839/files/MILAN_2022-08.zip
unzip MILAN_2022-08.zip -d data/milan/2022-08/
```

**Cite as:**
Parisot, O. et al. (2023). MILAN Sky Survey, a dataset of raw deep sky
images captured during one year with a Stellina automated telescope.
Data in Brief, 48, 109133.

---

## 2. Frigate Dataset (Request from Authors)

**What it is:** Wide-field FITS images of LEO specifically collected for
satellite detection by ExoAnalytic Solutions, using a QHY600M camera
at 9600×6422 pixels, 0.5-second exposures. Raw and pre-processed versions
both released. This is purpose-built for exactly this pipeline.

**Status:** Paper published, code on GitHub, raw FITS download link
marked "TBC" (to be confirmed) in the repo. Email authors to request
early access.

**GitHub:** https://github.com/DanSRoll/frigate
**Paper:** https://www.nature.com/articles/s41597-025-06220-0

**Email to send:**
```
To: [corresponding author from paper]
Subject: Request for Frigate dataset access

I am building an automated satellite identification pipeline using
ASTRiDE + SGP4 matching for Phase 1, and would like to use the
Frigate dataset for testing and validation. Could you share
download access to the FITS files? Happy to share any results
back with your team.
```

**When available, download to:** `data/frigate/raw/` and `data/frigate/processed/`

---

## 3. Space-Track GP_History (Required — Register Now)

**What it is:** Authoritative US Space Force satellite catalog.
Over 138 million historical TLE sets. Free public access.
This is the source of all TLE data for matching.

**Register:** https://www.space-track.org/auth/createAccount

**Important:** Read and agree to the terms of service.
Do not redistribute raw data. Do not make automated requests
more than once per 3 seconds (enforced rate limit).

**Set credentials as environment variables — never hardcode:**
```bash
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword

# Add to ~/.bashrc or ~/.zshrc to persist:
echo 'export SPACETRACK_USER=your@email.com' >> ~/.bashrc
echo 'export SPACETRACK_PASS=yourpassword' >> ~/.bashrc
```

**Test your access:**
```python
from spacetrack import SpaceTrackClient
import os

st = SpaceTrackClient(
    identity=os.environ['SPACETRACK_USER'],
    password=os.environ['SPACETRACK_PASS']
)
# Should return data without error:
result = st.gp(norad_cat_id=25544, format='json')  # ISS
print(result[0]['OBJECT_NAME'])  # Should print: ISS (ZARYA)
```

---

## 4. SatStreaks Dataset (Phase 2 — Annotated, for YOLO Training)

**What it is:** 3,073 densely annotated real images of satellite streaks
from Hubble Space Telescope (114,607 images scanned via citizen science)
and NASA Satellite Streak Watcher project (233 ground-based images).
Includes segmentation masks and bounding boxes.

**GitHub:** https://github.com/jijup/SatStreaks
**Paper:** CRV 2024 — "SatStreaks: Towards Supervised Learning for
Delineating Satellite Streaks from Astronomical Images"

**Note:** These are not FITS files — they are processed PNG/JPEG.
Use for Phase 2 YOLO-OBB annotation training only, not Phase 1.

---

## Sample Data for Testing (Phase 1 Day 1)

Before MILAN downloads complete, use these small FITS files to test
the parser immediately:

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

## Confirmed Test Cases (Ground Truth)

These are MILAN images where a Starlink pass has been manually
confirmed and the NORAD ID identified via Heavens-Above.com.
Use these as your Week 3–4 validation set.

**To build your own confirmed test cases:**
1. Pick a MILAN FITS file with a visible streak
2. Note DATE-OBS and observer location (Luxembourg: 49.61°N, 6.13°E)
3. Go to https://heavens-above.com → enter location → Starlink passes
4. Match the time and direction to confirm the NORAD ID
5. Record in results/confirmed_passes.json

**Format:**
```json
[
  {
    "fits_file": "data/milan/2022-08/frame_001234.fits",
    "obs_time_utc": "2022-08-15T21:43:12.4",
    "confirmed_norad_id": 48274,
    "object_name": "STARLINK-2183",
    "confirmed_by": "heavens-above.com cross-reference",
    "streak_visible": true
  }
]
```
