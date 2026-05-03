# Dependencies

## Environment Setup

```bash
# Create conda environment
conda create -n satid python=3.11 -y
conda activate satid

# Core astronomy (Phase 0 + Phase 3)
pip install astropy==6.1.0
pip install photutils==1.13.0
pip install sep==1.2.1            # fast background estimation
pip install astride==0.3.2        # ASTRiDE streak detection (Phase 0 baseline)
pip install sgp4==2.23            # SGP4 propagation
pip install skyfield==1.49        # high-level orbital + coordinate transforms
pip install spacetrack==0.14.0    # Space-Track API client

# ML stack (Phase 2 — Co-DINO)
# Install PyTorch first, then MMDet dependencies in order
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmengine==0.10.4
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
pip install mmdet==3.3.0
pip install ultralytics==8.2.0    # YOLO11-OBB baseline

# Image processing
pip install opencv-python-headless==4.10.0.84
pip install scikit-image==0.24.0  # Radon transform for angle refinement
pip install albumentations==1.4.3 # augmentation pipeline
pip install Pillow==10.4.0
pip install shapely==2.0.4        # rotated IoU via polygon intersection

# Data / compute
pip install numpy==1.26.4
pip install scipy==1.13.1
pip install pandas==2.2.2

# API + database (Phase 5)
pip install fastapi==0.111.0
pip install uvicorn[standard]==0.29.0
pip install sqlalchemy[asyncio]==2.0.30
pip install asyncpg==0.29.0       # PostgreSQL async driver
pip install aiosqlite==0.20.0     # SQLite async driver (default for local)
pip install pydantic==2.7.0
pip install python-multipart==0.0.9
pip install boto3==1.34.0         # S3 storage backend
pip install python-dotenv==1.0.1

# Utilities
pip install python-dateutil==2.9.0
pip install tqdm==4.66.4
pip install diskcache==5.6.3      # on-disk caching for Space-Track results
pip install click==8.1.7
pip install requests==2.32.0      # TLE catalog refresh

# Testing
pip install pytest==8.2.2
pip install pytest-cov==5.0.0
pip install httpx==0.27.0         # async test client for FastAPI

# Optional but useful
pip install matplotlib==3.9.0
pip install zenodo_get==1.6.1     # Zenodo dataset downloads

# Save environment
pip freeze > requirements.txt
```

## Quick Install (after cloning repo)
```bash
conda create -n satid python=3.11 -y
conda activate satid
pip install -r requirements.txt
```

## Separate requirements files

Keep two files:
- `requirements.txt` — full stack including torch + mmdet (for worker container)
- `requirements-api.txt` — API-only, no torch or mmdet (for api container, faster build)

`requirements-api.txt`:
```
astropy>=6.0.0
sgp4>=2.22
skyfield>=1.49
spacetrack>=0.14.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
sqlalchemy[asyncio]>=2.0.0
asyncpg>=0.29.0
aiosqlite>=0.20.0
pydantic>=2.6.0
python-multipart>=0.0.9
boto3>=1.34.0
python-dotenv>=1.0.0
requests>=2.31.0
```

---

## Key Library Notes

### ASTRiDE
Classical streak detector. Works directly on FITS data.
Main class: `astride.Streak(filepath, contour_threshold=3.0)`

```python
from astride import Streak
streak = Streak('image.fits', contour_threshold=3.0)
streak.detect()
# streak.streaks is a list of dicts with keys:
# x_start, y_start, x_end, y_end, slope, intercept,
# shape_factor, area, length, connectivity
```

**Known limitation:** Expects a file path, not an array.
If passing pre-processed data, write to a temp FITS file first.
ASTRiDE also has its own background subtraction — disable ours or
coordinate to avoid double-subtracting.

### sgp4
Low-level SGP4 propagator. Use `Satrec` (modern API, not `twoline2rv`).

```python
from sgp4.api import Satrec, jday

sat = Satrec.twoline2rv(tle_line1, tle_line2)
jd, fr = jday(year, month, day, hour, minute, second)
error_code, position_km, velocity_km_s = sat.sgp4(jd, fr)
# error_code == 0 means success
# position_km is in TEME frame (km from Earth center)
```

### skyfield
Use for coordinate transforms (TEME → topocentric RA/Dec).
Do NOT use skyfield's EarthSatellite for propagation — use sgp4 directly.
Use skyfield only for the TEME→observer conversion.

```python
from skyfield.api import load, wgs84
from skyfield.positionlib import TEME, Distance, Velocity
from skyfield.units import km_per_au

ts = load.timescale()
# Convert sgp4 TEME output to topocentric:
observer = wgs84.latlon(lat_deg, lon_deg, elevation_m=elev_m)
# See propagator.py for full implementation
```

### spacetrack
Space-Track API Python client.

```python
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
import os

st = SpaceTrackClient(
    identity=os.environ['SPACETRACK_USER'],
    password=os.environ['SPACETRACK_PASS']
)

# GP_History query with epoch range:
results = st.gp_history(
    epoch=op.inclusive_range('2024-04-01', '2024-04-07'),
    orderby='epoch desc',
    format='json'
)
```

**Rate limit:** Max 30 requests/minute. The client handles this automatically
if you use `iter_lines=True` for large queries.

### diskcache
Use for caching Space-Track results. Much simpler than Redis/Memcached.

```python
import diskcache as dc

cache = dc.Cache('data/cache')

# Store with TTL:
cache.set(cache_key, data, expire=48*3600)  # 48 hour TTL

# Retrieve:
data = cache.get(cache_key)  # Returns None if expired or missing
```

---

## Optional: astrometry.net (for plate solving without WCS headers)

MILAN FITS files from Stellina usually have valid WCS headers.
Frigate files may not. If WCS is missing, you need astrometry.net.

**Install astrometry.net (local):**
```bash
# macOS
brew install astrometry-net

# Ubuntu/Debian
sudo apt-get install astrometry.net

# Download index files (needed for solving, choose based on your FOV):
# For MILAN (~1° FOV): download index-4107 through index-4119
wget -P /usr/share/astrometry/ \
  http://data.astrometry.net/4100/index-4107.fits
```

**Python wrapper:**
```python
import subprocess

def solve_field(fits_path: str, timeout_sec: int = 60) -> bool:
    """Run astrometry.net plate solver. Returns True if solved."""
    result = subprocess.run([
        'solve-field', fits_path,
        '--no-plots', '--overwrite',
        '--downsample', '2',
        '--scale-units', 'arcsecperpix',
        '--scale-low', '1.0', '--scale-high', '2.0',
    ], capture_output=True, timeout=timeout_sec)
    return result.returncode == 0
```

---

## Environment Verification Script

After install, run this to verify everything is working:

```bash
python scripts/verify_environment.py
```

**Claude Code: create scripts/verify_environment.py with these checks:**

```python
"""Verify all dependencies are installed and working."""
import sys

checks = []

def check(name, fn):
    try:
        fn()
        checks.append((name, "✅ OK"))
    except Exception as e:
        checks.append((name, f"❌ FAIL: {e}"))

check("astropy",    lambda: __import__('astropy'))
check("astride",    lambda: __import__('astride'))
check("sgp4",       lambda: __import__('sgp4'))
check("skyfield",   lambda: __import__('skyfield'))
check("spacetrack", lambda: __import__('spacetrack'))
check("sep",        lambda: __import__('sep'))
check("opencv",     lambda: __import__('cv2'))
check("diskcache",  lambda: __import__('diskcache'))
check("spacetrack_env", lambda: (
    __import__('os').environ['SPACETRACK_USER'],
    __import__('os').environ['SPACETRACK_PASS']
))

for name, status in checks:
    print(f"{status}  {name}")

if any("FAIL" in s for _, s in checks):
    sys.exit(1)
else:
    print("\nAll checks passed. Ready to start Phase 1.")
```
