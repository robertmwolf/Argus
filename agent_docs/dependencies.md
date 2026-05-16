# Dependencies

## Why Python 3.11?

Python 3.11 is the pinned version for this project. Do not upgrade to 3.12 or
later without thorough re-testing.

**Reasons:**

- **mmcv pre-built CUDA wheels** — OpenMMLab publishes wheels for Python 3.8–3.11.
  The cu121/cu128 index (used for mmcv 2.1.0) has no 3.12 wheels, meaning any
  upgrade to 3.12 forces a source build that requires a matching MSVC + CUDA
  dev environment (fatal on Windows; painful on Linux).
- **numpy < 2.0 constraint** — `sep` and `astride` ship compiled C extensions
  built against numpy 1.x ABI.  Python 3.12 wheels for those packages either
  don't exist or are untested against our pinned numpy 1.26.4.
- **Tested stack** — every package version in this file was verified together on
  Python 3.11.  Upgrading Python requires re-verifying the entire matrix.

Python 3.11 reaches end-of-life in October 2027, so this is not an urgent
upgrade.  When the project does upgrade, the blocker to resolve first is mmcv
pre-built wheel availability for the new Python + CUDA combination.

---

## What is OpenMIM?

`openmim` (installed via `pip install openmim`) is the official package manager
for the OpenMMLab ecosystem (mmcv, mmdet, mmseg, etc.).  After installing it you
get the `mim` CLI:

```bash
pip install -U openmim
mim install mmcv==2.1.0
```

`mim install` is smarter than plain `pip install mmcv` because it automatically
picks the correct pre-built wheel from OpenMMLab's CUDA-indexed wheel server
(`download.openmmlab.com/mmcv/dist/cuXXX/torchY.Z/`) for your exact
CUDA+PyTorch combination.  Plain `pip install mmcv` may pull the wrong variant.

**When does it fail?** When no pre-built wheel exists for your combination —
most commonly on native Windows (see the Windows section below).  In that case
`mim install` falls back to compiling from source, which requires a working
MSVC + CUDA C++ toolchain and often fails with CUDA/PyTorch header errors.

---

## Windows — mmcv CUDA ops

> **TL;DR: Do not try to build mmcv from source on native Windows.  Use WSL2.**

As of May 2026 there is **no pre-built Windows wheel** for:

```
Windows + Python 3.11 + PyTorch 2.6+/cu128 + CUDA 12.8
```

Both `pip install mmcv` and `mim install mmcv==2.1.0` will fall back to source
compilation, which fails inside the CUDA/PyTorch C++ headers even from a VS 2022
Developer shell.  `mmcv-lite` installs cleanly but lacks the compiled CUDA ops
(`mmcv._ext`), so Co-DINO training will fail at the first deformable conv op.

**Recommended path for Windows workstations:**

1. Install WSL2 with Ubuntu 22.04:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
2. Inside WSL2, run `scripts/cloud_setup.sh` exactly as documented.  Linux
   wheels for mmcv are available for cu128 and install without compilation.
3. Expose the GPU to WSL2: ensure `nvidia-smi` works inside WSL2 before
   proceeding (`nvidia-smi` should show your GPU; if not, update the WSL CUDA
   driver from https://developer.nvidia.com/cuda/wsl).

**Docker alternative:** The ARGUS `docker-compose.yml` uses a CUDA base image
and installs all Linux dependencies at build time — another clean path that
avoids the Windows wheel problem entirely.

---

## Install Order — Critical Rule

**PyTorch and the MMDetection stack must be installed before `requirements.txt`.**
The wheel index URL differs by target platform.  Installing in the wrong order
or from the wrong index silently installs the wrong mmcv variant, which breaks
Co-DINO training.

---

## Platform A — Mac (dev / MPS / CPU only)

```bash
conda create -n satid python=3.11 -y
conda activate satid

# PyTorch (MPS backend; CPU fallback auto-selected by get_device())
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# MMDetection stack — CPU-only build (no CUDA ops needed on Mac)
pip install mmengine==0.10.4
pip install mmcv==2.1.0           # CPU wheel from PyPI — no custom index needed
pip install mmdet==3.3.0

# All remaining dependencies
pip install -r requirements.txt
```

Set before running any training or inference on Mac:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## Platform B — Linux / Docker (Lambda A100, CUDA 12.1)

This is the path used by `docker/Dockerfile.worker` and the Lambda Labs A100
cloud instance.

```bash
conda create -n satid python=3.11 -y
conda activate satid

# PyTorch cu121
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# MMDetection stack — cu121 wheels
pip install mmengine==0.10.4
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
pip install mmdet==3.3.0

# All remaining dependencies
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Platform C — Workstation / WSL2 (RTX 5070 Ti, CUDA 12.8 / Blackwell)

RTX 5070 Ti is a Blackwell GPU — requires PyTorch ≥ 2.6.0 (first release with
Blackwell support).  Use `scripts/cloud_setup.sh` which handles version
detection and the mim fallback automatically.

Manual steps if running without `cloud_setup.sh`:

```bash
conda create -n satid python=3.11 -y
conda activate satid

# PyTorch (latest stable ≥ 2.6 from cu128 index)
pip install "torch>=2.6.0" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Detect installed version for wheel selection
TORCH_VER=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")

# MMDetection stack — mim auto-selects wheel for detected torch+cuda
pip install -U openmim
pip install mmengine==0.10.4
mim install "mmcv==2.1.0" || \
    pip install mmcv==2.1.0 \
        -f "https://download.openmmlab.com/mmcv/dist/cu128/torch${TORCH_VER}/index.html"
pip install mmdet==3.3.0

# Verify CUDA ops loaded
python -c "import mmcv.ops; print('mmcv CUDA ops OK')"

# All remaining dependencies
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

> **Native Windows:** Do not attempt this platform natively — see the
> "Windows — mmcv CUDA ops" section above.  Use WSL2 and treat it as
> Platform C.

---

## Quick Install (after cloning — non-GPU environments only)

For Mac dev or API-only containers that do not need the ML training stack:

```bash
conda create -n satid python=3.11 -y
conda activate satid
pip install -r requirements.txt   # does NOT install torch/mmdet
```

For GPU environments use the platform-specific steps above.

---

## Full Package List

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
