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

## Dependency Freshness Policy

Last reviewed: 2026-05-21.

Do not upgrade ARGUS to the latest available versions across the board just to
"future proof" the project.  For this codebase, reproducibility and binary
compatibility are the future-proofing strategy.  The ML and astronomy stack has
several native-extension boundaries where a newest-version install can silently
move the project onto an untested ABI, a missing CUDA wheel, or a different
training runtime.

### Why not latest everything?

- **Python stays on 3.11 for now.**  Python 3.11 is still supported until
  October 2027, and the OpenMMLab stack is much easier to install on Python
  3.11 than on newer interpreters.  Do not move to Python 3.12+ until the full
  PyTorch + mmcv + mmdet + CUDA wheel matrix has been verified on Mac and
  WSL2/cloud GPU environments.
- **NumPy stays on 1.26.4 for now.**  `sep` and `astride` are native-extension
  packages used by the classical astronomy pipeline.  NumPy 2.x changed the C
  ABI for compiled extension modules, so upgrading NumPy requires explicitly
  testing every compiled dependency rather than treating it as a routine bump.
- **MMCV/MMDetection are wheel-matrix constrained.**  `mmcv` must match the
  active PyTorch and CUDA combination.  If a pre-built wheel is not available,
  installs fall back to source builds, which are brittle on Linux and not a
  viable native-Windows path for this project.
- **Training results must remain comparable.**  ARGUS is research software with
  recorded DINOv3/YOLO/classical baselines.  Major upgrades to PyTorch,
  Ultralytics, Albumentations, Astropy, Photutils, or OpenCV can change model
  behaviour, augmentation semantics, image processing, or numerical outputs.
  Those upgrades need benchmark notes, not drive-by version changes.

### Current environment notes

The local `satid` environment is useful for development but has accumulated
organically.  Treat it as an environment snapshot, not as the canonical install
contract.  On 2026-05-21 the active environment differed from
the documented requirements lanes in several ways:

- `torch==2.11.0` and `torchvision==0.26.0` are installed locally, while the
  documented Mac/Linux baseline still uses PyTorch 2.2 where appropriate.
- `albumentations==2.0.8`, `spacetrack==1.4.0`, `ultralytics==8.4.46`, and
  `pydantic==2.13.3` are installed locally, while the requirements files pin
  older versions.
- Both `opencv-python` and `opencv-python-headless` are installed locally, and
  `cv2` imports from the headless package.  Production/API installs should keep
  only the headless package unless a GUI dependency is deliberately needed.
- `pip check` reports `openxlab` dependency conflicts with local `filelock` and
  `rich` versions.  This appears isolated to OpenMIM/OpenXLab tooling, but it
  is another reason to create clean training environments before important
  runs.

Before a paper run, cloud training run, or long benchmark, create a clean
environment from the documented install path and run the verification and test
suite.  Do not rely on the current workstation environment merely because it
imports successfully.

### Upgrade lanes

Use separate upgrade lanes instead of a single "latest" sweep:

- **Low risk:** frontend patch/minor updates within the current major versions
  of React, Vite, Tailwind, ESLint, and type packages.  On 2026-05-21 the
  frontend was already modern and only a few patch/minor releases behind; npm
  reported no production audit vulnerabilities.
- **Medium risk:** API/runtime patch updates such as FastAPI, SQLAlchemy,
  Uvicorn, requests, Pillow, and boto3.  These still require API tests and an
  upload/inference smoke test.
- **High risk:** Python, NumPy, PyTorch, torchvision, mmcv, mmdet, Ultralytics,
  Albumentations, Astropy, Photutils, OpenCV, `sep`, and `astride`.  Upgrade
  these only in a dedicated branch with environment notes, import checks, unit
  tests, and benchmark comparison.

Minimum acceptance for a high-risk dependency upgrade:

1. Build a fresh environment using the target platform install path.
2. Run `python -m pip check`.
3. Verify imports for `torch`, `torchvision`, `mmcv.ops`, `mmdet`, `numpy`,
   `cv2`, `sep`, `astride`, `astropy`, `fastapi`, and `ultralytics`.
4. Run the offline pytest suite.
5. Run at least one API/inference smoke test.
6. Re-run the relevant evaluation benchmark if preprocessing, training,
   detection, or numerical libraries changed.
7. Record the old and new versions plus any metric movement in the PR or
   experiment notes.

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

The training environment should live on the WSL filesystem (for example
`~/Argus`), not under `/mnt/c/...`. Keeping the repo, conda environment, data,
and checkpoints inside Ubuntu avoids Windows filesystem latency and path edge
cases during long training runs.

## Install Order — Critical Rule

**PyTorch and the MMDetection stack must be installed before the inference or
training requirements lane.** The wheel index URL differs by target platform.
Installing in the wrong order or from the wrong index silently installs the
wrong mmcv variant, which breaks Co-DINO training and DINO inference.

The requirements files are split by environment:

| File | Use for | Excludes |
|------|---------|----------|
| `requirements-base.txt` | Shared astronomy, orbital mechanics, image processing, and utility packages | API, ML, training, and test extras |
| `requirements-api.txt` | Lightweight FastAPI/database service | torch, MMDetection, Ultralytics, Albumentations, benchmarks, tests |
| `requirements-inference.txt` | Model-serving worker after platform-specific torch/mmcv/mmdet install | Training/evaluation/test-only tooling |
| `requirements-training.txt` | Training and evaluation after platform-specific torch/mmcv/mmdet install | Test-only tooling |
| `requirements-dev.txt` | Local developer environment with tests | Platform-specific torch/mmcv/mmdet |
| `requirements.txt` | Compatibility aggregate for full local dev | Platform-specific torch/mmcv/mmdet |

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

# Full local dev/test dependencies
pip install -r requirements-dev.txt
```

Set before running any training or inference on Mac:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## Platform B — Linux CUDA (Lambda A100, CUDA 12.1)

This is the path used by the Lambda Labs A100 cloud instance.

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

# Inference worker dependencies
pip install -r requirements-inference.txt \
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

# Training/evaluation dependencies
pip install -r requirements-training.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# DINOv3 package used by models/dino/dinov3_adapter.py
pip install git+https://github.com/facebookresearch/dinov3.git

# Verify the pieces that commonly fail on a new workstation
python - <<'PY'
import torch
import mmcv.ops
import dinov3.models.vision_transformer

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mmcv CUDA ops OK")
print("dinov3 import OK")
PY
```

> **Native Windows:** Do not attempt this platform natively — see the
> "Windows — mmcv CUDA ops" section above.  Use WSL2 and treat it as
> Platform C.

---

## Quick Install (after cloning — non-GPU environments only)

For the lightweight API service only:

```bash
conda create -n satid python=3.11 -y
conda activate satid
pip install -r requirements-api.txt
```

For Mac local development with tests, first install the Mac PyTorch/MMDetection
stack from Platform A, then run:

```bash
pip install -r requirements-dev.txt
```

For GPU training or inference environments, use the platform-specific steps
above so torch/mmcv/mmdet come from the correct wheel indexes.

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

### Space-Track cache
Space-Track query results are cached as local JSON files under `data/cache`.
The cache is implemented in `src/matching/spacetrack_query.py` to avoid unsafe
pickle deserialization in third-party cache backends.

---

## Optional: ASTAP plate solver (for WCS-less FITS images)

ARGUS uses ASTAP to derive WCS when a FITS image lacks native WCS keywords
and no `.wcs` sidecar is present.  ASTAP is invoked via subprocess from
`inference/plate_solver.py`; the solve is skipped gracefully when ASTAP is
not installed.

ASTAP is fast (sub-second for constrained solves) and works offline.  When
`RA`/`DEC` header keywords are present (as in Brent's capture software),
ARGUS passes them as a pointing hint with a 2° search radius instead of the
default 5° blind radius.

### Installation

**macOS:**
```bash
# Download from https://www.hnsky.org/astap.htm
# Install the .dmg — binary lands at:
#   /Applications/ASTAP.app/Contents/MacOS/astap
```

**Linux:**
```bash
# Download the .deb or .rpm from https://www.hnsky.org/astap.htm
sudo dpkg -i astap_amd64.deb   # or rpm -i astap_amd64.rpm
# Binary lands at /usr/local/bin/astap
```

### Star catalog (one-time download)

ASTAP requires a star catalog.  The H18 catalog covers all fields; G18 is
smaller and sufficient for most work.  Download from the ASTAP site or
directly:

```bash
# H18 — full sky, ~2.5 GB, recommended
# G18 — smaller subset, ~500 MB, adequate for typical telescope FOVs
# Both available at https://www.hnsky.org/astap.htm → "Star databases"
# Place the downloaded files in the default catalog location:
#   macOS:  /Applications/ASTAP.app/Contents/MacOS/
#   Linux:  /usr/local/lib/astap/  (or alongside the binary)
```

If the catalog is in a non-default location, set `ASTAP_CATALOG_DIR`:
```bash
export ASTAP_CATALOG_DIR=/path/to/catalog
```

### Configuration

```bash
# Required only if ASTAP is not in a standard location:
export ASTAP_BIN=/Applications/ASTAP.app/Contents/MacOS/astap   # macOS
export ASTAP_BIN=/usr/local/bin/astap                            # Linux

# Optional tuning:
export ASTAP_CATALOG_DIR=/path/to/catalog   # non-default catalog location
export ASTAP_TIMEOUT=60                      # subprocess timeout (seconds)
export ASTAP_DOWNSAMPLE=2                    # 1=full res, 2=half (default)
```

### Verification

```bash
# Quick smoke test — should print the solved centre coordinates:
python -m inference.plate_solver data/uploads/<job_id>/<filename>.fits
```

### Alternative: astrometry.net

astrometry.net (`solve-field`) can also plate-solve FITS files but is not
integrated into the ARGUS pipeline.  Use it manually if ASTAP is unavailable:

```bash
brew install astrometry-net    # macOS
# or: sudo apt-get install astrometry.net

solve-field image.fits --no-plots --overwrite --downsample 2 \
    --scale-units arcsecperpix --scale-low 1.0 --scale-high 2.0
# Writes image.new (solved FITS) and image.wcs — copy the .wcs beside the
# original FITS and ARGUS will pick it up automatically on next upload.
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
