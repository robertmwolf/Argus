# ARGUS — Automated Recognition and Grading of Unidentified Streaks

## What This Is
An automated pipeline that identifies satellites in FITS telescope images
by detecting streaks and matching them against TLE orbital data from
Space-Track's GP_History API using SGP4 propagation and multi-factor
confidence scoring.

## Current Phase
**PHASE 1 — Classical Baseline (Weeks 1–4). No ML yet.**
Goal: Prove the end-to-end pipeline works on real FITS data.
Establish baseline accuracy metrics for comparison against Phase 2.

## Read These Before Starting Any Task
Always read the relevant agent_docs file before writing code:

- `agent_docs/architecture.md`   — full system design, component map, data flow
- `agent_docs/phase1_goals.md`   — week-by-week tasks and success metrics
- `agent_docs/datasets.md`       — where to get test data, download links
- `agent_docs/dependencies.md`   — exact packages, versions, install commands
- `agent_docs/service_roadmap.md` — FastAPI service, Docker, Cloudflare Tunnel, scale path
- `agent_docs/test_strategy.md`  — how to measure and record baseline accuracy
- `agent_docs/spacetrack.md`     — Space-Track API usage, rate limits, caching rules

## Stack
- Python 3.11, conda environment named `argus`
- Key libs: astropy, astride, sgp4, skyfield, spacetrack, opencv-python, scipy
- Testing: pytest
- No ML frameworks in Phase 1

## Project Structure
```
Argus/
├── CLAUDE.md
├── README.md
├── agent_docs/          ← read before coding
├── src/
│   ├── ingest/          ← FITS parsing
│   ├── detection/       ← ASTRiDE streak detection
│   ├── astrometry/      ← WCS plate solving, pixel→sky coords
│   └── matching/        ← Space-Track query, SGP4, scoring
├── tests/               ← pytest test files
├── results/             ← baseline metrics JSON output
└── data/                ← FITS data (not checked in)
    ├── milan/           ← MILAN sky survey FITS files
    └── sample/          ← small sample files for quick testing
```

## Academic Research Context
This project is academic research software. It builds on the following prior works:

- **ASTRiDE** — Automated Streak Detection for Astronomical Images
  (Kim et al., https://github.com/dwkim78/ASTRiDE)
- **StreakMind** — YOLO-OBB satellite streak detection model
  (StreakMind project, cite per their published paper/repo)
- **Danarianto et al. Prototype** — satellite identification prototype pipeline
  (Danarianto et al., cite per their published paper)

### Source Citation Rules
Whenever code directly implements, adapts, or is substantially derived from one
of the above works, add an inline citation comment at the function, class, or
code block level. Use this format:

```python
# Source: <AuthorOrProject> — <brief description of what was adapted>
# Ref: <DOI, URL, or "unpublished manuscript" as appropriate>
```

Apply citations to:
- Any algorithm, formula, or threshold copied or adapted from a prior work
- Any preprocessing step, model architecture, or scoring logic derived from a source
- Helper functions whose design is traceable to a specific paper or repo

Do **not** cite:
- Standard library usage or generic Python idioms
- astropy/sgp4/skyfield API calls that follow their own documentation
- Logic that is entirely original to this project

When in doubt, cite. Over-attribution is preferable to under-attribution in
academic research code.

## Code Standards
- Type hints on every function signature
- Google-style docstrings on every public function and class
- Every module has a `if __name__ == "__main__":` block for standalone testing
- Never hardcode credentials — use environment variables only
- All file paths via `pathlib.Path`, never raw strings
- Log with `logging` module, not `print()` (except __main__ blocks)

## Environment Variables Required
```bash
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
```

## Running Tests
```bash
conda activate argus
pytest tests/ -v
```

## Workflow Rules
- Build one week's tasks completely before moving to the next
- Write pytest tests alongside each module, not after
- Run pytest after every module is complete — fix failures before continuing
- Write baseline metrics to results/phase1_baseline.json at end of Week 4
- Ask for a plan before writing code for any module over 100 lines


═══════════════════════════════════════════════════════════
HARDWARE ADDENDUM — READ THIS BEFORE WRITING ANY CODE
═══════════════════════════════════════════════════════════

This addendum modifies the main implementation plan to account 
for a two-machine workflow:

  DEVELOPMENT MACHINE: MacBook Air M3, 16GB unified RAM
    - All code is written and tested here
    - Uses CPU or MPS (Apple Silicon GPU) only
    - No NVIDIA CUDA available

  TRAINING MACHINE: Rented cloud GPU (Lambda Labs A100 40GB)
    - Used only when code is complete and ready to train
    - CUDA 12.1, PyTorch 2.2.0
    - Accessed via SSH

Every decision in this project must respect this constraint.
Code that only works on CUDA is not acceptable.
All modules must run on CPU/MPS for development.

═══════════════════════════════════════════════════════════
DEVICE ABSTRACTION — IMPLEMENT THIS FIRST
═══════════════════════════════════════════════════════════

Before writing any other code, create inference/device.py:

```python
# inference/device.py

import torch
import logging

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Returns the best available device in priority order:
      1. CUDA (cloud GPU)
      2. MPS (Apple Silicon)
      3. CPU (fallback)

    Never hardcode 'cuda' anywhere in the codebase.
    Always call get_device() instead.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available, using CPU — training will be slow")
    return device


def get_device_config() -> dict:
    """
    Returns device-appropriate training hyperparameters.
    These are conservative values safe for each environment.
    """
    device = get_device()

    configs = {
        "cuda": {
            "batch_size": 2,
            "num_workers": 4,
            "pin_memory": True,
            "image_size": 800,
            "mixed_precision": True,    # AMP on CUDA
            "gradient_checkpointing": True,
        },
        "mps": {
            "batch_size": 1,
            "num_workers": 0,           # MPS doesn't support multiprocessing workers
            "pin_memory": False,        # not supported on MPS
            "image_size": 400,          # halved to fit in 16GB unified memory
            "mixed_precision": False,   # MPS AMP support is incomplete
            "gradient_checkpointing": True,
        },
        "cpu": {
            "batch_size": 1,
            "num_workers": 2,
            "pin_memory": False,
            "image_size": 400,
            "mixed_precision": False,
            "gradient_checkpointing": True,
        },
    }

    return configs[device.type]
```

Import get_device() in every file that touches PyTorch tensors.
Never write torch.device('cuda') or .cuda() anywhere.
Always write .to(get_device()) instead.

Set this environment variable in your shell before running anything:
  export PYTORCH_ENABLE_MPS_FALLBACK=1

Add it to .env.example as well.

═══════════════════════════════════════════════════════════
MODEL SIZE SELECTION — TWO CONFIGS REQUIRED
═══════════════════════════════════════════════════════════

The main plan specifies Co-DINO with Swin-L backbone.
Swin-L requires ~22GB VRAM minimum — it cannot run on the MacBook.

You must maintain TWO model configurations at all times:

  models/dino/streak_codino_swin_t.py   ← development (Mac-safe)
  models/dino/streak_codino_swin_l.py   ← production (cloud GPU)

The Swin-T config is for all local development, debugging, and 
integration testing. The Swin-L config is only used during the 
cloud training run.

Selection via environment variable:
  MODEL_SIZE=tiny    → loads Swin-T config  (default)
  MODEL_SIZE=large   → loads Swin-L config  (cloud only)

Implement this in inference/pipeline.py:

```python
import os
from inference.device import get_device, get_device_config

def load_model():
    model_size = os.getenv("MODEL_SIZE", "tiny")
    device = get_device()
    device_cfg = get_device_config()

    if model_size == "large" and device.type != "cuda":
        raise EnvironmentError(
            "MODEL_SIZE=large requires CUDA. "
            "On Mac, use MODEL_SIZE=tiny for development."
        )

    config_map = {
        "tiny":  "models/dino/streak_codino_swin_t.py",
        "large": "models/dino/streak_codino_swin_l.py",
    }

    config_path = config_map[model_size]
    # ... rest of MMDetection model loading
```

Add to .env.example:
  MODEL_SIZE=tiny

═══════════════════════════════════════════════════════════
LOCAL DEVELOPMENT DATASET
═══════════════════════════════════════════════════════════

The full dataset of 2335 FITS images is too large to iterate on 
quickly during development on the Mac.

Create a development subset tool:

--- training/make_dev_subset.py ---

```python
"""
Creates a small reproducible subset of the dataset for fast
local iteration on Mac. Not used during cloud training.

Usage:
  python -m training.make_dev_subset \
    --annotation data/annotations/train.json \
    --output data/annotations/dev_subset.json \
    --n-images 50 \
    --seed 42

Selection strategy:
  - 20 images with no streaks
  - 20 images with short streaks (< 269px)
  - 10 images with long streaks (>= 269px)
This preserves class balance in the dev subset.
"""
```

Add to .env.example:
  USE_DEV_SUBSET=true       # set to false for cloud training run

In training/dataset.py, read USE_DEV_SUBSET and load the 
appropriate annotation file automatically.

During all local development, training/testing is done on
the 50-image subset only. Cloud training uses the full dataset.

═══════════════════════════════════════════════════════════
MPS-SPECIFIC COMPATIBILITY RULES
═══════════════════════════════════════════════════════════

These MMDetection and PyTorch operations are either broken or 
slow on MPS. Handle each as specified:

1. DataLoader num_workers must be 0 on MPS.
   Use get_device_config()["num_workers"] everywhere.
   Never hardcode num_workers.

2. torch.cuda.amp.autocast() crashes on MPS.
   Wrap all AMP usage:

```python
   from inference.device import get_device

   device = get_device()
   use_amp = device.type == "cuda"

   with torch.autocast(device_type=device.type, enabled=use_amp):
       outputs = model(inputs)
```

3. Some torchvision ops (e.g. nms, roi_align) have incomplete 
   MPS implementations. If you get a "not implemented for MPS" 
   error, add a CPU fallback:

```python
   def safe_nms(boxes, scores, iou_threshold):
       device = boxes.device
       if device.type == "mps":
           # Fall back to CPU for this op
           result = torchvision.ops.nms(
               boxes.cpu(), scores.cpu(), iou_threshold
           )
           return result.to(device)
       return torchvision.ops.nms(boxes, scores, iou_threshold)
```

4. pin_memory=True crashes on MPS. Always use 
   get_device_config()["pin_memory"].

5. skimage.transform.radon (used in postprocess.py) runs on CPU 
   only — this is fine and expected. Do not attempt to move it 
   to MPS.

6. The Shapely library (rotated IoU) is CPU-only — also fine 
   and expected.

═══════════════════════════════════════════════════════════
PRETRAINED WEIGHTS — MAC-SAFE DOWNLOAD
═══════════════════════════════════════════════════════════

The cloud-pretrained Co-DINO Swin-L weights file is ~2.4GB.
Do NOT download it during local development.

The Swin-T weights are ~340MB and should be downloaded for 
local development.

Create a script: scripts/download_weights.py

```python
"""
Downloads pretrained weights appropriate for current environment.

Usage:
  python scripts/download_weights.py

Behavior:
  MODEL_SIZE=tiny  → downloads Swin-T Co-DINO COCO weights (~340MB)
  MODEL_SIZE=large → downloads Swin-L Co-DINO COCO weights (~2.4GB)

Weights are saved to: weights/ directory (gitignored)
Skip download if file already exists.
"""

WEIGHT_URLS = {
    "tiny": {
        "url": "https://download.openmmlab.com/mmdetection/v3.0/"
               "co_dino/co_dino_5scale_swin_t_3x_coco/"
               "co_dino_5scale_swin_t_3x_coco.pth",
        "filename": "co_dino_swin_t_coco.pth",
        "sha256": "...",   # fill in after first download
    },
    "large": {
        "url": "https://download.openmmlab.com/mmdetection/v3.0/"
               "co_dino/co_dino_5scale_swin_l_16xb1_3x_coco/"
               "co_dino_5scale_swin_l_16xb1_3x_coco.pth",
        "filename": "co_dino_swin_l_coco.pth",
        "sha256": "...",
    },
}
```

Add weights/ to .gitignore.

═══════════════════════════════════════════════════════════
FAST ITERATION MODE
═══════════════════════════════════════════════════════════

On Mac, full pipeline runs must complete in under 60 seconds 
for a single image to be usable during development.

Add a --fast flag to the pipeline that skips slow operations:

  inference/pipeline.py --fast
    - Skips Radon angle refinement (uses raw DINO box angle)
    - Skips satellite cross-identification
    - Skips database write (prints to stdout instead)
    - Uses image_size=256 regardless of config

This lets you verify the detection loop is working end-to-end 
in seconds rather than minutes while on Mac.

Implement as:
  FAST_MODE=true  in environment, or
  pipeline.run(image, fast=True) in code

═══════════════════════════════════════════════════════════
CLOUD TRAINING HANDOFF PACKAGE
═══════════════════════════════════════════════════════════

When local development is complete and ready for cloud training,
the following must be ready to transfer. Create a script that 
validates and packages everything:

--- scripts/prepare_cloud_training.py ---

This script must verify and report on each item before you 
SSH into the rented GPU:

CHECKLIST — script must confirm all of these pass:

  [ ] data/annotations/train.json exists and is valid COCO JSON
        - Run: python -c "import json; json.load(open('...'))"
        - Report: N images, N annotations, class names

  [ ] data/annotations/val.json exists and is valid COCO JSON

  [ ] All FITS paths in annotations resolve to actual files

  [ ] training/dataset.py iterates 5 batches without error
        using dev subset on CPU (USE_DEV_SUBSET=true)

  [ ] models/dino/streak_codino_swin_t.py is valid MMDet config
        - Run: python -m mmdet.utils.check_config ...

  [ ] models/dino/streak_codino_swin_l.py is valid MMDet config

  [ ] training/augmentations.py runs on a sample image without error

  [ ] inference/pipeline.py runs end-to-end in FAST_MODE=true

  [ ] api/main.py starts without error (uvicorn --reload)

  [ ] docker-compose up --build completes without error
        (using CPU/MPS, MODEL_SIZE=tiny)

  [ ] requirements.txt is pinned (all packages have == versions)

On success, print:
  "✓ Ready for cloud training. 
   Transfer the following to your GPU instance:
   - Your full FITS dataset
   - data/annotations/
   - The entire repository
   Run: rsync -avz --exclude='.git' --exclude='uploads' 
        . user@lambda-instance:/home/ubuntu/streakmind/"

On any failure, print which check failed and why, then exit 1.

═══════════════════════════════════════════════════════════
CLOUD INSTANCE SETUP SCRIPT
═══════════════════════════════════════════════════════════

Create scripts/cloud_setup.sh — run this once after SSHing 
into the rented GPU instance:

```bash
#!/bin/bash
# Run once on Lambda Labs A100 instance after rsync
# Usage: bash scripts/cloud_setup.sh

set -e

echo "=== Installing system dependencies ==="
sudo apt-get update -q
sudo apt-get install -y libgl1 libglib2.0-0

echo "=== Installing Python dependencies ==="
pip install -q -U pip
pip install -q torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -q -U openmim
mim install -q mmengine mmcv mmdet
pip install -r requirements.txt

echo "=== Downloading Swin-L weights ==="
MODEL_SIZE=large python scripts/download_weights.py

echo "=== Verifying CUDA ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found'; \
           print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"

echo "=== Setting environment ==="
cp .env.example .env
sed -i 's/MODEL_SIZE=tiny/MODEL_SIZE=large/' .env
sed -i 's/USE_DEV_SUBSET=true/USE_DEV_SUBSET=false/' .env

echo ""
echo "=== Setup complete. Start training with: ==="
echo "    python -m training.train_dino"
```

═══════════════════════════════════════════════════════════
CLOUD TRAINING SCRIPT
═══════════════════════════════════════════════════════════

--- training/train_dino.py ---

The training script must:

  1. Call get_device() at startup and log which device is active
  2. Load device config via get_device_config()
  3. Apply two-stage fine-tuning as specified in main plan:
       Stage 1 epochs 1-15:  backbone frozen (lr_mult=0.0)
       Stage 2 epochs 16-50: backbone unfrozen (lr_mult=0.01)
  4. Save checkpoints every 5 epochs to weights/checkpoints/
  5. Save best checkpoint (highest val mAP) to weights/best.pth
  6. Log to both stdout and training/logs/run_{timestamp}.log
  7. Print ETA after each epoch based on elapsed time

Add a --smoke-test flag:
  Runs 2 epochs on 10 images, verifies loss decreases, exits.
  Used to confirm cloud setup works before committing to full run.
  Usage: python -m training.train_dino --smoke-test

The smoke test must complete in under 5 minutes on an A100.
Run this first thing after cloud_setup.sh completes.

═══════════════════════════════════════════════════════════
WEIGHTS RETRIEVAL AFTER TRAINING
═══════════════════════════════════════════════════════════

After training completes on the cloud instance, retrieve weights:

Create scripts/fetch_weights.sh:

```bash
#!/bin/bash
# Run on your Mac after cloud training completes
# Usage: bash scripts/fetch_weights.sh user@lambda-ip

REMOTE=$1
if [ -z "$REMOTE" ]; then
  echo "Usage: bash scripts/fetch_weights.sh user@instance-ip"
  exit 1
fi

mkdir -p weights/
echo "Fetching best checkpoint..."
rsync -avz --progress \
  $REMOTE:/home/ubuntu/streakmind/weights/best.pth \
  weights/streakmind_codino_swin_l.pth

echo "Fetching training logs..."
rsync -avz \
  $REMOTE:/home/ubuntu/streakmind/training/logs/ \
  training/logs/

echo ""
echo "Done. Update your .env:"
echo "  MODEL_SIZE=large"
echo "  MODEL_WEIGHTS=weights/streakmind_codino_swin_l.pth"
echo ""
echo "Note: MODEL_SIZE=large requires MPS or CUDA."
echo "For inference only on Mac MPS, this may work."
echo "If you get memory errors, convert weights to Swin-T"
echo "using scripts/distill_to_swin_t.py (future work)."
```

═══════════════════════════════════════════════════════════
PHASE SEQUENCING — REVISED FOR THIS HARDWARE PLAN
═══════════════════════════════════════════════════════════

Follow this exact order. Do not start a phase until the 
previous one is complete and tested.

PHASE 1 — Data Pipeline (Mac, CPU)
  Goal: COCO JSON produced, FITSStreakDataset iterates cleanly
  Verify: python -m training.convert_labels && 
          python -m training.dataset --smoke-test
  Environment: USE_DEV_SUBSET=true, MODEL_SIZE=tiny

PHASE 2 — Model Config (Mac, no GPU needed)
  Goal: Both Swin-T and Swin-L configs pass mmdet config check
  Verify: python scripts/download_weights.py (Swin-T only)
  Do NOT start Swin-L training locally

PHASE 3 — Augmentation Pipeline (Mac, CPU)
  Goal: augmentations.py runs on sample images, synthetic 
        streak injection produces valid bounding boxes
  Verify: python -m training.augmentations --visualize

PHASE 4 — Integration Test (Mac, MPS, MODEL_SIZE=tiny)
  Goal: Full pipeline runs end-to-end in FAST_MODE
  Verify: python -m inference.pipeline --fast --image data/raw/sample.fits
  Expected: <60 seconds wall time

PHASE 5 — API + Frontend (Mac, CPU)
  Goal: docker-compose up works, browser can upload image 
        and see results using Swin-T model
  Verify: curl -F "file=@data/raw/sample.fits" localhost:8000/api/upload
  Note: results will be low quality with Swin-T and no fine-tuning,
        this is expected — you are testing the plumbing, not accuracy

PHASE 6 — Cloud Handoff Validation (Mac)
  Goal: python scripts/prepare_cloud_training.py passes all checks
  This is your go/no-go gate before spending money on GPU rental

PHASE 7 — Cloud Training (Lambda Labs A100)
  Goal: Fine-tuned Swin-L weights, val mAP > 90%
  Steps:
    1. Rent A100 instance on Lambda Labs
    2. rsync repo + data to instance
    3. bash scripts/cloud_setup.sh
    4. python -m training.train_dino --smoke-test
    5. python -m training.train_dino   (full run, ~6-10hrs)
    6. bash scripts/fetch_weights.sh user@instance-ip
    7. Terminate instance immediately after fetch

PHASE 8 — Evaluation (Mac, MPS with Swin-L weights)
  Goal: Reproduce ≥94% precision, ≥97% recall from StreakMind paper
  Verify: python -m eval.benchmark

═══════════════════════════════════════════════════════════
COST GUARDRAILS
═══════════════════════════════════════════════════════════

Add a training time estimator to train_dino.py.
After the first epoch completes, print:

  "Epoch 1/50 complete in Xm Ys.
   Estimated total training time: Xh Ym
   Estimated cost at $1.29/hr (Lambda A100): $X.XX
   Press Ctrl+C within 30 seconds to abort if cost is unexpected."

Then sleep(30) before continuing to epoch 2.

This prevents accidentally running a misconfigured training job
for hours before noticing something is wrong.

═══════════════════════════════════════════════════════════
WHAT NOT TO BUILD YET
═══════════════════════════════════════════════════════════

Do not implement the following until Phase 7 is complete 
and you have real fine-tuned weights:

  - Satellite cross-identification with live Space-Track API
    (use the local TLE file only for now)
  - Multi-frame tracklet association
    (implement the DB schema but leave the logic as a stub)
  - The Swin-L → Swin-T weight distillation script
    (only needed if Mac inference with large model is too slow)

Stub these with:
  raise NotImplementedError("Implement after cloud training — see Phase 7")

So the codebase is complete in structure but honest about 
what is not yet functional.

═══════════════════════════════════════════════════════════
FIRST TASK
═══════════════════════════════════════════════════════════

Begin with inference/device.py. 

Show me the file, then run:
  python -c "from inference.device import get_device, 
             get_device_config; print(get_device()); 
             print(get_device_config())"

Confirm it prints 'mps' and the MPS config dict on this Mac.
Then proceed to Phase 1.