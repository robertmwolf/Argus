# ML Detection Pipeline — Co-DINO with Swin Backbone

## Overview

Phase 0 (classical ASTRiDE baseline, `src/`) is complete.
This document covers the ML replacement: a Co-DINO transformer detector
fine-tuned on satellite streak data, replacing YOLO-OBB with a
higher-accuracy oriented bounding box model.

---

## Hardware Constraints — Read First

```
DEVELOPMENT MACHINE: MacBook Air M3, 16GB unified RAM
  - All code is written and tested here
  - CPU or MPS (Apple Silicon) only — no CUDA

TRAINING MACHINE: Lambda Labs A100 40GB (rented when ready)
  - CUDA 12.1, PyTorch 2.2.0
  - Accessed via SSH
```

**Non-negotiable rule:** Code that only works on CUDA is not acceptable.
All modules must run on CPU/MPS for development.

Set this in your shell before running anything:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## Project Structure (new directories)

```
Argus/
├── inference/
│   ├── device.py          ← device abstraction (implement first)
│   └── pipeline.py        ← model loading + end-to-end inference
├── models/
│   └── dino/
│       ├── streak_codino_swin_t.py   ← dev config (Mac-safe, Swin-T)
│       └── streak_codino_swin_l.py   ← production config (A100, Swin-L)
├── training/
│   ├── convert_labels.py  ← FITS annotations → COCO JSON
│   ├── dataset.py         ← FITSStreakDataset (PyTorch Dataset)
│   ├── augmentations.py   ← Albumentations + synthetic streak injection
│   ├── make_dev_subset.py ← 50-image reproducible dev subset
│   └── train_dino.py      ← training entry point
├── eval/
│   └── benchmark.py       ← reproduces StreakMind paper metrics
├── scripts/
│   ├── download_weights.py       ← fetches pretrained COCO weights
│   ├── prepare_cloud_training.py ← go/no-go validation before GPU rental
│   ├── cloud_setup.sh            ← run once on Lambda instance
│   └── fetch_weights.sh          ← rsync weights back to Mac after training
└── weights/               ← gitignored, not checked in
    └── checkpoints/
```

---

## Device Abstraction — Implement First

**File:** `inference/device.py`

```python
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
    Conservative values safe for each environment.
    """
    device = get_device()

    configs = {
        "cuda": {
            "batch_size": 2,
            "num_workers": 4,
            "pin_memory": True,
            "image_size": 800,
            "mixed_precision": True,
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

Import `get_device()` in every file that touches PyTorch tensors.
**Never write** `torch.device('cuda')` or `.cuda()` anywhere.
Always write `.to(get_device())` instead.

---

## Model Size Selection

Swin-L requires ~22GB VRAM — it cannot run on the MacBook.
Maintain two configs at all times:

| File | Backbone | Purpose |
|------|----------|---------|
| `models/dino/streak_codino_swin_t.py` | Swin-T | Dev (Mac-safe) |
| `models/dino/streak_codino_swin_l.py` | Swin-L | Production (cloud) |

Selection via environment variable:
```bash
MODEL_SIZE=tiny    # default — loads Swin-T
MODEL_SIZE=large   # cloud only — loads Swin-L
```

In `inference/pipeline.py`:
```python
def load_model():
    model_size = os.getenv("MODEL_SIZE", "tiny")
    device = get_device()

    if model_size == "large" and device.type != "cuda":
        raise EnvironmentError(
            "MODEL_SIZE=large requires CUDA. "
            "On Mac, use MODEL_SIZE=tiny for development."
        )

    config_map = {
        "tiny":  "models/dino/streak_codino_swin_t.py",
        "large": "models/dino/streak_codino_swin_l.py",
    }
    # ... rest of MMDetection model loading
```

---

## Pretrained Weights

```python
WEIGHT_URLS = {
    "tiny": {
        "url": "https://download.openmmlab.com/mmdetection/v3.0/"
               "co_dino/co_dino_5scale_swin_t_3x_coco/"
               "co_dino_5scale_swin_t_3x_coco.pth",
        "filename": "co_dino_swin_t_coco.pth",
        "size_mb": 340,
    },
    "large": {
        "url": "https://download.openmmlab.com/mmdetection/v3.0/"
               "co_dino/co_dino_5scale_swin_l_16xb1_3x_coco/"
               "co_dino_5scale_swin_l_16xb1_3x_coco.pth",
        "filename": "co_dino_swin_l_coco.pth",
        "size_mb": 2400,
    },
}
```

- Download Swin-T weights for local dev (340MB).
- Do NOT download Swin-L (~2.4GB) during local development.
- `scripts/download_weights.py` reads `MODEL_SIZE` and fetches the right file.
- `weights/` is gitignored.

---

## Local Development Dataset

Full dataset (2335 FITS images) is too large for fast Mac iteration.

**`training/make_dev_subset.py`** — creates a reproducible 50-image subset:
- 20 images with no streaks
- 20 images with short streaks (< 269px)
- 10 images with long streaks (≥ 269px)

```bash
python -m training.make_dev_subset \
  --annotation data/annotations/train.json \
  --output data/annotations/dev_subset.json \
  --n-images 50 \
  --seed 42
```

`training/dataset.py` reads `USE_DEV_SUBSET` env var and loads the right annotation file:
```bash
USE_DEV_SUBSET=true   # local dev (default)
USE_DEV_SUBSET=false  # cloud training
```

---

## MPS Compatibility Rules

These operations are broken or slow on MPS. Handle each as specified:

**1. DataLoader workers**
```python
# Always use get_device_config()["num_workers"] — never hardcode
DataLoader(dataset, num_workers=get_device_config()["num_workers"])
```

**2. AMP / autocast**
```python
device = get_device()
use_amp = device.type == "cuda"

with torch.autocast(device_type=device.type, enabled=use_amp):
    outputs = model(inputs)
```

**3. NMS and roi_align fallback**
```python
def safe_nms(boxes, scores, iou_threshold):
    device = boxes.device
    if device.type == "mps":
        result = torchvision.ops.nms(
            boxes.cpu(), scores.cpu(), iou_threshold
        )
        return result.to(device)
    return torchvision.ops.nms(boxes, scores, iou_threshold)
```

**4. pin_memory** — always use `get_device_config()["pin_memory"]`

**5. skimage.transform.radon** (postprocess.py) — CPU only, expected, do not move to MPS

**6. Shapely rotated IoU** — CPU only, expected

---

## Training — Two-Stage Fine-Tuning

`training/train_dino.py` implements:

| Stage | Epochs | Backbone LR multiplier |
|-------|--------|------------------------|
| 1     | 1–15   | 0.0 (frozen)           |
| 2     | 16–50  | 0.01 (unfrozen)        |

Checkpoints saved every 5 epochs to `weights/checkpoints/`.
Best checkpoint (highest val mAP) saved to `weights/best.pth`.

**Cost guardrail** — after epoch 1, print:
```
Epoch 1/50 complete in Xm Ys.
Estimated total training time: Xh Ym
Estimated cost at $1.29/hr (Lambda A100): $X.XX
Press Ctrl+C within 30 seconds to abort if cost is unexpected.
```
Then `sleep(30)` before starting epoch 2.

**Smoke test flag:**
```bash
python -m training.train_dino --smoke-test
# Runs 2 epochs on 10 images, verifies loss decreases, exits.
# Must complete in under 5 minutes on A100.
```

---

## Fast Iteration Mode (Mac)

Pipeline must complete in under 60 seconds per image on Mac.

```bash
FAST_MODE=true python -m inference.pipeline --image data/raw/sample.fits
```

When `FAST_MODE=true`:
- Skips Radon angle refinement (uses raw DINO box angle)
- Skips satellite cross-identification
- Skips database write (prints to stdout)
- Uses `image_size=256` regardless of config

---

## Not Yet Implemented (stubs)

Leave these as `raise NotImplementedError(...)` until Phase 7 weights exist:

- Satellite cross-identification with live Space-Track API
- Multi-frame tracklet association (DB schema defined, logic stubbed)
- Swin-L → Swin-T weight distillation

---

## Phase Sequence

Follow this order exactly. Do not start a phase until the previous one passes.

### Phase 1 — Data Pipeline (Mac, CPU)
Goal: COCO JSON produced, `FITSStreakDataset` iterates cleanly.
```bash
python -m training.convert_labels
python -m training.dataset --smoke-test
```
Env: `USE_DEV_SUBSET=true`, `MODEL_SIZE=tiny`

### Phase 2 — Model Config (Mac, no GPU needed)
Goal: Both Swin-T and Swin-L configs pass `mmdet` config check.
```bash
python scripts/download_weights.py   # Swin-T only
```
Do NOT start Swin-L training locally.

### Phase 3 — Augmentation Pipeline (Mac, CPU)
Goal: `augmentations.py` runs on sample images, synthetic streak injection produces valid bounding boxes.
```bash
python -m training.augmentations --visualize
```

### Phase 4 — Integration Test (Mac, MPS, MODEL_SIZE=tiny)
Goal: Full pipeline runs end-to-end in FAST_MODE.
```bash
python -m inference.pipeline --fast --image data/raw/sample.fits
```
Expected: < 60 seconds wall time. MPS fallback warnings are normal.

### Phase 5 — API + Frontend (Mac, CPU)
Goal: `docker-compose up` works, browser can upload image and see results with Swin-T model.
```bash
curl -F "file=@data/raw/sample.fits" localhost:8000/api/upload
```
Note: results will be low quality with Swin-T and no fine-tuning — this is expected. Test the plumbing, not accuracy.

### Phase 6 — Cloud Handoff Validation (Mac)
Goal: `scripts/prepare_cloud_training.py` passes all checks. This is the go/no-go gate before spending money on GPU rental.

Checklist (script must confirm all):
- `data/annotations/train.json` valid COCO JSON
- `data/annotations/val.json` valid COCO JSON
- All FITS paths in annotations resolve to actual files
- `training/dataset.py` iterates 5 batches without error (dev subset, CPU)
- Both Swin-T and Swin-L MMDet configs valid
- `training/augmentations.py` runs on sample image without error
- `inference/pipeline.py` runs end-to-end in `FAST_MODE=true`
- `api/main.py` starts without error
- `docker-compose up --build` completes (CPU/MPS, `MODEL_SIZE=tiny`)
- `requirements.txt` is fully pinned (all `==` versions)

### Phase 7 — Cloud Training (Lambda Labs A100)
```bash
# 1. Rent A100 instance on Lambda Labs
# 2. rsync repo + data to instance
rsync -avz --exclude='.git' --exclude='uploads' . user@lambda-instance:/home/ubuntu/streakmind/
# 3. bash scripts/cloud_setup.sh
# 4. python -m training.train_dino --smoke-test
# 5. python -m training.train_dino   (full run, ~6-10 hrs)
# 6. bash scripts/fetch_weights.sh user@instance-ip
# 7. Terminate instance immediately after fetch
```
**A10 fallback:** If no A100 is available, an A10 (24GB) at $0.60/hr handles Swin-L with gradient checkpointing at batch size 1 — roughly 2× longer but still under $20.

Target: val mAP > 90%.

### Phase 8 — Evaluation (Mac, MPS with Swin-L weights)
Goal: reproduce ≥ 94% precision, ≥ 97% recall from StreakMind paper.
```bash
python -m eval.benchmark
```

---

## Environment Variables

```bash
# Required for all phases
export PYTORCH_ENABLE_MPS_FALLBACK=1

# ML pipeline
export MODEL_SIZE=tiny          # tiny (dev) or large (cloud only)
export USE_DEV_SUBSET=true      # false for cloud training run
export FAST_MODE=false          # true for quick end-to-end smoke tests

# Existing
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
```

Add all of the above to `.env.example`.

---

## Target Metrics

From StreakMind paper (what Phase 8 must reproduce or exceed):

| Metric | Target |
|--------|--------|
| Precision | ≥ 94% |
| Recall | ≥ 97% |
| val mAP (after fine-tuning) | > 90% |
