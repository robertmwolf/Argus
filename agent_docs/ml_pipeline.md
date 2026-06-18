# ML Detection Pipeline — Co-DINO with Swin / DINOv3 Backbone

> **Note (2026-05-14):** This document was written for the original Swin backbone.
> A DINOv3 ViT-B/L backbone has since been integrated on `feature/dinov3-backbone`
> (now merged to `main`). See `agent_docs/dinov3_plan.md` for DINOv3-specific
> architecture, training strategy, and phase results. The Swin rules below remain
> valid for the Swin-T/L configs.

## Overview

Phase 0 (classical ASTRiDE baseline, `src/`) is complete.
This document covers the ML replacement: a Co-DINO transformer detector
fine-tuned on satellite streak data.

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
│   ├── merge_annotations.py ← SatStreaks masks + GTImages → COCO splits
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

`scripts/merge_annotations.py` builds the current full split files from
SatStreaks and GTImages:

```bash
python scripts/convert_gtimages.py \
  --strk-dir data/GTImages \
  --output data/annotations/gtimages.json \
  --negatives-output data/annotations/gtimages_negatives.json

python scripts/merge_annotations.py --seed 42 --val-fraction 0.2
```

SatStreaks entries now require a real mask file. The merge script reads image
dimensions with Pillow, converts each foreground mask into a COCO `bbox`,
records mask foreground pixel count as `area`, and stores a coarse `obb` using
the mask bbox center. Empty or missing masks are skipped. GTImages IDs are
offset by `1_000_000` so they cannot collide with SatStreaks IDs.

---

## FITS WCS Loading

`inference/fits_loader.py` returns both `wcs` and `wcs_source`. WCS lookup order:

1. Celestial WCS in the FITS header.
2. Same-stem `.wcs` / `.WCS` sidecar, especially ASTAP/SkyTrack GTImages files.
3. `None` when no valid WCS is available.

API upload processing copies an available sidecar next to the temporary FITS
path before running inference, preserving RA/Dec conversion even though the
uploaded file is processed outside its original directory.

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

**7. Postprocess grouping/fusion** — after Radon angle refinement and streak
extent tracing, per-detector duplicate boxes are removed with rotated-IoU NMS.
Cross-detector detections are assigned a shared `streak_id` when they overlap by
rotated-IoU ≥ 0.5, overlap by IoMin ≥ 0.3, or are collinear fragments of the
same physical streak. Each grouped streak then receives a fused OBB spanning the
outer projected endpoints of its member fragments before WCS/cross-ID and API
serialization.

**8. Endpoint tracing — `extend_obb_to_streak_extent`** — called for every
detection after Radon angle refinement. Traces the full image along the refined
streak axis and finds where the signal (strip max across ±`sample_halfwidth` px)
drops to background. The resulting run containing the OBB centre replaces the
initial OBB width. Key parameters (set in `pipeline.py`):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `sample_halfwidth` | **15 px** | ±8 px caused false "no centre run" failures when the Radon angle seed was off by even 1°, causing the tracer to miss the streak axis at t=0 and fall back to the original short OBB. 15 px is wide enough to reliably straddle the streak. |
| `threshold_sigma` | **3σ** above image median | 3σ keeps the centre-run boundary tight enough to exclude stars along the same azimuth. 2σ causes the centre run to absorb off-axis stars and overshoot to 2× the true length. |

For heatmap `line_segment` detections the pipeline preserves the native
`cx`/`cy` from `_line_to_obb` when calling `extend_obb_to_streak_extent`.
Reconstructing from the axis-aligned bbox centroid would shift the anchor
point to the edge of the partially-detected segment, changing the angle seed
and causing drift at the far tips.

---

## Training — Two-Stage Fine-Tuning

`training/train_dino.py` implements:

| Stage | Epochs | Backbone LR multiplier |
|-------|--------|------------------------|
| 1     | 1–15   | 0.0 (frozen)           |
| 2     | 16–50  | 0.01 (unfrozen)        |

Checkpoints saved every 5 epochs to `weights/checkpoints/`.
Best checkpoint (highest val mAP) saved to `weights/best.pth`.

Training CLI overrides are available for checkpoint-initialised or timeboxed
runs:

```bash
python -m training.train_dino \
  --work-dir weights/local_run \
  --load-from weights/local_run/best_coco_bbox_mAP_epoch_50.pth \
  --max-epochs 10 \
  --val-interval 2 \
  --checkpoint-interval 2
```

Use `--resume` to resume an interrupted run from the work directory. Use
`--load-from` when starting a new run initialized from an existing checkpoint.

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
- Skips satellite cross-identification
- Skips database write (prints to stdout)
- Uses `image_size=256` regardless of config

Radon angle refinement still runs in fast mode; it is required for usable OBB
geometry and remains bounded by the 512 px crop cap.

---

## Deferred Work (stubs)

Leave these as `raise NotImplementedError(...)` until Phase 7 weights exist:

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
Goal: API and frontend dev servers run, browser can upload image and see results with Swin-T model.
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
- Split requirements files are present and pinned; torch/mmcv/mmdet remain
  platform-specific installs documented in `agent_docs/dependencies.md`

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
