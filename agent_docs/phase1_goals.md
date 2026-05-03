# Phase 1 Goals — StreakMind Data Pipeline

> **Phase 0 (Classical Baseline) is complete.** See the bottom of this file
> for the completed Phase 0 week-by-week record.
> Phase 1 is the StreakMind data pipeline: FITS loader, PyTorch dataset,
> label conversion, and augmentations in `streakmind/`.

## Guiding Principle
Phase 1 must produce two things before Phase 2 (model training) can start:
1. `convert_labels.py` must produce a valid COCO JSON file
2. `FITSStreakDataset.__getitem__()` must iterate without error

Show these outputs before proceeding to Phase 2.

---

## Module 1: `streakmind/inference/fits_loader.py`

### Class: `FITSLoader`

```python
def load(path: str) -> dict:
    """Load a FITS file and normalize for model input.

    Returns dict with keys:
      array       — np.ndarray uint8 H×W×3 (Z-score normalized, 3-channel)
      wcs         — astropy.wcs.WCS object (or None if header has no WCS)
      exposure_time — float seconds (or None)
      filename    — str basename
      shape       — tuple (H, W)
    """

def fits_to_png(fits_path: str, output_path: str) -> None:
    """Convert FITS to PNG via load(). Save with cv2.imwrite."""

def extract_wcs_metadata(wcs: WCS, pixel_coords: list[tuple]) -> list[dict]:
    """Convert pixel (x,y) list to RA/Dec.

    Returns list of dicts: {x_pix, y_pix, ra_deg, dec_deg}
    """
```

**Normalization rule:** Z-score, NOT min-max.
- Clip pixel values to ±3σ from the image mean
- Scale clipped range to [0, 255] uint8
- Stack grayscale to 3 channels (shape H×W×3)

Rationale: FITS images have extreme dynamic range. Min-max crushes faint
streaks into noise. Z-score preserves relative contrast.

### Test file: `tests/streakmind/test_fits_loader.py`
- [ ] `load()` returns dict with all required keys and correct dtypes
- [ ] Output array is uint8, shape (H, W, 3)
- [ ] `fits_to_png()` creates a file at the given path
- [ ] `extract_wcs_metadata()` returns correct RA/Dec for image center pixel
- [ ] Corrupted FITS file raises a clear exception, not a cryptic crash
- [ ] Missing WCS header → `wcs` key is None, no crash

---

## Module 2: `streakmind/training/convert_labels.py`

### Function: `convert_yolo_obb_to_coco`

```python
def convert_yolo_obb_to_coco(
    yolo_label_dir: str,
    fits_dir: str,
    output_json: str
) -> None:
    """Convert YOLO OBB label files to COCO JSON.

    YOLO OBB format (per line): class cx cy w h angle_deg
      - cx, cy, w, h normalized 0–1
      - angle_deg in degrees (not normalized)

    COCO JSON output:
      images: [{id, file_name, width, height}]
      annotations: [{
        id, image_id, category_id,
        bbox: [x1, y1, w, h],   ← axis-aligned, denormalized pixels
        area: float,
        obb: [cx, cy, w, h, angle_deg],  ← denormalized pixels, stored in extra field
        iscrowd: 0
      }]
      categories: [{"id": 0, "name": "streak"}]

    Prints summary: total images, total streaks,
      streak length distribution (min / mean / max / p75 px)
    """
```

### Test file: `tests/streakmind/test_convert_labels.py`
- [ ] Output JSON parses without error and has `images`, `annotations`, `categories`
- [ ] Category list is `[{"id": 0, "name": "streak"}]`
- [ ] Each annotation has an `obb` field with 5 values
- [ ] `bbox` values are pixel-space (not 0–1 normalized)
- [ ] Empty label directory → valid COCO JSON with zero annotations
- [ ] Prints summary statistics to stdout

---

## Module 3: `streakmind/training/dataset.py`

### Class: `FITSStreakDataset(torch.utils.data.Dataset)`

```python
def __init__(self, annotation_file: str, transforms=None):
    """Load COCO JSON and build id→metadata and id→annotation lookups."""

def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
    """Load FITS via FITSLoader (NOT the PNG) and build target dict.

    Target dict keys:
      boxes       — FloatTensor [N, 4] in [x1, y1, x2, y2] pixel coords
      labels      — LongTensor [N] all zeros (single class: streak)
      image_id    — IntTensor scalar
      obb_params  — FloatTensor [N, 5] as [cx, cy, w, h, angle_deg]
                    (stored separately for angle supervision)
    """
```

**Important:** Always load from the source FITS file, never from the cached PNG.
The PNG is only for visualization. Training must use the normalized FITS data
to ensure reproducibility.

### Test file: `tests/streakmind/test_dataset.py`
- [ ] `len(dataset)` equals number of images in annotation file
- [ ] `__getitem__` returns `(Tensor, dict)` with correct key set
- [ ] `boxes` shape is `[N, 4]`, dtype float32
- [ ] `labels` shape is `[N]`, all values 0
- [ ] `obb_params` shape is `[N, 5]`
- [ ] Dataset with transforms applied does not crash

---

## Module 4: `streakmind/training/augmentations.py`

### Function: `get_train_transforms() -> A.Compose`

Build augmentation pipeline using albumentations in this exact order:
1. `HorizontalFlip(p=0.5)`
2. `VerticalFlip(p=0.5)`
3. `RandomRotate90(p=0.5)`
4. `ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=180, p=0.8)`
5. `RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7)`
6. `GaussNoise(var_limit=(10, 50), p=0.5)`
7. `Blur(blur_limit=3, p=0.3)`

### Class: `SyntheticStreakInject(A.DualTransform)`

```
# Source: StreakMind — synthetic streak injection for class balancing
# Ref: StreakMind paper/repo (cite per published source)
```

With probability `p`:
- Inject 1–3 synthetic streaks into the image
- Each streak parameters:
  - Angle: uniform [0°, 180°)
  - Length: uniform [50px, image diagonal]
  - Brightness: image_mean_sky + U(1σ, 5σ)
  - Cross-section: Gaussian profile, σ = 1.5 px
  - Start position: random, ensuring streak stays fully in-frame
- Return updated image AND updated `bboxes`/`obb_params`

This is critical for balancing long vs short streak representation.

### Function: `get_val_transforms() -> A.Compose`
No augmentations — return raw normalized image only.

### Test file: `tests/streakmind/test_augmentations.py`
- [ ] `get_train_transforms()` runs on a dummy image without error
- [ ] Output image stays uint8 with shape unchanged
- [ ] `SyntheticStreakInject` adds at least one streak to bboxes when triggered
- [ ] `get_val_transforms()` returns the image unchanged

---

## Phase 1 Gate

Before starting Phase 2, verify both:

```bash
# 1. COCO conversion produces valid output
python streakmind/training/convert_labels.py \
  --yolo-labels data/annotations/yolo/ \
  --fits-dir data/raw/ \
  --output streakmind/data/annotations/train.json

# 2. Dataset iterates cleanly
python streakmind/training/dataset.py streakmind/data/annotations/train.json
# Expected: prints first item shape + target keys, no errors

pytest tests/streakmind/ -v
# Expected: all tests pass
```

---

---

## Appendix: Phase 0 — Classical Baseline (Complete)

The classical baseline is implemented in `src/`. It remains the comparison
baseline for evaluation in Phase 8.

### Week 1 — FITS Ingest (`src/ingest/fits_parser.py`) ✅
- `FITSImage` dataclass with all header fields
- Z-score normalization to float32
- Graceful handling of missing optional fields

### Week 2 — ASTRiDE Streak Detector (`src/detection/classical_detector.py`) ✅
- Background subtraction (sep), 16-bit conversion, ASTRiDE detection
- Returns `StreakDetection` dataclass list
- Tunable `contour_threshold` and `min_length_px` parameters

### Weeks 3–4 — Astrometry + SGP4 Matching (pending)
Still to build for classical baseline:
- `src/astrometry/plate_solver.py`
- `src/matching/spacetrack_query.py`
- `src/matching/spatial_filter.py`
- `src/matching/propagator.py`
- `src/matching/matcher.py` + `scorer.py`

These can be built in parallel with StreakMind Phase 1, as the matching
logic is shared with StreakMind's `inference/crossid.py`.
