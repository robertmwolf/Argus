# Phase 1 Goals — ARGUS Data Pipeline

> **Phase 0 (Classical Baseline) is complete.** See the bottom of this file
> for the completed Phase 0 week-by-week record.
> Phase 1 is the ARGUS data pipeline: FITS loader, PyTorch dataset,
> label conversion, and augmentations.

## Guiding Principle
Phase 1 must produce two things before Phase 2 (model training) can start:
1. `convert_labels.py` must produce a valid COCO JSON file
2. `FITSStreakDataset.__getitem__()` must iterate without error

Show these outputs before proceeding to Phase 2.

---

## Module 1: `inference/fits_loader.py`

### Class: `FITSLoader`

```python
def load(path: str) -> dict:
    """Load a FITS file and normalize for model input.

    Returns dict with keys:
      array       — np.ndarray uint8 H×W×3 (ARGUS_NORM normalized, 3-channel)
      wcs         — astropy.wcs.WCS object (or None if no FITS/sidecar WCS)
      wcs_source  — "fits", "sidecar", or None
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

**Normalization rule:** Use `ARGUS_NORM` and match training preprocessing.
The current local Swin-T weights use Z-score, NOT min-max.
- Clip pixel values to ±3σ from the image mean
- Scale clipped range to [0, 255] uint8
- Stack grayscale to 3 channels (shape H×W×3)

Rationale: FITS images have extreme dynamic range. Min-max crushes faint
streaks into noise. Z-score preserves relative contrast.

### Test file: `tests/test_fits_loader.py`
- [ ] `load()` returns dict with all required keys and correct dtypes
- [ ] Output array is uint8, shape (H, W, 3)
- [ ] `fits_to_png()` creates a file at the given path
- [ ] `extract_wcs_metadata()` returns correct RA/Dec for image center pixel
- [ ] Corrupted FITS file raises a clear exception, not a cryptic crash
- [ ] Missing WCS header → `wcs` key is None, no crash
- [ ] Same-stem `.wcs` sidecar loads when FITS header has no celestial WCS

---

## Module 3: `training/dataset.py`

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

### Test file: `tests/test_dataset.py`
- [ ] `len(dataset)` equals number of images in annotation file
- [ ] `__getitem__` returns `(Tensor, dict)` with correct key set
- [ ] `boxes` shape is `[N, 4]`, dtype float32
- [ ] `labels` shape is `[N]`, all values 0
- [ ] `obb_params` shape is `[N, 5]`
- [ ] Dataset with transforms applied does not crash

---

## Module 4: `training/augmentations.py`

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

### Test file: `tests/test_augmentations.py`
- [ ] `get_train_transforms()` runs on a dummy image without error
- [ ] Output image stays uint8 with shape unchanged
- [ ] `SyntheticStreakInject` adds at least one streak to bboxes when triggered
- [ ] `get_val_transforms()` returns the image unchanged

---

## Phase 1 Gate

Before starting Phase 2, verify:

```bash
# Dataset iterates cleanly
python training/dataset.py data/annotations/train.json
# Expected: prints first item shape + target keys, no errors

pytest tests/ -v
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

These can be built in parallel with ARGUS Phase 1, as the matching
logic is shared with ARGUS's `inference/crossid.py`.
