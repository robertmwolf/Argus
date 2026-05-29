# ARGUS Data Strategy

**Status:** Active policy — governs all training runs from Run 4 onward.
**Last updated:** 2026-05-28

---

## 1. Goals and Principles

ARGUS is built to detect satellite streaks in ground-based FITS images from
telescopes similar to the Atwood Observatory instrument.  The data strategy
follows from three goals:

1. **Train, validate, and test within the production domain.**  Every split
   that drives a training or quality decision should come from real ground-based
   FITS data.  Cross-domain datasets (space-based, different imaging format) may
   be retained as secondary benchmarks to track relative trends, but they do not
   define quality gates.

2. **Stratify by streak geometry, not by object identity.**  What the model
   needs to generalise across is the visual morphology space — streak length,
   thickness, brightness, and orientation.  Splitting by NORAD ID controls for
   object identity, which is not what the model learns.  Splitting by geometry
   ensures every morphology type appears in every split.

3. **Generalize to similar-but-not-identical scopes through augmentation, not
   through domain-mixed training data.**  Controlled augmentation (scale jitter,
   PSF blur, noise injection) is more predictable and debuggable than mixing
   images from instruments with large domain gaps.

---

## 2. Production Domain Definition

The target inference domain is:

| Property | Value |
|---|---|
| Image format | FITS (16-bit, calibrated: dark + flat subtracted) |
| Sensor class | Cooled monochrome CMOS or CCD (e.g. ZWO ASI2600MM Pro) |
| Pixel scale | 0.5 – 3.0 arcsec/px |
| Focal length | 300 – 1500 mm |
| Exposure time | 0.3 – 2.0 s |
| Sky background | Dark site, low light pollution |
| Scheduling | SkyTrack or equivalent (LEO/GEO passes, known NORAD IDs) |
| Streak morphology | Predominantly linear, 0.5–2 px wide at model input |

**Canonical reference instrument: Atwood Observatory**
- Camera: ZWO ASI2600MM Pro (6248×4176 px, 3.76 µm pixels)
- Focal length: 612 mm
- Plate scale: **1.27 arcsec/px**
- Exposure: 0.5 s
- Site: 43.6735556°N 81.0204722°W, 365 m ASL

At these parameters a typical LEO satellite crossing at 1–2°/s angular velocity
produces a streak of 1,400–2,800 px on the native sensor — after tiling to 400 px
model input, this compresses to 90–180 px diagonal.  Short streaks (<269 px on
native sensor) require either GEO/high-altitude objects or shorter exposures and
are genuinely rare in normal Atwood operations.

Scopes that fall within the production domain: same sensor family, same site
class, focal lengths within ~2× of 612 mm, exposures within ~4× of 0.5 s.
Scopes outside this range (e.g. HST, wide-field 50 mm lenses, space-based
cameras) are not the inference target and should not drive training decisions.

---

## 3. Dataset Inventory

### 3.1 Ground-truth corpus (Atwood Observatory)

All images are FITS, calibrated, from SkyTrack-scheduled passes.
Annotations are COCO-format bounding boxes in native pixel coordinates.

| Session | Date | Streak images | Negatives | NORAD IDs | Streak range (native px) | Status |
|---|---|---|---|---|---|---|
| Night 1 (`Img_20260412_Atwood`) | 2026-04-12 | 578 | 91 | 68 | p10=373, median=624, p90=1003 | ✅ Annotated |
| Night 2 (`Img_20260515_Atwood`) | 2026-05-15 | 204 | 27 | 39 | min=215, median=687, max=1404 | ✅ Annotated |
| Geo session (`Geo_20260520_Atwood`) | 2026-05-20 | 11 | 0 | ~5 | short-ish (GEO slow motion) | ✅ Annotated |
| **Total** | | **793** | **118** | **~107 unique** | | |

Annotation files: `data/annotations/gtimages.json`,
`data/annotations/gtimages_negatives.json`,
`data/annotations/brentimages_20260515.json`,
`data/annotations/brentimages_20260515_negatives.json`,
`data/annotations/geo_20260520.json`.

Raw images: `/Volumes/External/TrainingData/raw/BrentImages/`

### 3.2 Supplemental training data — Frigate (short-band only)

| Property | Value |
|---|---|
| Source | ExoAnalytic Solutions QHY600M — public dataset |
| Format | Processed PNG (2325×1555 px, tiled to 400 px crops) |
| Streak length | 20–80 px in tile coordinates (~short-band only) |
| Annotation status | 191 frames annotated (377 OBBs), 159 negatives |
| Tiled training set | 717 tiles, 655 annotations |
| Role | **Training only, short-band morphology coverage** |
| Domain distance | Moderate — PNG not FITS, different camera/site |

Frigate is included in training because it is the **only available source of
short-band streak examples**.  It is excluded from validation and test sets
because it is not in the production domain.  A diversity-sampled subset
(~150–200 tiles, see §6.2) is preferred over the full 717 tiles to avoid
the training distribution being dominated by a single-night, single-instrument
source.

### 3.3 Secondary benchmark — SatStreaks

| Property | Value |
|---|---|
| Source | Hubble Space Telescope archival + mixed ground-based PNGs (public) |
| Format | JPEG/PNG, 4096×4096 px |
| Role | **Secondary cross-domain benchmark only — never training data** |
| Interpretation | Trend indicator across runs; NOT a production accuracy figure |

SatStreaks is excluded from training because the domain gap (space-based,
no atmosphere, JPEG/PNG format, inconsistent pixel scale) is too large to
reliably generalise to Atwood FITS production images.  The frozen test split
(`test.json`, 308 images) is retained solely to detect regressions across
model versions.  If the SatStreaks benchmark score drops between runs, it
warrants investigation but is not itself a quality gate.

---

## 4. Streak Length Band Definitions

Bands classify streaks by their diagonal bounding-box length in
**original image pixel coordinates** (not model-input coordinates, not
arcseconds).  Both GT annotations and model predictions are evaluated in
original-image space after MMDetection's keep-ratio rescale.

| Band | Threshold | Atwood equivalent | Physical meaning at Atwood |
|---|---|---|---|
| Short | diagonal < 269 px | < 342 arcsec (~5.7') | GEO objects, very high-altitude passes, or short exposures |
| Medium | 269 – 800 px | 342 – 1,016 arcsec (5.7' – 16.9') | High-elevation LEO, MEO objects |
| Long | diagonal > 800 px | > 1,016 arcsec (> 16.9') | Typical LEO pass at moderate elevation |

**These thresholds are a detection-difficulty proxy, not a physically invariant
size classification.**  The same physical event produces different pixel lengths
on instruments with different plate scales.  When comparing metrics across
sources with different sensor configurations, interpret band labels accordingly.

At 400 px model input, the scale factor for Atwood (6248 px wide) is ≈ 0.064.
A "long" streak of 800 px native → ≈ 51 px at model input.  The model never
sees streaks that look "long" in the classical sense; it sees moderately-sized
linear objects in a 400 px window.

---

## 5. Geometry-Based Stratification

The core principle: **stratify the corpus by visual morphology, not by object
identity or capture night**.  This ensures every split contains examples from
across the full difficulty space the model will encounter in production.

### 5.1 Stratification Dimensions

Four dimensions characterise the visual difficulty of a streak annotation:

#### Length (primary)
Diagonal of the COCO `[x, y, w, h]` bounding box in native pixels.
Computed directly from annotations — no image loading required.
Use the three bands defined in §4; sub-divide the long band into
long-moderate (800–1200 px) and long-extreme (>1200 px) if sample size permits.

#### Aspect ratio (thickness proxy)
`max(w, h) / min(w, h)` from the bounding box.
Captures streak sharpness and width:
- **High (> 20)**: clean, thin, well-focused streak — easy for the detector
- **Moderate (5–20)**: typical LEO streak under normal seeing
- **Low (< 5)**: wide/blurry (poor seeing, defocus) or very short streak

Does not require image loading.  A proxy for optical quality and PSF width.

#### SNR / faintness (critical)
Peak streak signal relative to local background noise.
**Requires loading the FITS file.**

Computation: for each annotation, load the native FITS image, extract a region
slightly larger than the bounding box, estimate background as the median of a
surrounding annulus (excluding the bbox), estimate noise as the robust sigma
(MAD × 1.4826) of the background annulus, compute:

```
SNR = (mean_pixel_in_bbox - background_median) / background_sigma
```

Classification:
- **Bright (SNR > 20)**: easily detectable; model reliably finds these
- **Medium (5 < SNR ≤ 20)**: typical production case
- **Faint (SNR ≤ 5)**: near detection threshold; these are the hard cases

**This dimension is the most important gap in the current data strategy.**
Run 3 missed 49 long streaks (16.6% FN rate on the long band).  The missed
streaks are almost certainly the faint ones.  Without stratifying by SNR, a
test set will be biased toward bright examples and will overstate production
recall.

#### Orientation (angle)
Angle of the bbox major axis (degrees from horizontal).
Distribution in the current corpus is already roughly uniform (mean 45.5°).
Include as a sanity check in split construction but do not use as a primary
stratification axis unless a gap is found.

### 5.2 Feature Table

A feature extraction script (`scripts/extract_streak_features.py` — **to be
built**) produces a CSV / JSON feature table with one row per annotation:

```
annotation_id, image_id, session_id, file_name,
length_px, aspect_ratio, angle_deg,
snr,                    # requires FITS loading; null if file not accessible
band,                   # short | medium | long (from §4)
snr_class,              # bright | medium | faint (or null)
aspect_class            # thin | normal | chunky
```

This table is the input to the split construction script.

### 5.3 Split Construction

A stratified split script (`scripts/build_stratified_splits.py` — **to be
built**) takes the feature table and produces geometry-balanced splits:

**Algorithm:**
1. Define cells in the 2D grid of (length_band) × (snr_class).
   Aspect ratio used as a tertiary check, not a primary bin axis.
2. For each cell, collect all annotation IDs in that cell.
3. Assign images (not individual annotations) to splits: when an image
   is assigned to a split, all its annotations go with it.
4. Within each cell, sample proportionally to the target split ratio
   (70 % train / 15 % val / 15 % test) using a fixed random seed.
5. If a cell has fewer than 3 images, assign all to training.

**Output files:**
- `data/annotations/atwood_train.json` — training images from Atwood corpus
- `data/annotations/val_atwood.json` — validation images, geometry-balanced
- `data/annotations/test_atwood.json` — held-out test, geometry-balanced, **frozen**

The Atwood test set is frozen on first construction and never modified.
New Atwood nights are added to the training pool only, after zero-shot
evaluation (see §7).

### 5.4 Expected Split Sizes (current corpus, ~793 images)

With a 70/15/15 split:

| Split | Images | Notes |
|---|---|---|
| `atwood_train.json` | ~555 | Plus Frigate diversity subset |
| `val_atwood.json` | ~120 | Replaces `val.json` as MMDet training signal |
| `test_atwood.json` | ~120 | Primary production quality benchmark; frozen |

These are small.  Every new Atwood night that is annotated and added to the
pool adds ~150–250 images.  After 3–4 more nights the splits will be
meaningfully larger.

---

## 6. Training Set Construction

### 6.1 Composition

The canonical training set for Run 4+ is built by
`scripts/build_training_json.py` from `data/sessions/manifest.yaml`:

| Source | Role | Notes |
|---|---|---|
| `atwood_train.json` | Core ground-truth — all bands | ~555 images from §5.3 split |
| Frigate diversity subset | Short-band morphology | ~150–200 tiles; see §6.2 |

Frigate tiles are appended after the Atwood portion.  Their mix weight in
the manifest can be adjusted with `--mix-ratio frigate:<N>` if the short
band needs oversampling.

### 6.2 Frigate Diversity Subset

Rather than using all 717 Frigate tiles (which come from a single night, single
instrument), select a diversity-maximising subset of ~150–200 tiles that covers:

- Full range of short-streak orientations (0–180°)
- Both thin and moderate aspect ratios
- Both bright and medium SNR examples

A Frigate geometry sampler (`scripts/sample_frigate_tiles.py` — **to be
built**) computes bbox-level geometry features (no FITS loading needed — tiles
are already at model input scale) and runs a greedy diversity selection.

### 6.3 Augmentation for Cross-Scope Generalisation

To make the model robust to similar-but-not-identical scopes, apply these
augmentations to all Atwood training images:

| Augmentation | Parameter range | Rationale |
|---|---|---|
| Scale jitter | ±25% random resize of crops | Simulates pixel scales 0.95–1.6 arcsec/px around Atwood's 1.27 |
| Gaussian blur | σ 0.5–2.0 px | Simulates different seeing, PSF, focus quality |
| Background noise | Poisson at 0.5–2× baseline | Simulates different gain, QE, sky background level |
| Contrast stretch jitter | ±10% of `apply_norm()` parameters | Simulates different photometric conditions |
| Horizontal + vertical flip | p=0.5 each | Already present; keep |

These augmentations are applied in `training/augmentations.py`.  The scale
jitter and blur are **not yet implemented** and are the next augmentation
additions after the split scripts are built.

### 6.4 Synthetic Streak Injection

`scripts/augment_short_medium.py` injects synthetic streaks into existing
backgrounds.  It should be run against `atwood_train.json` to supplement
medium-band coverage and — after adding an `snr_scale` parameter — to inject
faint long streaks targeting the 49 FN gap from Run 3.

Required addition to `training/augmentations.py:SyntheticStreakInject`:
- `snr_scale` parameter (float, 0.1–1.0): multiplies the injected streak
  brightness relative to the current full-brightness injection.
  Values < 0.3 produce near-threshold faint streaks.

---

## 7. New Atwood Night Workflow

When a new Atwood capture night is available:

1. Annotate the night using the standard workflow
   (see `agent_docs/datasets.md` §1 — Workflow for a new capture night).

2. Add to the session manifest (`data/sessions/manifest.yaml`) with
   `split: holdout`.

3. Run zero-shot evaluation **before** adding to training:
   ```bash
   python scripts/zero_shot_eval.py \
       --annotation data/annotations/newnight_YYYYMMDD.json \
       --raw-dir /path/to/night \
       --scope atwood \
       --label "Atwood YYYYMMDD (zero-shot)"
   ```

4. Assign NORAD IDs from the new night to splits.  NORAD IDs that already
   appear in `val_atwood.json` or `test_atwood.json` must go to training
   only.  New NORAD IDs not yet seen can go to any split — assign to training
   unless the val or test pools are significantly smaller than their target
   proportions.

5. Change manifest `split` to `train`.  Rebuild the training JSON:
   ```bash
   python scripts/build_training_json.py \
       --output data/annotations/all_train_nodm_v<N>.json
   ```

---

## 8. New Scope Onboarding

See `agent_docs/datasets.md` §"Adding a New Telescope" for the full workflow.

Summary:
- Night 1 from any new scope is always `split: holdout` — evaluate zero-shot first.
- Promote to `split: train` after annotation of Night 2+ and zero-shot review.
- The Atwood test set (`test_atwood.json`) is Atwood-only and is not used to
  evaluate other scopes.  Use `zero_shot_eval.py` for new-scope quality checks.

---

## 9. Validation Set During Training

During MMDetection training, `val_atwood.json` (once built) replaces `val.json`
as the validation target.  `val.json` (SatStreaks + Atwood N1, legacy mixed file)
should no longer be used as a primary training signal.

Until `val_atwood.json` is built, continue using `val.json` as a monitoring
signal only — interpret val mAP as a trend indicator, not a production accuracy
figure.

---

## 10. Test and Benchmark Hierarchy

| Set | File | Domain | Role | Quality gate? |
|---|---|---|---|---|
| **Atwood test** | `test_atwood.json` | Ground FITS, Atwood | Primary production benchmark | **Yes** |
| SatStreaks benchmark | `test.json` | HST/mixed PNG | Cross-domain trend tracker | No |
| Frigate eval | `frigate_streaks.json` | Ground PNG, QHY600M | Short-band spot-check | No |

A model version passes the quality gate if `test_atwood.json` long-band
recall ≥ 85% at conf ≥ 0.30, IoU ≥ 0.50.  (Threshold to be updated as
the test set grows and the model matures.)

SatStreaks `test.json` is never a quality gate.  If the SatStreaks score drops
between runs, investigate — but do not reintroduce SatStreaks into training to
recover it.

---

## 11. Implementation Status

### Done
- [x] Session manifest (`data/sessions/manifest.yaml`) — all sources, split assignments, policy
- [x] Manifest-driven training JSON builder (`scripts/build_training_json.py`) — mix ratios, include/exclude
- [x] Fine-tune config (`models/dino/streak_dinov3_vitb_400px_ft.py`) — 10× lower LR, loads from run3 best
- [x] Zero-shot evaluation script (`scripts/zero_shot_eval.py`) — per-band recall, decision recommendation
- [x] Synthetic augmentation script (`scripts/augment_short_medium.py`) — medium-band injection
- [x] SatStreaks excluded from training in manifest
- [x] Band threshold documentation — px in original image space, not arcseconds (see §4)
- [x] Run 3 complete — mAP=0.782, P=94.9%, R=83.8%, best checkpoint epoch 13

### To be built (in priority order)
- [x] **`scripts/extract_streak_features.py`** — per-annotation feature table
      (length, aspect_ratio, angle, SNR from FITS).  SNR computed consistently
      from raw FITS pixels across all sessions; falls back to annotation
      attributes when FITS unavailable.  Output: `data/features/atwood_streak_features.csv`.
- [x] **`scripts/build_stratified_splits.py`** — geometry-balanced
      `atwood_train.json`, `val_atwood.json`, `test_atwood.json` from feature table.
      Stratifies by band × snr_class; cells with < 3 images all go to training;
      negatives split randomly at same ratio.
- [x] **`scripts/sample_frigate_tiles.py`** — diversity-maximising subset of
      ~150–200 Frigate tiles using Furthest-Point Sampling in
      length × aspect × angle × tile-position feature space.
      Output: `data/annotations/frigate_diversity_<N>.json`.
- [x] **`snr_scale` in `SyntheticStreakInject`** — faint streak augmentation to
      target the 49 FN long-streak gap from Run 3.  Use `snr_scale=0.1–0.3`
      for near-threshold faint streak injection.
- [x] **Scale jitter and PSF blur augmentation** in `training/augmentations.py`
      `get_train_transforms()`: `RandomScale(±25%)` and
      `GaussianBlur(σ 0.5–2.0 px)` added for cross-scope generalisation.
- [ ] **Run 4** — first training run on geometry-stratified Atwood+Frigate data,
      no SatStreaks.  Evaluate against `test_atwood.json` once built.

### Run 4 prerequisites (all now complete)
1. Run `extract_streak_features.py` → `data/features/atwood_streak_features.csv`
2. Run `build_stratified_splits.py` → `atwood_train.json`, `val_atwood.json`, `test_atwood.json`
3. Run `sample_frigate_tiles.py` → `frigate_diversity_250.json`
4. Build final training JSON:
   ```bash
   python scripts/build_training_json.py \
       --include atwood_20260412 atwood_20260515 atwood_geo_20260520 \
       --output data/annotations/all_train_run4.json
   # Then manually merge with frigate_diversity_250.json if needed,
   # or add frigate_diversity_250 as a manifest entry (split: train).
   ```

---

## 12. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-28 | SatStreaks excluded from training | Domain gap (HST/mixed PNG) too large; risks teaching features that don't transfer to Atwood FITS |
| 2026-05-28 | SatStreaks test.json retained as secondary benchmark | Useful for detecting cross-run regressions; not a production accuracy figure |
| 2026-05-28 | Geometry-based stratification adopted | Visual morphology space is what the model generalises across; NORAD ID splitting controls the wrong variable |
| 2026-05-28 | Frigate retained for short-band training only | Only available source of short-streak morphology; diversity-sampled to prevent single-instrument dominance |
| 2026-05-28 | SNR/faintness added as stratification dimension | Run 3's 49 FN long-streak misses are almost certainly faint; without this dimension, test sets overstate production recall |
| 2026-05-28 | Augmentation (scale jitter, PSF blur) planned for cross-scope generalisation | More predictable than mixing data from instruments with large domain gaps |
