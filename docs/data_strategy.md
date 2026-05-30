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
| Night 1 (`Img_20260412_Atwood`) | 2026-04-12 | 578 | 91 | 68 | p10=373, median=624, p90=1003 | ✅ train |
| Night 2 (`Img_20260515_Atwood`) | 2026-05-15 | 204 | 27 | 39 | min=215, median=687, max=1404 | ✅ train |
| Geo session (`Geo_20260520_Atwood`) | 2026-05-20 | 11 | 0 | ~5 | short-ish (GEO slow motion) | ✅ train |
| Night 3 (`Img_20260527_Atwood`) | 2026-05-27 | 507 | 25 | 70 | mostly long (p50~900px) | ✅ train |
| Night 4 (`Img_20260528_Atwood`) | 2026-05-28 | 175 | 18 | ~40 | heavily long (96% long band) | ✅ train |
| **Total** | | **1,475** | **161** | **~200 unique** | | |

Annotation files in `data/annotations/`:
`gtimages.json`, `brentimages_20260515.json`, `geo_20260520.json`,
`atwood_20260527.json`, `atwood_20260528.json` (plus corresponding `_negatives.json`).

Raw images: `/Volumes/External/TrainingData/raw/BrentImages/`

**Geometry-stratified splits** (rebuilt 2026-05-30 from all 5 nights):
- `atwood_train.json` — 1,129 images, 1,081 annotations (70%)
- `val_atwood.json` — 240 images, 228 annotations (15%)
- `test_atwood.json` — 240 images, 228 annotations (15%)

### 3.2 Supplemental training data — Frigate (cluster-2 only, Run 5+)

| Property | Value |
|---|---|
| Source | ExoAnalytic Solutions QHY600M — public dataset |
| Format | Processed PNG (2325×1555 px) |
| Full corpus | 350 annotated frames, 377 OBBs (191 positive, 159 negative) |
| Usable subset | **Cluster 2: 48 annotations from 9 frames** (≥35px native, AR≥2.0) |
| Tiling | 110px crops → 400px model input (3.64× magnification) |
| Streak at model input | 127–240px (overlaps Atwood short band) |
| Role | **Short-band morphology supplement** |
| Domain distance | Moderate — PNG not FITS, different camera/site |

**Corpus structure (analysed 2026-05-30):** The corpus has a bimodal
length distribution with a natural break at ~25px:
- **Cluster 1** (86%, 323 annotations): <25px, AR~1.0 — near-circular blobs.
  Confirmed to provide no training value: model trained on these in Run 4 showed
  0% short-band recall on Atwood, and 1.6% recall on frigate itself at the
  matching training scale.
- **Cluster 2** (13%, 48 annotations): ≥35px, AR≥2.0 — genuine linear streaks.
  At 3.64× tiling, these appear as 127–240px at model input.

Only cluster 2 is included in Run 5+. Built by `scripts/build_frigate_cluster2.py`.
Annotation file: `data/annotations/frigate_cluster2_tiled_110.json`.

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

### 6.2 Frigate Cluster-2 Subset

Rather than using all 717 tiles or the diversity-sampled set, Run 5+ uses only
the **cluster-2 subset**: annotations with native diagonal ≥35px and aspect
ratio ≥2.0. These are the only tiles that represent genuine linear streak
morphology. The 110px tiling (3.64× magnification) brings them to 127–240px
at model input — directly comparable to Atwood short-band detections.

Key properties of the cluster-2 subset:
- 48 positive annotations from 9 source frames
- 27 negative tiles (blank regions from the same frames)
- Orientation coverage: 30–90° and 150–180° dominant; 0–30° sparse
- Built by `scripts/build_frigate_cluster2.py`

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

3. Export reviewed usable frames into holdout COCO files.  The interactive
   annotator's `brentimages_annotations.json` includes positives, blanks,
   rejected frames, and pending frames; only positives and reviewed blanks
   should enter zero-shot evaluation:
   ```bash
   python scripts/prepare_atwood_holdout.py \
       --input /Volumes/External/TrainingData/raw/BrentImages/Img_YYYYMMDD_Atwood/brentimages_annotations.json \
       --session-id atwood_YYYYMMDD \
       --mirror-external
   ```

4. Run zero-shot evaluation **before** adding to training:
   ```bash
   python scripts/zero_shot_eval.py \
       --annotation data/annotations/atwood_YYYYMMDD.json \
       --negatives data/annotations/atwood_YYYYMMDD_negatives.json \
       --raw-dir /Volumes/External/TrainingData/raw/BrentImages/Img_YYYYMMDD_Atwood \
       --scope atwood_YYYYMMDD \
       --label "Atwood YYYYMMDD (zero-shot)"
   ```

5. Assign NORAD IDs from the new night to splits.  NORAD IDs that already
   appear in `val_atwood.json` or `test_atwood.json` must go to training
   only.  New NORAD IDs not yet seen can go to any split — assign to training
   unless the val or test pools are significantly smaller than their target
   proportions.

6. Change manifest `split` to `train`.  Rebuild the training JSON:
   ```bash
   python scripts/build_training_json.py \
       --output data/annotations/all_train_nodm_v<N>.json
   ```

### 7.1 Current Holdout Nights

The following newly reviewed Atwood nights are intentionally held out for
post-training zero-shot evaluation and must not be promoted to `train` until
their reports are recorded:

| Session | Raw directory | Positives | Annotations | Negatives | Excluded |
|---|---|---:|---:|---:|---:|
| `atwood_20260527` | `Img_20260527_Atwood` | 507 | 559 | 25 | 109 rejected, 168 pending |
| `atwood_20260528` | `Img_20260528_Atwood` | 175 | 185 | 18 | 5 rejected, 301 pending |

Prepared files live in both `data/annotations/` and
`/Volumes/External/TrainingData/annotations/`:
`atwood_20260527.json`, `atwood_20260527_negatives.json`,
`atwood_20260528.json`, and `atwood_20260528_negatives.json`.

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

| Set | File | Images | Domain | Role | Quality gate? |
|---|---|---|---|---|---|
| **Atwood test** | `test_atwood.json` | 240 | Ground FITS, Atwood | Primary production benchmark | **Yes** |
| SatStreaks benchmark | `test.json` | 308 | HST/mixed PNG | Cross-domain trend tracker | No |
| Frigate eval | `frigate_streaks_eval.json` | 350 | Ground PNG, QHY600M | Short-band spot-check | No |

**Quality gate (updated for Run 5 test set):**
A model version passes if `test_atwood.json` produces:
- Long-band recall ≥ 85% at conf ≥ 0.20, IoU ≥ 0.50
- Medium-band recall ≥ 65% at conf ≥ 0.20, IoU ≥ 0.50

(Threshold updated from conf ≥ 0.30 following Run 4 FN analysis showing 45% of
false negatives were detected at conf 0.10–0.29.)

SatStreaks `test.json` is never a quality gate.  If the SatStreaks score drops
between runs, investigate — but do not reintroduce SatStreaks into training to
recover it.

**Note on val/test set size change:** Run 5 val/test sets (240 images each) are
larger than Run 4 (133 images each) because the stratified re-split incorporated
all 5 Atwood nights. Run 4 and Run 5 test results are therefore **not directly
comparable** — treat them as independent evaluations.

---

## 11. Implementation Status

### Done
- [x] Session manifest (`data/sessions/manifest.yaml`) — all sources, split assignments, policy
- [x] Manifest-driven training JSON builder (`scripts/build_training_json.py`) — mix ratios, include/exclude
- [x] Zero-shot evaluation script (`scripts/zero_shot_eval.py`) — per-band recall, decision recommendation
- [x] SatStreaks excluded from training in manifest
- [x] Band threshold documentation — px in original image space, not arcseconds (see §4)
- [x] Run 3 complete — mAP=0.782, P=94.9%, R=83.8%, best checkpoint epoch 13
- [x] `scripts/extract_streak_features.py` — per-annotation SNR/geometry feature table
- [x] `scripts/build_stratified_splits.py` — geometry-balanced splits; `--sessions`/`--append` flags added
- [x] `scripts/build_frigate_cluster2.py` — cluster-2 filtered 110px tiled annotation builder
- [x] `scripts/build_synthetic_short.py` — synthetic short-band streak injection into Atwood backgrounds
- [x] `snr_scale` range in `SyntheticStreakInject` — full brightness range via `--snr-scale-max`
- [x] **Run 4 complete** — ViT-S OBB: mAP@50=0.611 val / 0.518 test; Centerline: val_dice=0.2327
- [x] **Run 4 FN root-cause analysis** — 45% threshold issue, 55% truly missed; SNR not primary driver
- [x] **Frigate corpus analysis** — bimodal distribution; cluster 1 (86% blobs) excluded from Run 5
- [x] **Run 5 dataset built** — all 5 Atwood nights re-stratified; `all_train_run5.json` 2,064 imgs / 1,956 anns
- [x] Holdout nights (atwood_20260527, atwood_20260528) promoted to `split: train` after zero-shot eval

### To be done for Run 5
- [ ] **Train Run 5** — ViT-B backbone on `all_train_run5.json` / `val_atwood.json` on RTX workstation
- [ ] **Lower confidence threshold** — evaluate Run 5 at conf=0.20 (45% of Run 4 FNs recoverable)
- [ ] **Evaluate on `test_atwood.json`** — primary benchmark; goal: medium recall ≥ 65%, long ≥ 85%
- [ ] **Frigate cluster-2 eval** — run `eval_frigate_tiled.py` against Run 5 checkpoint to validate short-band learning

---

## 12. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-28 | SatStreaks excluded from training | Domain gap (HST/mixed PNG) too large; risks teaching features that don't transfer to Atwood FITS |
| 2026-05-28 | SatStreaks test.json retained as secondary benchmark | Useful for detecting cross-run regressions; not a production accuracy figure |
| 2026-05-28 | Geometry-based stratification adopted | Visual morphology space is what the model generalises across; NORAD ID splitting controls the wrong variable |
| 2026-05-28 | Frigate retained for short-band training only | Only available source of short-streak morphology outside Atwood |
| 2026-05-28 | SNR/faintness added as stratification dimension | Run 3's FN long-streak misses likely faint; without this dimension, test sets overstate production recall |
| 2026-05-29 | Run 4 OBB medium-band recall (49%) is primary failure mode on test_atwood | Medium band is 67% of Atwood annotations; investigate FN root cause before retraining |
| 2026-05-29 | Holdout nights (atwood_20260527, atwood_20260528) kept out of training until zero-shot eval recorded | Policy: never promote a night to training before its holdout eval report is committed |
| 2026-05-30 | Run 4 FN root-cause: 45% threshold issue, 55% truly missed; SNR not primary driver | 53% of FNs are bright (SNR>20); faint augmentation is low-priority; confidence calibration is key |
| 2026-05-30 | Frigate cluster-1 (86% blobs, <25px, AR~1) excluded from Run 5 | Model trained on cluster-1 showed 0% Atwood short-band recall and 1.6% frigate recall at matching scale |
| 2026-05-30 | Frigate cluster-2 (13%, ≥35px, AR≥2) tiled at 110px for 3.64× zoom | At model input, 35–66px streaks appear as 127–240px — comparable to Atwood short band |
| 2026-05-30 | All 5 Atwood nights re-stratified together | Appending long-heavy holdout nights without re-stratification would skew distribution; val/test should also reflect new nights |
| 2026-05-30 | Synthetic short-band injection added (380 images, snr_scale 0.2–1.0) | Only 21 real Atwood short annotations across 5 nights (1.4%); synthetic fill raises short band to 23% of training |
| 2026-05-30 | conf threshold target lowered to 0.20 for Run 5 eval | 45% of Run 4 FNs had correct-location predictions at conf 0.10–0.29; threshold drop recovers them at no training cost |
