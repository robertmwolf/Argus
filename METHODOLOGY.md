# ARGUS Methodology
### Automated Recognition and Grading of Unidentified Streaks

**Version:** 2026-05-16  
**Benchmark commit:** see `results/multi_method_benchmark.json`  
**Code:** `inference/`, `src/detection/`, `inference/confidence.py`, `inference/postprocess.py`  
**Cite as:** cite this repository per its `CITATION.cff`

ARGUS detects satellite streak artifacts in astronomical FITS images using five
independent detectors fused into a Unified Confidence Score (UCS). On a 308-image
test set (SatStreaks JPEG exports), the ensemble achieves F1 = 42.3% (P = 29.9%,
R = 72.1%), while the primary ML detector (DINOv2 ViT-B / DINO-DETR) achieves
mAP@0.5 = 75.5% and recall = 89.3%. On long streaks (≥ 1,000 px), ensemble F1
reaches 49.0%.

---

## Table of Contents

1. [Problem Statement and Scope](#1-problem-statement-and-scope)
2. [Datasets](#2-datasets)
3. [Detector Architectures](#3-detector-architectures)
4. [Post-Processing Pipeline](#4-post-processing-pipeline)
5. [Unified Confidence Score](#5-unified-confidence-score)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Results](#7-results)
8. [Comparison with Prior Work](#8-comparison-with-prior-work)
9. [Limitations and Future Work](#9-limitations-and-future-work)
10. [Reproducibility Checklist](#10-reproducibility-checklist)
11. [References](#11-references)

---

## 1. Problem Statement and Scope

### Detection task

Single-class oriented bounding box (OBB) detection of linear streak artifacts in
2D astronomical images. A streak is a bright, linear trail produced when a
satellite or other rapidly-moving object crosses the field during a single
exposure. The task requires localising each streak with a tight OBB and estimating
its orientation angle.

**Why non-trivial:**
- Streaks span aspect ratios up to ~3,000:1 (length to width).
- Streak brightness varies by orders of magnitude across different sensors,
  exposure times, and target altitude.
- Labeled training examples are scarce relative to background diversity (stars,
  galaxies, diffuse nebulae, JPEG compression artifacts).
- At long exposures, streaks curve — linear models underfit; at short exposures
  they are only a few pixels wide, making detection geometry-sensitive.

### Evaluation metric

Intersection-over-Union (IoU) ≥ 0.5 between predicted bounding boxes and COCO
axis-aligned ground-truth annotations. Rotated-OBB IoU is used within the
postprocessing pipeline; axis-aligned IoU is used for final benchmark evaluation
because the ground-truth annotations are axis-aligned.

### Out of scope for this document

Cross-identification (TLE catalog lookup, SGP4 propagation, multi-factor
confidence scoring) is a downstream module evaluated separately. It is not
included in the detection metrics reported here.

---

## 2. Datasets

### 2.1 SatStreaks (Primary Evaluation)

**Origin:** Citizen-science labelling sweep of HST archival images and ground-based
observations, released as the SatStreaks benchmark at CRV 2024. The full dataset
covers 114,607 HST images and 233 ground-based images.

**Format:** PNG/JPEG exports — **not raw FITS pixel data.** This is the single
most important caveat for cross-study comparison. JPEG compression alters the
pixel-value distribution compared to raw 16-bit FITS files and degrades the signal
for classical threshold-based detectors (see §3.1 and §3.2).

**ARGUS test split:** `scripts/merge_annotations.py --seed 42 --val-fraction 0.2`
converts segmentation masks to COCO axis-aligned bounding boxes and produces a
fixed, reproducible split. The test set used for all benchmark results in this
document contains **308 images with 308 ground-truth streaks**.

**Streak-length distribution:** 92% of test-set streaks are ≥ 1,000 px (long
streaks dominate). Short-streak (<400 px) and medium-streak (400–999 px)
performance cannot be reliably assessed from this test set alone.

### 2.2 GTImages (Cross-ID Benchmark — Not Used for Detection Evaluation)

**Origin:** 759 FITS images from a single-site ground station (Ontario, Canada,
43.67°N 81.02°W) captured with SkyTrack 1.9.8. Raw 16-bit, 6248×4176 px, 0.5 s
exposures with ASTAP-solved WCS sidecars.

**Composition:** 593 usable labelled streak images; 93 no-streak images; 68 unique
NORAD IDs (79% Starlink). Median streak length: 624 px (p10 = 373, p90 = 1003).

**Why excluded from detection benchmarking:** Single-night, single-site, minimal
background diversity. GTImages is used for cross-identification accuracy only (SGP4
residual scoring) and as a negative-example source during training. Its raw FITS
format does however make it the correct domain for evaluating ASTRiDE and the
OpenCV classical detectors.

### 2.3 Training Data

The merged training corpus consists of the SatStreaks train split (~2,460 images,
PNG/JPEG) and GTImages labelled and negative examples (shuffled, seed 42). YOLO
is trained exclusively on tiled 640 px crops derived from SatStreaks (3,023 source
images producing ~14,385 tiles).

### 2.4 Dataset Comparability Note

Results from ARGUS are **not directly comparable** to results from StreakMind or
ASTRiDE without the following caveats:

| Factor | ARGUS | StreakMind | Comparable? |
|--------|-------|------------|-------------|
| Test domain | JPEG exports (SatStreaks) | Raw FITS | No |
| IoU threshold | 0.5 | 0.8 | No |
| Test set size | 308 images / 308 streaks | 110 real streaks | No |
| Mean streak length | ≫ 1,000 px (92% are long) | 203.5 px | No |
| Multi-frame association | No | Yes | Design difference |

YOLO metrics within ARGUS are also not directly comparable to DINOv2 metrics:
YOLO is evaluated on its native tiled validation split (604 source images → ~2,881
tiles), whereas DINOv2 is evaluated on the full-image COCO test set. Tiled IoU
matching inflates mAP relative to full-image evaluation.

ASTRiDE cannot be meaningfully evaluated on JPEG exports because JPEG compression
alters the pixel distribution that its sigma-threshold contour detection relies on.
Absence of ASTRiDE results on the SatStreaks test set is expected, not a gap.

---

## 3. Detector Architectures

ARGUS runs five detectors independently on every image and collects all outputs
into a single pool before post-processing.

### 3.1 ASTRiDE — Phase 0 Classical Baseline

**Implementation:** `src/detection/classical_detector.py`  
**When active:** Always (on raw FITS input)

**Algorithm:**

1. SEP (Source Extractor in Python) background subtraction using a mesh-based
   background model.
2. Raw FITS pixel data thresholded at `contour_threshold = 3.0σ` above the local
   background to produce a binary image.
3. ASTRiDE boundary-tracing extracts object contours and computes morphological
   parameters, most importantly `shape_factor` (elongation metric; lower = more
   streak-like).
4. Shape filtering: retain detections with shape_factor below the ASTRiDE default
   threshold.
5. Minimum length cutoff: 20 px.
6. Endpoints extracted by projecting each contour along its principal axis.

**Relationship to Kim et al. 2017:** ARGUS follows the same boundary-tracing +
shape_factor filtering pipeline as the reference implementation. The primary
difference is ensemble integration: in ARGUS, ASTRiDE's output is one of five
detector streams, its effective confidence is capped at 0.6 (see §5.2), and its
contribution to the Unified Confidence Score is weighted by its empirical F-0.5
reliability score.

**JPEG limitation:** This detector requires raw integer-valued FITS pixel data.
On JPEG-compressed SatStreaks exports the sigma threshold and shape_factor criteria
behave differently because JPEG compression redistributes the pixel-value
distribution. ASTRiDE is therefore not evaluated on the primary benchmark (§7)
and should be assessed on GTImages raw FITS only.

### 3.2 OpenCV Connected-Components Detector

**Implementation:** `_run_classical_detector()` in `inference/pipeline.py`  
**When active:** Always

**Algorithm:**

1. Threshold the top 0.5% of pixel values in the Z-score normalised uint8 image.
   Rationale: satellite streaks are among the brightest structures in the image;
   the 0.5% threshold is dataset-agnostic and requires no per-image tuning.
2. Morphological closing with a rectangular kernel (size configured for typical
   streak width) to bridge short gaps caused by uneven streak brightness.
3. Connected-component labelling (8-connectivity).
4. Retain components whose long-axis length ≥ 80 px and aspect ratio ≥ 5.
   Rationale: stars and compact sources have aspect ratio close to 1; the ≥ 5
   threshold discriminates linear structures without setting an angle-dependent
   length threshold.
5. PCA on each retained component's pixel coordinates gives the principal axis
   (streak direction) and the two endpoint positions.

**Performance on JPEG test set:** P = 1.4%, R = 1.0%, F1 = 1.1%. JPEG compression
smears pixel values, causing the 0.5% brightness threshold to capture compressed
star halos and noise rather than genuine streaks. This detector contributes more
meaningfully on raw FITS images (uncompressed, 16-bit) where the top-0.5%
threshold correctly isolates only the brightest structures.

### 3.3 YOLO11n-OBB

**Implementation:** Ultralytics YOLO11 with Oriented Bounding Box (OBB) head  
**Variants:**
- **Full dataset** (`weights/run_full_yolo_obb/run/weights/best.pt`, ~5.4 MB) — active when weights file is present; reported as the primary YOLO result
- **Dev subset** (`weights/yolo_tiled/run/weights/best.pt`) — fallback; lower performance, not reported in headline results

**Training (full dataset):**
- Training images: 3,023 source images from SatStreaks train split, tiled to
  640 × 640 px with overlap → approximately 14,385 tiles
- 15 epochs; best checkpoint at epoch 13
- Hardware: Apple M3 CPU (~9 hours)

**Validation (full dataset, tiled protocol):**

| Metric | Value |
|--------|-------|
| mAP@0.5 | 67.3% |
| mAP@0.75 | 57.4% |
| Precision | 57.2% |
| Recall | 84.6% |
| F1 | 68.2% |

> **Protocol note:** These metrics are from YOLO's native tiled validation split
> (604 source images → ~2,881 tiles at 640 px) using the same tiling scheme as
> training. Tiled IoU matching inflates mAP relative to full-image evaluation. The
> YOLO numbers are **not directly comparable** to the DINOv2 or UCS results in §7,
> which use the full-image COCO protocol.

**Relationship to StreakMind:** StreakMind (arXiv 2605.03429) also uses YOLO11 OBB
as its detection backbone. The architectures are similar; the key differences are:

- ARGUS trains on JPEG/PNG exports; StreakMind trains on raw FITS.
- StreakMind adds inter-frame association to reject single-frame spurious detections.
- StreakMind's training set includes explicit no-streak (background-only) images;
  ARGUS's YOLO training set does not.
- StreakMind evaluates at IoU = 0.8; ARGUS at IoU = 0.5.

Direct comparison of P/R numbers is not valid (see §2.4).

### 3.4 DINOv2 ViT-B/16 + DINO-DETR (Primary ML Detector)

**Config:** `models/dino/streak_dinov3_vitb.py`  
**Checkpoint:** `weights/dinov3_vitb_augmented/best_coco_bbox_mAP_epoch_10.pth` (~330 MB)  
**When active:** Always (primary detector)

#### 3.4.1 Backbone: DINOv2 ViT-B/16

**Naming clarification:** ARGUS refers to this model internally as "DINOv3"
(codebase naming convention). The underlying backbone is **DINOv2** (Oquab et al.
2024, arXiv 2304.07193) — this is not a distinct public Meta AI release.

| Property | Value |
|----------|-------|
| Architecture | Vision Transformer, ViT-B/16 |
| Pre-training | Self-supervised (DINO + iBOT objectives) |
| Training data | LVD-142M (curated from 1.2B web images) |
| Parameters | 86 M |
| Embed dimension | 768 |
| Patch size | 16 px |
| Checkpoint file | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| Training state | **Entirely frozen** (`requires_grad=False`, `lr_mult=0.0`) |

**Why LVD-142M over SAT-493M:** Night-sky FITS images have near-black backgrounds,
PSF-shaped point sources, and thin high-aspect-ratio bright lines. The SAT-493M
checkpoint uses pixel normalization tuned for colorful terrestrial Earth-observation
imagery (mean = (0.430, 0.411, 0.296)) — a systematic domain mismatch for
dark-field astronomical data. LVD-142M is domain-neutral and uses standard ImageNet
normalization (mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)).

**Feasibility probe (Phase A):** Cosine dissimilarity between DINOv2 ViT-B features
of streak patches vs. background patches = **0.095** (gate: > 0.05 → PASS). This
confirms that the frozen backbone separates streak and background semantics before
any task-specific fine-tuning.

#### 3.4.2 PatchToPyramid Adapter

DINOv2 ViT is isotropic — it produces flat feature maps at a single stride (H/16,
W/16), not a feature pyramid. The Deformable DETR neck requires multi-scale
features. The `PatchToPyramid` adapter in `models/dino/dinov3_adapter.py` bridges
this gap:

```
DINOv2 ViT-B/16
  ↓  get_intermediate_layers(x, n=4, reshape=True)
  →  4 × (B, 768, H/16, W/16) feature maps at 4 depths
  ↓  PatchToPyramid adapter
     - 1×1 conv projections: 768 → 256 channels (each level)
     - Bilinear upsample to produce [H/8, H/16, H/32, H/64] strides
  →  FPN-compatible feature pyramid
  ↓  Deformable DETR neck + DINO-DETR head
```

Trainable parameters in adapter: ~4 M (1×1 convolutions only).

#### 3.4.3 DINO-DETR Detection Head

**Reference:** Zhang et al. 2022 (arXiv 2203.03605), ICCV 2023.

DINO-DETR extends DETR with three key innovations:

1. **Contrastive denoising:** Adds both positive and negative noised copies of
   ground-truth boxes during training, using a contrastive loss to suppress
   duplicate predictions.
2. **Mixed query selection:** Hybrid anchor box initialization (part static,
   part from encoder output), improving convergence speed.
3. **Look Forward Twice (LFT):** Refined box prediction that reuses intermediate
   decoder features.

| Property | Value |
|----------|-------|
| Detection class | Single class (streak vs. background) |
| Input resolution | 1,280 px longest edge (256 px in fast mode) |
| Confidence threshold | 0.05 (default; configurable via `CONFIDENCE_THRESHOLD`) |
| Trainable parameters | ~40 M (adapter + head; backbone contributes 0) |

#### 3.4.4 Training Protocol

| Stage | Config | Epochs | Hardware | Best mAP@0.5 |
|-------|--------|--------|----------|--------------|
| Phase C | Frozen ViT-B, dev subset (50 images) | 50 | Mac M3 MPS | 0.274 (dev subset) |
| Phase C² | Frozen ViT-B, full merged dataset | 4 (augmented from ep 4) | Mac M3 MPS | **0.74** (test.json) |
| Phase D | Frozen ViT-L, full dataset | 50 (PENDING) | RTX 5070 Ti | TBD |

Augmentation pipeline: Albumentations transforms + synthetic streak injection
(`training/augmentations.py`). Best checkpoint: epoch 10 of augmented run.

**Validation metrics (test.json):**
- mAP@0.5:0.95 = 35.5%
- mAP@0.5 = 53.2%

Note: the mAP@0.5 = 0.74 result above is from Phase C² evaluation on `test.json`;
the 53.2% figure is the validation split result from the augmented Phase C² run.
These are from different evaluation points in the same training progression.

#### 3.4.5 Co-DINO Context

Co-DINO (Zong et al. 2022, arXiv 2211.12860) collaborative hybrid training
(auxiliary ATSS + Faster R-CNN heads with one-to-many label assignments) was used
to initialise the now-archived Swin-T backbone path
(`models/dino/streak_codino_swin_t.py`). The active DINOv2 path uses DINO-DETR
directly without Co-DINO auxiliary heads. The Swin-T/Co-DINO path is retained in
the repository for ablation reference; its benchmark result (mAP@0.5 = 0.19) is
included in §8.

#### 3.4.6 Test-Time Augmentation (TTA)

When `TTA_ENABLED=true`, inference also runs on horizontal and vertical flips;
bounding boxes are mapped back to original coordinates before postprocessing.
TTA is enabled by default and contributes to the 2,969 raw predictions reported
for DINOv2 ViT-B in §7.

---

## 4. Post-Processing Pipeline

This pipeline applies to all five detector outputs after initial inference.

### 4.1 Image Normalisation

Raw FITS pixel data (any bit depth) is normalised before passing to ML detectors:

1. Compute Z-score: subtract the pixel mean, divide by the pixel standard deviation.
2. Rescale to uint8 [0, 255], clipping at ±3σ. Pixels brighter than 3σ above the
   mean are clipped to 255; pixels dimmer than 3σ below the mean are clipped to 0.
3. Convert single-channel science image to 3-channel uint8 `(H, W, 3)` (RGB-like)
   so DINOv2 receives the expected tensor shape.

**Why Z-score over histogram equalization:** Z-score normalisation is linear —
it preserves relative brightness differences between stars, streaks, and background.
Histogram equalization redistributes pixel values to achieve uniform density,
which introduces artificial contrast that confuses brightness-threshold-based
classical detectors and changes the apparent signal-to-noise ratio of faint streaks.

### 4.2 Radon Transform Angle Refinement

**Implementation:** `inference/postprocess.py::refine_angle()`  
**Source attribution:** Radon refinement approach follows StreakMind (arXiv 2605.03429).

DINOv2 produces axis-aligned bounding boxes — streak orientation is not directly
predicted. The initial angle estimate `atan2(height, width)` can carry up to ±45°
error. Radon refinement achieves a mean angle error of **0.018°** on the test set.

**Step-by-step algorithm:**

1. Crop the image region inside the DINOv2 bounding box.

2. Convert to float32 greyscale (average over RGB channels if 3-channel).

3. **Subtract image median; clip negatives to zero.**
   ```python
   crop = np.clip(crop - np.median(crop), 0.0, None)
   ```
   *Why this is essential:* Without background subtraction, the DC sky level
   (~120 counts in a typical normalised uint8 image) dominates the Radon
   sinogram variance at every angle. The variance maximum gets pulled toward
   the axis most compressed by the crop geometry (typically 90° for tall narrow
   crops), producing a systematically biased angle estimate. Subtracting the
   median removes the DC component so that only the streak signal drives the
   variance maximisation.

4. **Downsample large crops to ≤ 512 px** (longest side, bilinear interpolation).
   DINO bboxes scaled from a 1,280 px inference pass on a 6,000 px sensor can
   produce crops of 2,000–3,000 px. Radon on these with ~30 angles takes many
   minutes on CPU; downsampling to 512 px reduces this to under 1 second while
   preserving sub-degree angular precision.

5. **Compute the Radon sinogram** over a ±15° window around the initial angle
   estimate (`angle_search_range = 15.0°`, step = 1°):
   ```python
   radon_center = 90.0 - initial_angle
   radon_angles = np.arange(radon_center - 15.0, radon_center + 15.0 + 1.0, 1.0)
   sinogram = radon(crop, theta=radon_angles, circle=False)
   ```

6. **Select sinogram column with maximum variance:**
   ```python
   best_idx = int(np.argmax(sinogram.var(axis=0)))
   ```
   *Why variance maximisation works:* `skimage.transform.radon` at angle θ
   rotates the image by −θ and sums columns, integrating along lines
   perpendicular to θ. When θ = 90° − φ_streak, the streak projects onto a
   single column with a sharp bright peak (high variance). At any other angle,
   the streak smears across multiple columns (low variance).

7. **Convert back to streak angle:**
   ```
   φ_streak = (90° − θ_radon_best) mod 180°
   ```

**Coordinate convention:** skimage Radon at angle θ integrates along lines
perpendicular to θ, so the sinogram column with maximum variance corresponds to
the projection perpendicular to the streak — hence `φ_streak = 90° − θ_radon`.

**Measured performance:** Mean angle error = 0.018° on the 308-image test set.
At 1,000 px streak length, this corresponds to 0.3 px endpoint displacement —
effectively exact for localisation purposes.

**CPU-only:** `skimage.transform.radon` is NumPy-backed. Do not attempt to move
Radon computation to MPS or CUDA.

### 4.3 Oriented Bounding Box Construction

**Implementation:** `inference/postprocess.py::bbox_to_obb()`

Given the axis-aligned DINOv2 bounding box `[x1, y1, x2, y2]` and the
Radon-refined angle `θ`, the OBB is constructed as:

```
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
bw = |x2 - x1|,  bh = |y2 - y1|

w (long axis)  = bw·|cos θ| + bh·|sin θ|
h (short axis) = bw·|sin θ| + bh·|cos θ|
```

This gives the OBB dimensions consistent with the axis-aligned bounding box at
the given angle: the OBB just encloses the axis-aligned box when rotated by θ.

### 4.4 Streak Extent Tracing

**Implementation:** `inference/postprocess.py::extend_obb_to_streak_extent()`

DINOv2 bounding boxes frequently cover only a portion of a long streak — the
detector captures the highest-confidence region, not the full extent. This
function traces the streak axis across the entire image to recover the true
endpoints.

**Step-by-step algorithm:**

1. **Parameterise the streak axis** as `(cx + t·cos θ, cy + t·sin θ)` where `t`
   is distance from the OBB centre along the streak direction. Compute the full
   t-range that keeps the parameterised point inside the image.

2. **Vectorised strip sampling:** For each integer step `t` along the axis, sample
   a perpendicular strip of ±8 px (parameter `sample_halfwidth = 8`):
   ```python
   _strip_max = gray[_yi, _xi].max(axis=1)   # max per strip, not mean
   ```
   Using the strip maximum (not mean) lets a 1–2 px wide streak clear the
   threshold even when the OBB centre is slightly offset from the actual streak
   axis.

3. **Threshold:** Mark position `t` as "bright" if `strip_max > background + 3σ`
   where `background = image median` and `σ = image std`
   (`threshold_sigma = 3.0`).

4. **Group contiguous bright positions** into runs with gap tolerance = 5 px.

5. **Select the run containing t = 0** (the OBB centre is on the streak — DINOv2
   is guaranteed to have detected this region):
   ```python
   centre_runs = [(s, e) for s, e in runs if s <= 0.0 <= e]
   ```
   *Why t = 0 selection:* Selecting by containment of the OBB centre prevents
   isolated noise spikes beyond the streak tip from inflating the endpoint
   position — a critical design choice for reliable streak length measurement.
   If t = 0 is not bright (rare), fall back to the longest run.

6. **Never shrink:** If the candidate run would reduce `w` below the original OBB
   width, keep the original OBB unchanged.

7. **Update OBB centre and length** to match the selected run:
   ```python
   new_w  = t_end - t_start
   new_cx = cx + (t_start + t_end) / 2 * cos_a
   new_cy = cy + (t_start + t_end) / 2 * sin_a
   ```

### 4.5 Per-Detector NMS

Within each detector's output, oriented bounding boxes are converted to Shapely
polygons. Greedy NMS (sorted by confidence descending) suppresses any detection
whose rotated-IoU with a higher-confidence kept detection exceeds 0.5. This
collapses TTA's three passes and each classical detector's duplicate firings to at
most one box per streak per method.

### 4.6 Cross-Detector Grouping (IoU + IoMin)

After per-detector NMS, detections from **different** methods are grouped into
shared `streak_id` groups rather than suppressed.

**Overlap criteria (OR logic):**
- Rotated-IoU ≥ 0.5, **or**
- IoMin ≥ 0.3

where IoMin (Intersection-over-Minimum) is defined as:
```
IoMin(A, B) = area(A ∩ B) / min(area(A), area(B))
```

**Why IoMin is necessary:** For a thin streak (e.g., 5 × 500 px), a 3 px lateral
offset between two detectors of the same physical streak causes rotated-IoU to
drop to ~0.25 (below the 0.5 threshold), while the two boxes still cover the same
streak almost completely. IoMin in this case ≈ 1.0 because the intersection covers
nearly all of the smaller box. Using IoU alone would assign different `streak_id`
values to the same physical object.

**Nothing is discarded:** All per-method detections are preserved within each
group. The frontend exposes per-method agreement as a quality signal
("3 of 4 detectors agree on this streak").

**Compression effect on test set:**  
3,192 individual predictions from all detectors → **742 streak-level groups**
after cross-detector grouping.

---

## 5. Unified Confidence Score

**Implementation:** `inference/confidence.py::compute_unified_confidence()`

### 5.1 Motivation

DINOv2 ViT-B in isolation (confidence threshold 0.05) produces:
- Recall: 89.3% (finds nearly all streaks)
- Precision: 9.3% (2,694 false positives across 308 images, ~8.7 FP/image)

The false positives are primarily elongated cloud structures and diffuse nebulae
that DINOv2's features generalise to. The UCS fuses all five detector outputs to
raise precision without catastrophic recall loss.

### 5.2 Formula

The UCS is computed in five steps using per-detector profiles from
`DETECTOR_PROFILES` in `inference/confidence.py`.

**Step 1 — F-0.5 reliability weight:**

```
w_i = (1 + 0.5²) · P_i · R_i / (0.5² · P_i + R_i)
    = 1.25 · P_i · R_i / (0.25 · P_i + R_i)
```

Beta = 0.5 gives a precision-heavy weight: a detector with high precision
contributes more than one with high recall at the same F1. This reflects the
goal of raising precision while the recall cost is acceptable.

**Step 2 — Confidence ceiling:**

```
eff_i = min(conf_i, ceiling_i)    if ceiling_i is set
eff_i = conf_i                    otherwise
```

The ceiling addresses detectors whose raw confidence magnitude is miscalibrated.
ASTRiDE routinely reports confidence ≥ 0.95 on false positives; its ceiling is
set to 0.6. With a ceiling, the formula trusts that the detector *fired* (presence
of the detection), but not *how confidently* it claims to have detected a streak
(magnitude of the confidence).

**Step 3 — Weighted Noisy-OR combination:**

```
P_weighted = 1 − ∏_i (1 − w_i · eff_i)
```

Each detector's weighted effective confidence is treated as an independent
probabilistic vote for the streak existing. Multiple agreeing high-weight detectors
push the score toward 1. A single low-weight detector with high confidence can
contribute at most `w_i · ceiling_i`.

**Step 4 — False-negative adjustment:**

```
fn_penalty = (1/n) · Σ_i recall_i · max(0, 0.5 − eff_i)
score_fn   = P_weighted · (1 − 0.2 · fn_penalty)
```

When a high-recall detector has low effective confidence (below 0.5), its silence
carries information — the streak may not exist. The coefficient 0.2 keeps this
adjustment mild; even a high-recall detector completely silent (eff = 0) reduces
the score by at most ~14%.

**Step 5 — Divergence penalty:**

```
divergence      = std(eff_i)                         [over all detectors in group]
score_final     = min(0.99, score_fn · (1 − 0.15 · divergence))
```

When detectors strongly disagree (high variance in effective confidences), the
ensemble cannot resolve the ambiguity. The coefficient 0.15 is mild — complete
disagreement (std ≈ 0.5) reduces the score by only 7.5%.

### 5.3 Detector Profiles

Current `DETECTOR_PROFILES` from `inference/confidence.py`:

| Key | Detector | Precision | Recall | F-0.5 weight | Ceiling | Source |
|-----|----------|-----------|--------|--------------|---------|--------|
| `tiny` | DINO Swin-T | 0.667 | 0.733 | 0.690 | None | Phase 8 measured |
| `yolo` | YOLO11-OBB | 0.632 | 0.400 | 0.556 | None | Phase 8 measured |
| `dinov3_vitb` | DINOv3 ViT-B | 0.80 | 0.78 | 0.793 | None | **Estimated** from mAP@0.5=0.74; update post Phase D |
| `dinov3_vitl` | DINOv3 ViT-L | 0.85 | 0.82 | 0.842 | None | **Estimated** Phase D target |
| `large` | DINO Swin-L | 0.75 | 0.75 | 0.750 | None | **Estimated** pre-Phase D |
| `astride` | ASTRiDE | 0.50 | 0.70 | 0.543 | 0.6 | **Estimated**; ceiling set for miscalibrated FP confidence |

> **Important:** The `dinov3_vitb` profile values (P=0.80, R=0.78) are *estimated*
> and do not reflect the benchmark results in §7 (P=9.3%, R=89.3%). The benchmark
> results are measured at confidence threshold 0.05 on the full test set; the
> profile values represent a different operating point used to weight the UCS
> formula. After Phase D evaluation, update `dinov3_vitb` with measured values
> from `eval/benchmark.py`.

### 5.4 Updating After Training

After any new training run, update the relevant `DetectorProfile` in
`inference/confidence.py` with measured P and R from `eval/benchmark.py`. Stale
values silently under- or over-weight a detector's contribution. See the README
"Updating Detector Profiles After Training" section for the mechanical steps.

---

## 6. Evaluation Methodology

### 6.1 Metrics

| Metric | Definition |
|--------|------------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 · P · R / (P + R) |
| mAP@0.5 | COCO-protocol mean average precision at IoU ≥ 0.5 |
| mAP@0.75 | COCO-protocol mean average precision at IoU ≥ 0.75 |
| Per-band F1 | F1 computed separately for short (<400 px), medium (400–999 px), and long (≥1,000 px) streaks |

**IoU matching:** Axis-aligned predicted bounding boxes vs. axis-aligned COCO
ground-truth annotations. For streaks with ground-truth height < 5 px, axis-aligned
IoU is used rather than rotated-OBB IoU to avoid scoring artifacts from near-zero-
area ground-truth polygons.

### 6.2 Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Confidence threshold | 0.05 |
| Per-detector NMS IoU | 0.5 |
| Cross-detector grouping | IoU ≥ 0.5 or IoMin ≥ 0.3 |
| Test set | 308 images, 308 GT streaks (SatStreaks, JPEG) |
| Benchmark date | 2026-05-16 |
| Results file | `results/multi_method_benchmark.json` |
| Evaluation code | `eval/benchmark.py` |

### 6.3 Caveats

- YOLO is evaluated on its own tiled val split (604 source images, ~2,881 640 px
  tiles), not the 308-image COCO test set. The two protocols are not directly
  comparable.
- ASTRiDE is not evaluated (JPEG test set; see §3.1).
- The test set is 92% long streaks (≥ 1,000 px). Short and medium F1 values of
  0% reflect this distribution, not detection capability at those lengths.
- The UCS threshold is not separately calibrated: 0.05 is inherited from the DINO
  confidence threshold. Platt scaling or isotonic regression on the UCS score
  would improve calibration.

---

## 7. Results

### 7.1 Per-Method Results

Benchmark: 308 images, 308 GT streaks, SatStreaks JPEG test set, 2026-05-16.

| Detector | Protocol | Precision | Recall | F1 | mAP@0.5 | mAP@0.75 | n preds |
|----------|----------|----------:|-------:|---:|---------:|---------:|--------:|
| **Unified Confidence Score** | COCO full-image | **29.9%** | 72.1% | **42.3%** | 40.6% | 31.8% | 742 |
| DINOv2 ViT-B (epoch 10) | COCO full-image | 9.3% | **89.3%** | 16.8% | **75.5%** | **59.4%** | 2,969 |
| YOLO11n-OBB full | YOLO tiled val† | 57.2% | 84.6% | 68.2% | 67.3% | 57.4% | — |
| OpenCV connected-comp | COCO full-image | 1.4% | 1.0% | 1.1% | 0.01% | 0.01% | 223 |
| ASTRiDE | JPEG (not applicable) | — | — | — | — | — | — |
| DINOv3 ViT-L (Phase D) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Co-DINO Swin-T (archived) | COCO full-image | — | — | — | 19.0% | 16.7% | — |

† YOLO evaluated on tiled val split — not directly comparable to COCO full-image
evaluation above.

### 7.2 Confusion Matrices (IoU ≥ 0.5)

**Unified Confidence Score:**

| | Predicted + | Predicted − |
|---|---:|---:|
| **Actual +** | TP ≈ 222 | FN ≈ 86 |
| **Actual −** | FP ≈ 520 | TN = n/a |

*Analysis:* The 520 FPs are primarily DINOv2 false positives on elongated cloud and
galaxy features that survive grouping because no second detector fired on them (no
IoU/IoMin partner to form a multi-detector group). Without a corroborating
detection, the UCS assigns a non-trivial score based on DINOv2's evidence alone.
The 86 FNs are streaks where DINOv2 also missed (33 raw DINOv2 FNs) plus
additional cases where grouping shifted the consensus detection below the IoU = 0.5
threshold at the grouped centre.

**DINOv2 ViT-B:**

| | Predicted + | Predicted − |
|---|---:|---:|
| **Actual +** | TP ≈ 275 | FN ≈ 33 |
| **Actual −** | FP ≈ 2,694 | TN = n/a |

*Analysis:* The 33 FNs are short or faint streaks below the 0.05 confidence floor
at 1,280 px input resolution. The 2,694 FPs are DINOv2's generalisation to any
elongated bright structure — the frozen backbone's features, pre-trained on natural
images, do not discriminate streaks from cloud filaments or galaxy arms.

**OpenCV:**

| | Predicted + | Predicted − |
|---|---:|---:|
| **Actual +** | TP ≈ 3 | FN ≈ 305 |
| **Actual −** | FP ≈ 220 | TN = n/a |

*Analysis:* JPEG compression destroys the brightness distribution that the 0.5%
threshold relies on. The 3 TPs are extremely bright, very long streaks that survived
compression with sufficient contrast. The 220 FPs are compressed star halos and
noise structures that exceed the 0.5% brightness threshold.

### 7.3 Per-Band F1

| Detector | Short < 400 px | Medium 400–999 px | Long ≥ 1,000 px |
|----------|---------------:|------------------:|----------------:|
| Unified Confidence Score | 0.0% | 3.6% | **49.0%** |
| DINOv2 ViT-B | 0.0% | 3.1% | 16.9% |
| OpenCV | 0.0% | 0.0% | 1.8% |
| ASTRiDE | — | — | — |

The test set contains 284 long streaks (≥ 1,000 px) out of 308 total — 92% of the
ground truth. Short and medium F1 values of 0% reflect test-set composition rather
than detector capability: there are too few short/medium examples to provide a
reliable F1 estimate. ARGUS has not been benchmarked on a short-streak-dominated
test set.

### 7.4 Radon Angle Accuracy

Mean angle error: **0.018°** across the 308-image test set.

At 1,000 px streak length, 0.018° angular error corresponds to 0.3 px endpoint
displacement — effectively exact for any downstream analysis. At 100 px streak
length, this corresponds to 0.03 px displacement.

### 7.5 Ensemble Grouping Compression

| Metric | Before grouping | After grouping | Change |
|--------|----------------:|---------------:|-------:|
| Total predictions | 3,192 | 742 | −76.8% |
| Precision | 9.3% | 29.9% | +20.6 pp |
| Recall | 89.3% | 72.1% | −17.2 pp |
| F1 | 16.8% | 42.3% | +25.5 pp |
| Long-streak F1 | 16.9% | 49.0% | +32.1 pp |

---

## 8. Comparison with Prior Work

### 8.1 Summary Table

| System | Backbone | Training Data | Test Data | P | R | F1 | IoU | Notes |
|--------|----------|--------------|-----------|---|---|----|-----|-------|
| **ASTRiDE** (Kim et al. 2017) | Classical (boundary-tracing + Radon) | None | Not reported | Not reported | Not reported | Not reported | Not specified | No published P/R benchmark |
| **StreakMind** (arXiv 2605.03429) | YOLO11 OBB | 2,335 FITS (765 streaks + 1,523 bg + 280 synthetic) | 110 real FITS streaks | 94% | 97% | 95.4% | 0.8 | Mean streak length 203.5 px; raw FITS domain |
| **DINO-DETR** (Zhang et al. 2022) | ResNet-50 / Swin-L | COCO 2017 | COCO 2017 | AP-based | AP-based | — | COCO AP | 49.4 AP (R50), 63.3 AP test-dev (SwinL); used as ARGUS detection head |
| **Co-DINO** (Zong et al. 2022) | ResNet-50 / ViT-L | COCO 2017 | COCO 2017 | AP-based | AP-based | — | COCO AP | 51.2 AP (R50), 66.0 AP (ViT-L); initialised ARGUS archived Swin-T path |
| **DINOv2** (Oquab et al. 2024) | ViT-g | LVD-142M | ImageNet | — | — | — | — | 86.5% ImageNet linear (ViT-g); 84.5% (ViT-B); used as ARGUS frozen backbone |
| **ARGUS DINOv2 ViT-B** | DINOv2 ViT-B/16 (frozen) | SatStreaks + GTImages (~2,460 images) | SatStreaks test (308 JPEG) | 9.3% | 89.3% | 16.8% | 0.5 | mAP@0.5 = 75.5%; high FP rate |
| **ARGUS YOLO11n-OBB full** | YOLO11 nano OBB | SatStreaks (3,023 tiled 640 px) | YOLO tiled val† | 57.2% | 84.6% | 68.2% | 0.5 | †Tiled protocol; not comparable to full-image eval |
| **ARGUS Ensemble (UCS)** | 5-detector ensemble | — | SatStreaks test (308 JPEG) | 29.9% | 72.1% | 42.3% | 0.5 | F1 = 49% on long streaks; 742 grouped predictions |
| **Co-DINO Swin-T** (archived) | Swin-T | Full merged | SatStreaks test | — | — | — | 0.5 | mAP@0.5 = 0.19; baseline before DINOv2 integration |

### 8.2 Methodology Comparisons

**ARGUS vs. ASTRiDE (Kim et al. 2017)**

ARGUS integrates ASTRiDE as its Phase 0 classical detector, adopting the same
boundary-tracing + shape_factor filtering algorithm from the reference
implementation (`github.com/dwkim78/ASTRiDE`). The ARGUS integration adds: SEP
background subtraction upstream of ASTRiDE's contour detection; ensemble weighting
via F-0.5 score; and a confidence ceiling (0.6) to suppress ASTRiDE's tendency
to report high raw confidence on false positives. Kim et al. do not publish a
precision/recall benchmark on a standardised dataset, so numerical comparison is
not possible. ASTRiDE's primary value in ARGUS is on raw FITS images where its
sigma-threshold approach can detect faint streaks that escape ML detectors trained
on higher-contrast JPEG data.

**ARGUS vs. StreakMind (arXiv 2605.03429)**

Both systems use YOLO11 OBB as a detection component. The architectures diverge
significantly beyond that:

- StreakMind adds inter-frame association across multiple exposures, rejecting
  single-frame spurious detections. This is the key precision lever that ARGUS
  currently lacks.
- ARGUS uses a 5-detector ensemble with IoMin grouping to raise precision from
  single-frame evidence alone.
- StreakMind's training set is 2,335 raw FITS images; ARGUS's YOLO training uses
  3,023 JPEG/PNG source images.
- StreakMind evaluates at IoU = 0.8 on 110 raw FITS streaks (mean length 203.5 px);
  ARGUS evaluates at IoU = 0.5 on 308 JPEG streaks (92% ≥ 1,000 px).

StreakMind reports P = 94%, R = 97%; ARGUS reports P = 29.9%, R = 72.1%.
**These numbers cannot be directly compared** (see §2.4). A fair comparison would
require evaluating both systems on the same test set, with the same IoU threshold,
in the same image domain.

> **Correction to README:** The original README described StreakMind as
> "DINO-DETR-based." This is incorrect. StreakMind (arXiv 2605.03429) uses
> YOLO11 OBB as its detection backbone, not DINO-DETR.

**ARGUS vs. DINO-DETR / Co-DINO / DINOv2**

ARGUS uses DINO-DETR (Zhang et al. 2022) as its detection head and DINOv2 (Oquab
et al. 2024) as its frozen feature backbone. Co-DINO (Zong et al. 2022) pretrained
weights initialised the now-archived Swin-T path; the active DINOv2 path uses
DINO-DETR directly without Co-DINO's auxiliary heads.

The DINOv2 backbone's self-supervised features transfer to streak detection despite
pretraining exclusively on natural images. The Phase A feasibility probe (cosine
dissimilarity between streak and background DINOv2 features = 0.095 > 0.05 gate)
confirmed semantic separability before any task-specific training. With the backbone
entirely frozen, only the ~44 M parameters in the PatchToPyramid adapter and
DINO-DETR head learn from streak data, yielding mAP@0.5 = 0.74 at Phase C² (4
epochs, full dataset) — +0.55 above the Co-DINO Swin-T baseline (mAP@0.5 = 0.19).

### 8.3 Comparability Assessment: ARGUS vs. StreakMind

| Factor | ARGUS | StreakMind | Comparable? |
|--------|-------|------------|-------------|
| Image domain | JPEG exports (SatStreaks) | Raw FITS | No — different pixel distributions |
| IoU threshold | 0.5 | 0.8 | No — higher IoU threshold raises both P and R for well-localised detections |
| Test set size | 308 images / 308 streaks | 110 real streaks | No — different statistical power |
| Streak length dist. | 92% ≥ 1,000 px | Mean 203.5 px | No — ARGUS dominated by long streaks; StreakMind includes many short |
| Multi-frame association | No | Yes | Design difference — StreakMind can reject single-frame FPs |
| Evaluated detector | 5-detector ensemble | YOLO11 OBB | Different design |
| Training domain | JPEG/PNG | Raw FITS | No — different distribution for background and noise |

**Conclusion:** Direct numerical comparison of ARGUS and StreakMind P/R figures
should not be made without domain-matched evaluation on a shared test set.

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

**Precision:** The ensemble precision of 29.9% (UCS) reflects DINOv2's high false-
positive rate on elongated non-streak structures (clouds, galaxy filaments). The
grouping and UCS reduce but do not eliminate single-detector false positives that
have no multi-detector corroboration.

**No multi-frame association:** Single-frame detections cannot be rejected by
requiring cross-frame confirmation. StreakMind's inter-frame association addresses
this directly. For ARGUS, a genuine satellite streak appears in consecutive frames
with predictable angular displacement; exploiting this would substantially improve
precision without sacrificing recall.

**Short-streak performance:** F1 = 0% for short (<400 px) and medium (400–999 px)
streaks on this test set. The test set's 92% long-streak composition prevents
assessment of short-streak capability. Dedicated evaluation on a balanced dataset
is required.

**Classical detectors on JPEG:** OpenCV (1% recall) and ASTRiDE (not evaluated)
are designed for raw FITS pixel distributions. Their contributions are understated
by the current JPEG benchmark. Evaluation on GTImages raw FITS would give a more
representative picture.

**YOLO evaluation protocol:** The YOLO tiled validation split is not comparable to
the COCO full-image protocol. No fair head-to-head between YOLO and DINOv2 exists
in the current benchmark.

**DINOv2 ViT-L pending (Phase D):** All results in this document use the ViT-B
backbone. The larger ViT-L (300 M parameters, RTX 5070 Ti) may yield substantially
different precision/recall behaviour.

**UCS not calibrated:** The UCS formula is not calibrated to produce true
probabilities. Platt scaling or isotonic regression on held-out data would improve
score interpretation.

**TLE catalog coverage:** Cross-identification leaves objects unidentified when the
local catalog lacks TLE coverage for the observation time window.

### 9.2 Future Work

- **Phase D:** DINOv2 ViT-L/16 (300 M parameters, RTX 5070 Ti), 50 epochs frozen,
  target mAP@0.5 ≥ 0.74. Update `DETECTOR_PROFILES` with measured P/R after
  evaluation.
- **Multi-frame association:** Implement inter-frame linking to reject single-frame
  detections without cross-frame confirmation (StreakMind Phase 8 equivalent).
- **Raw FITS evaluation:** Run the full 5-detector ensemble on GTImages raw FITS
  to measure ASTRiDE and OpenCV contributions in their correct domain.
- **YOLO full-image re-evaluation:** Re-evaluate YOLO11n-OBB on the shared COCO
  full-image test set to enable fair head-to-head comparison with DINOv2.
- **UCS calibration:** Apply Platt scaling or isotonic regression to the UCS score
  distribution on a held-out calibration split.
- **Short-streak benchmark:** Evaluate on a balanced test set with short-streak
  representation.

---

## 10. Reproducibility Checklist

The following steps reproduce the headline benchmark results recorded in
`results/multi_method_benchmark.json` (2026-05-16).

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Clone the SatStreaks dataset:
   ```bash
   git clone https://github.com/jijup/SatStreaks data/satstreaks
   ```

3. Reproduce the exact test split (seed 42, 20% validation):
   ```bash
   python scripts/merge_annotations.py --seed 42 --val-fraction 0.2
   ```

4. Obtain the DINOv2 ViT-B checkpoint:
   `weights/dinov3_vitb_augmented/best_coco_bbox_mAP_epoch_10.pth` (~330 MB).
   This checkpoint is not distributed with the repository; it must be trained
   locally or obtained from the ARGUS authors.

5. Obtain the YOLO11n-OBB full-dataset checkpoint:
   `weights/run_full_yolo_obb/run/weights/best.pt` (~5.4 MB).
   This can be reproduced by running:
   ```bash
   bash scripts/train_yolo_full.sh
   ```
   (~9 hours on Apple M3 CPU, ~30 minutes on GPU).

6. Run the benchmark evaluation:
   ```bash
   MODEL_WEIGHTS=weights/dinov3_vitb_augmented/best_coco_bbox_mAP_epoch_10.pth \
   MODEL_SIZE=dinov3_vitb USE_DEV_SUBSET=false \
   python -m eval.benchmark \
       --run-pipeline \
       --annotations data/annotations/test.json \
       --output results/repro_benchmark.json
   ```

7. Verify results match `results/multi_method_benchmark.json`.

**What is and is not reproducible without re-training:**
- DINOv2 ViT-B, OpenCV, and UCS results: reproducible by running step 6 with the
  provided checkpoint.
- YOLO tiled val metrics: require re-running `yolo val` on the YOLO tiled split
  (not the COCO eval above); reproducible by re-training with `scripts/train_yolo_full.sh`.
- ASTRiDE results: require raw FITS input; not reproducible on the SatStreaks JPEG
  test set.

---

## 11. References

Kim, D.-W., Trippe, S., & Byun, H. (2017). ASTRiDE: Automated Streak Detection
for Astronomical Images. *The Astronomical Journal*, 153(6), 235.
https://github.com/dwkim78/ASTRiDE | https://ascl.net/1605.009

Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L. M., & Shum, H.-Y.
(2022). DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object
Detection. arXiv:2203.03605. *ICCV 2023*.
https://arxiv.org/abs/2203.03605

Zong, Z., Song, G., & Liu, Y. (2022). DETRs with Collaborative Hybrid Assignments
Training. arXiv:2211.12860.
https://arxiv.org/abs/2211.12860

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V.,
Fernandez, P., Haziza, F., Massa, F., El-Nouby, A., Assran, M., Ballas, N.,
Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V.,
... Bojanowski, P. (2024). DINOv2: Learning Robust Visual Features without
Supervision. arXiv:2304.07193. *Transactions on Machine Learning Research*.
https://arxiv.org/abs/2304.07193

StreakMind Collaboration (2026). [Title from arXiv:2605.03429]. *Astronomy &
Astrophysics*. arXiv:2605.03429.
https://arxiv.org/abs/2605.03429

SatStreaks Dataset (2024). Towards Supervised Learning for Delineating Satellite
Streaks from Astronomical Images. *Computer and Robot Vision (CRV 2024)*.
https://github.com/jijup/SatStreaks
