# DarkMatters Dataset Evaluation Report

_Generated: 2026-05-18 04:23 UTC_
_YOLO weights: `weights/run_full_yolo_obb/run/weights/best.pt`_

## 1. Label Distribution

| Label | Count |
|---|---|
| good_negative | 588 |
| positive | 283 |
| hard_negative | 267 |
| hard_positive | 2 |
| hard_positive_dup1 | 2 |
| hard_positive_dup2 | 2 |
| **Total** | **1144** |

- **Streak positives (positive + hard_positive):** 285
- **Negatives (good + hard):** 855

## 2. Image Compatibility

### Positive images (streak present)
- Sampled: 30 / Missing: 0
- Unique sizes: ['(3000, 2001)']
- Color modes: ['L']
- Mean brightness: 0.2543 ± 0.0046

### Negative images (no streak)
- Sampled: 30 / Missing: 0
- Unique sizes: ['(3000, 2001)']
- Color modes: ['L']
- Mean brightness: 0.2564 ± 0.0082

## 3. YOLO OBB Probe (Option B feasibility)

| Metric | Value |
|---|---|
| Images probed | 283 |
| Images missing | 0 |
| With any detection (conf ≥ 0.25) | 2 (0.7%) |
| With high-conf detection (conf ≥ 0.5) | 0 (0.0%) |
| Mean detection confidence | 0.2807 |
| Mean streak angle | -0.02° |
| Inference time | 75.2 s |

**Verdict:** **NOT VIABLE via YOLO** — detector fires on <15% of positive images; instrument/resolution mismatch is likely. Manual annotation or discard.
**Estimated high-quality pseudo-annotations:** 0

## 4. Negative Pool (Option A)

- COCO JSON negatives built: **855** images (zero annotations each)
- ARGUS already has: ~91 GTImages negatives
- Net gain: +855 negatives (9.4× current pool)
- Caveat: JPEG previews at 3000×2001 px vs FITS originals; instrument response differs.

## 5. Instrument / Seeing Characterization

### Positive images
- Metadata matched: 281 / 285
- FWHM: 2.242 ± 0.514 (arcsec/px proxy)
- Eccentricity: 0.403
- Filter distribution: {'OIII': 86, 'SII': 45, 'HA': 33, 'G': 32, 'B': 31, 'L': 27, 'R': 27, '': 2}
- Top objects: ['NGC2239', 'NGC1055', 'B72', 'IC5148', 'NGC1360']

### Negative images
- Metadata matched: 786 / 855
- FWHM: 2.373 ± 0.55
- Eccentricity: 0.433

## 6. Recommendation

### Option A — Add negatives to training pool
**Proceed.** 855 JPEG negatives are ready as `results/darkmatters_eval/negatives.json`. Run `training/convert_labels.py` or merge directly into `data/annotations/` if the image dimensions match your tiling pipeline. Flag these with `source=darkmatters` in annotation attributes.

### Option B — Pseudo-annotate positives with YOLO
**Do not use YOLO pseudo-annotations.** Detection rate (1%) is too low. Options: (1) manually annotate positives in LabelImg/CVAT (~3 hr), or (2) discard positives and use only the negatives (Option A).

---
_Report produced by `scripts/evaluate_darkmatters_dataset.py`_