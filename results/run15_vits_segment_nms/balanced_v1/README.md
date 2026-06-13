# Run 15 (ViT-S) — val_balanced_v1 eval

**Date:** 2026-06-12
**Eval set:** `data/annotations/val_balanced_v1.json` — 59 short / 100 medium / 80 long, Frigate excluded.
**Pipeline:** fixed stitch (transitivity guard, angle 10°, gap 200, conf_floor 0.5) + peak/top-K filter, scored with architecture-aligned bands (short[50,400)/medium[400,1000)/long[1000+)).
**Method:** heatmap-cache zero-GPU threshold sweep.

## Results

| Setting | Prec | Recall | F1 | short-R | med-R | long-R | Npred |
|---------|------|--------|-----|---------|-------|--------|-------|
| pf0 @0.5  | 0.227 | 0.900 | 0.362 | 0.610 | 0.800 | 0.925 | 1612 |
| pf0 @0.6  | 0.304 | 0.916 | 0.457 | 0.661 | 0.830 | 0.938 | 1174 |
| pf0 @0.7  | 0.408 | 0.929 | 0.567 | 0.746 | 0.850 | 0.925 |  873 |
| pf0 @0.8  | 0.539 | 0.920 | 0.680 | 0.678 | 0.870 | 0.925 |  617 |
| pf85 @0.5 | 0.415 | 0.912 | 0.571 | 0.610 | 0.820 | 0.938 |  587 |
| pf85 @0.6 | 0.417 | 0.920 | 0.574 | 0.661 | 0.830 | 0.950 |  614 |
| **pf85 @0.7** | **0.464** | **0.933** | **0.619** | **0.746** | 0.850 | 0.938 | 617 |
| pf85 @0.8 | 0.540 | 0.920 | 0.681 | 0.678 | 0.870 | 0.925 |  580 |

`pf85` = `--peak-floor 0.85`.

## Headline

The model was always good — the old eval was broken. Versus the pre-fix numbers
(F1 ≈ 0.16, recall ≈ 0.13, medium/short recall 0%), the corrected pipeline shows:

- **Overall recall 0.90–0.93** across the whole length range.
- **Every band works:** short 0.61–0.75, medium 0.80–0.87, long 0.92–0.95.
  The "0% medium/short recall" was entirely a banding + stitch + sweep artifact.
- **Best F1 = 0.68** at threshold 0.8; **best recall = 0.933** at 0.7 + peak-floor 0.85.

## Peak-floor effect

The peak floor matters most at **low thresholds**: at t=0.5 it lifts precision
0.227 → 0.415 (1612 → 587 predictions) with no recall loss. At t≥0.8 the
threshold already removes the noise, so pf0 and pf85 converge. Net: peak-floor
0.85 lets you operate at a lower threshold (higher short-band recall) without
paying the precision cost.

## Recommended operating point

**t=0.7, peak-floor 0.85:** recall 0.933, precision 0.464, F1 0.619, with the
best short-band recall (0.746) — short is the clean single-tile measure of model
capability. Push to t=0.8 if precision matters more than short-band recall.

Precision (0.46–0.54) remains the weaker axis — residual cross-tile fragments
and faint FPs. Next precision levers: tune top-K, or raise peak-floor further.
