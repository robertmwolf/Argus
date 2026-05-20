# StreakMindYOLO GTImages Comparison

- Epochs: `15`
- Image size: `1280`
- Batch: `2`
- Training domain: raw FITS (GTImages) — methodology-matched to StreakMind
- Evaluation: held-out GTImages test split

## IoU = 0.5

| Track | Precision | Recall | F1 | mAP@0.5 | Angle error | Predictions |
|---|---:|---:|---:|---:|---:|---:|
| real_only | 0.521 | 0.439 | 0.476 | 0.260 | 32.45 | 48 |
| paper_long | 0.259 | 0.246 | 0.252 | 0.065 | 25.86 | 54 |
| adapted | 0.216 | 0.193 | 0.204 | 0.042 | 49.09 | 51 |
| gtimages_plus_frigate | 0.086 | 0.088 | 0.087 | 0.008 | 89.81 | 58 |

## IoU = 0.8

| Track | Precision | Recall | F1 | mAP@0.5 | Angle error | Predictions |
|---|---:|---:|---:|---:|---:|---:|
| real_only | 0.000 | 0.000 | 0.000 | 0.260 | 0.00 | 48 |
| paper_long | 0.018 | 0.018 | 0.018 | 0.065 | 0.23 | 54 |
| adapted | 0.000 | 0.000 | 0.000 | 0.042 | 0.00 | 51 |
| gtimages_plus_frigate | 0.000 | 0.000 | 0.000 | 0.008 | 0.00 | 58 |
| **StreakMind** *(reference)* | **0.940** | **0.970** | **0.955** | — | — | — |

## Notes

- StreakMind reference figures (P=94%, R=97%) are from arXiv:2605.03429, evaluated on
  La Sagra Observatory FITS frames with IoU=0.8. Direct comparison is approximate:
  different site, different epoch, different sky background diversity.
- GTImages is single-night single-site; the combined track adds Frigate FITS frames
  for additional background diversity.
