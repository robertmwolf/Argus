# StreakMindYOLO GTImages Comparison

- Epochs: `15`
- Image size: `1280`
- Batch: `2`
- Training domain: raw FITS (GTImages) — methodology-matched to StreakMind
- Evaluation: held-out GTImages test split

## IoU = 0.5

| Track | Precision | Recall | F1 | mAP@0.5 | Angle error | Predictions |
|---|---:|---:|---:|---:|---:|---:|
| real_only | 0.536 | 0.526 | 0.531 | 0.308 | 39.00 | 56 |
| paper_long | 0.262 | 0.281 | 0.271 | 0.075 | 33.79 | 61 |
| adapted | 0.242 | 0.263 | 0.252 | 0.065 | 59.79 | 62 |
| gtimages_plus_frigate | 0.085 | 0.088 | 0.086 | 0.008 | 89.81 | 59 |

## IoU = 0.8

| Track | Precision | Recall | F1 | mAP@0.5 | Angle error | Predictions |
|---|---:|---:|---:|---:|---:|---:|
| real_only | 0.000 | 0.000 | 0.000 | 0.308 | 0.00 | 56 |
| paper_long | 0.016 | 0.018 | 0.017 | 0.075 | 0.23 | 61 |
| adapted | 0.000 | 0.000 | 0.000 | 0.065 | 0.00 | 62 |
| gtimages_plus_frigate | 0.000 | 0.000 | 0.000 | 0.008 | 0.00 | 59 |
| **StreakMind** *(reference)* | **0.940** | **0.970** | **0.955** | — | — | — |

## Notes

- StreakMind reference figures (P=94%, R=97%) are from arXiv:2605.03429, evaluated on
  La Sagra Observatory FITS frames with IoU=0.8. Direct comparison is approximate:
  different site, different epoch, different sky background diversity.
- GTImages is single-night single-site; the combined track adds Frigate FITS frames
  for additional background diversity.
