# StreakMindYOLO GTImages Comparison

- Epochs: `15`
- Image size: `640`
- Batch: `8`
- Training domain: raw FITS (GTImages) — methodology-matched to StreakMind
- Evaluation: held-out GTImages test split

## IoU = 0.5

| Track | Precision | Recall | F1 | mAP@0.5 | Angle error | Predictions |
|---|---:|---:|---:|---:|---:|---:|
| adapted | 0.041 | 0.070 | 0.052 | 0.005 | 67.05 | 98 |

## IoU = 0.8

| Track | Precision | Recall | F1 | mAP@0.5 | Angle error | Predictions |
|---|---:|---:|---:|---:|---:|---:|
| adapted | 0.000 | 0.000 | 0.000 | 0.005 | 0.00 | 98 |
| **StreakMind** *(reference)* | **0.940** | **0.970** | **0.955** | — | — | — |

## Notes

- StreakMind reference figures (P=94%, R=97%) are from arXiv:2605.03429, evaluated on
  La Sagra Observatory FITS frames with IoU=0.8. Direct comparison is approximate:
  different site, different epoch, different sky background diversity.
- GTImages is single-night single-site; the combined track adds Frigate FITS frames
  for additional background diversity.
