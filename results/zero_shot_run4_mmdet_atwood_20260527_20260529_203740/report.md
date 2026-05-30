# Zero-Shot Evaluation Report

**Scope:** run4_mmdet_atwood_20260527  
**Label:** Run 4 MMDet — Atwood 2026-05-27 zero-shot holdout (507 imgs)  
**Checkpoint:** `weights/run4_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth`  
**Date:** 2026-05-29 21:57  
**Confidence threshold:** 0.30 | IoU threshold: 0.50  

## Decision

> INVESTIGATE — long recall 57.0% < 60%.  Significant domain shift detected.  Check: (1) pixel scale vs Atwood (1.27 arcsec/px); (2) FITS normalisation (apply_norm output range); (3) anchor box coverage for the new streak length distribution.  Fine-tuning alone may not be sufficient.

## Results vs Run 3 Baseline (standard test set)

| Metric | This scope (zero-shot) | Run 3 baseline | Delta |
|--------|------------------------|----------------|-------|
| Precision | 71.4% | 94.8% | -23.4pp |
| Recall | 51.9% | 83.8% | -31.9pp |
| F1 | 60.1% | 89.0% | -28.9pp |
| COCO mAP | 0.276 | 0.782 | -0.506 |
| COCO mAP@50 | 0.515 | 0.878 | -0.363 |

## Per-Band Recall

Bands: short < 269 px diagonal, medium 269–800 px, long > 800 px  
*(Recall is None when there are no GT annotations in that band.)*

| Band | This scope | Run 3 baseline | n (this scope) | Delta |
|------|------------|----------------|----------------|-------|
| Short | 0.0% | 100.0% | 9 | -100.0pp |
| Medium | 4.4% | 90.9% | 45 | -86.5pp |
| Long | 57.0% | 83.4% | 505 | -26.4pp |

## Detailed COCO Metrics

- mAP:     0.276
- mAP@50:  0.515
- mAP@75:  0.268
- mAP_s:   0.000
- mAP_m:   0.055
- mAP_l:   0.291

## Detailed P/R

- Precision: 71.4%
- Recall:    51.9%
- F1:        60.1%
- TP: 290  FP: 116  FN: 269
- GT annotations: 559  Predictions above conf: 406

## Next Steps

1. Investigate domain shift before training:
   - Pixel scale: this scope vs Atwood (1.27 arcsec/px baseline)
   - FITS normalisation: check `apply_norm()` output range on a sample
   - Streak length distribution: compare band histogram to training data
2. If pixel scale differs significantly, consider a new resolution tier
   in the training config (current: 400px tiles).
3. Only proceed to fine-tuning after understanding the cause of the gap.