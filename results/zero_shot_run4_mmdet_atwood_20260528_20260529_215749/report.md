# Zero-Shot Evaluation Report

**Scope:** run4_mmdet_atwood_20260528  
**Label:** Run 4 MMDet — Atwood 2026-05-28 zero-shot holdout (175 imgs)  
**Checkpoint:** `weights/run4_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth`  
**Date:** 2026-05-29 22:27  
**Confidence threshold:** 0.30 | IoU threshold: 0.50  

## Decision

> INVESTIGATE — long recall 46.6% < 60%.  Significant domain shift detected.  Check: (1) pixel scale vs Atwood (1.27 arcsec/px); (2) FITS normalisation (apply_norm output range); (3) anchor box coverage for the new streak length distribution.  Fine-tuning alone may not be sufficient.

## Results vs Run 3 Baseline (standard test set)

| Metric | This scope (zero-shot) | Run 3 baseline | Delta |
|--------|------------------------|----------------|-------|
| Precision | 59.7% | 94.8% | -35.1pp |
| Recall | 44.9% | 83.8% | -38.9pp |
| F1 | 51.2% | 89.0% | -37.7pp |
| COCO mAP | 0.218 | 0.782 | -0.564 |
| COCO mAP@50 | 0.447 | 0.878 | -0.431 |

## Per-Band Recall

Bands: short < 269 px diagonal, medium 269–800 px, long > 800 px  
*(Recall is None when there are no GT annotations in that band.)*

| Band | This scope | Run 3 baseline | n (this scope) | Delta |
|------|------------|----------------|----------------|-------|
| Short | 0.0% | 100.0% | 2 | -100.0pp |
| Medium | 0.0% | 90.9% | 5 | -90.9pp |
| Long | 46.6% | 83.4% | 178 | -36.8pp |

## Detailed COCO Metrics

- mAP:     0.218
- mAP@50:  0.447
- mAP@75:  0.199
- mAP_s:   0.000
- mAP_m:   0.000
- mAP_l:   0.226

## Detailed P/R

- Precision: 59.7%
- Recall:    44.9%
- F1:        51.2%
- TP: 83  FP: 56  FN: 102
- GT annotations: 185  Predictions above conf: 139

## Next Steps

1. Investigate domain shift before training:
   - Pixel scale: this scope vs Atwood (1.27 arcsec/px baseline)
   - FITS normalisation: check `apply_norm()` output range on a sample
   - Streak length distribution: compare band histogram to training data
2. If pixel scale differs significantly, consider a new resolution tier
   in the training config (current: 400px tiles).
3. Only proceed to fine-tuning after understanding the cause of the gap.