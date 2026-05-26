# Comprehensive Evaluation Report

**Model:** DINOv3 ViT-B GT+DM+SatStreaks (4 epochs frozen)  
**Checkpoint:** `weights/run_gt_dm_satstreaks_dinov3_vitb/best_coco_bbox_mAP_epoch_4.pth`  
**Date:** 2026-05-26 10:22  
**Confidence threshold (P/R/band):** 0.30 | IoU threshold: 0.50  

## Summary

| Set | mAP | mAP@50 | mAP@75 | Precision | Recall | F1 |
|-----|-----|--------|--------|-----------|--------|----|
| Standard test (satstreaks) | 0.600 | 0.755 | 0.644 | 71.2% | 72.4% | 71.8% |
| Frigate (zero-shot) | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% | 0.0% |
| BrentImages Night 2 (zero-shot) | 0.085 | 0.296 | 0.030 | 47.8% | 31.9% | 38.2% |
| DarkMatters holdout (zero-shot) | 0.564 | 0.720 | 0.606 | 71.2% | 69.1% | 70.1% |

## Per-Band Recall (conf ≥ 0.30, IoU ≥ 0.50)

Bands: short < 269 px diagonal, medium 269–800 px, long > 800 px

| Set | Short recall | Short n | Medium recall | Medium n | Long recall | Long n |
|-----|-------------|---------|--------------|----------|-------------|--------|
| Standard test (satstreaks) | 50.0% | 2 | 72.7% | 11 | 72.5% | 295 |
| Frigate (zero-shot) | 0.0% | 377 | — | 0 | — | 0 |
| BrentImages Night 2 (zero-shot) | 0.0% | 5 | 14.2% | 120 | 60.8% | 79 |
| DarkMatters holdout (zero-shot) | 50.0% | 2 | 64.3% | 14 | 69.4% | 317 |

## Detailed Results

### Standard test (satstreaks)

**COCO metrics:**
- mAP: 0.600
- mAP@50: 0.755
- mAP@75: 0.644
- mAP_s: -1.000
- mAP_m: 0.364
- mAP_l: 0.614

**P/R @ conf≥0.30:**
- Precision: 71.2%
- Recall: 72.4%
- F1: 71.8%
- TP: 223  FP: 90  FN: 85
- GT annotations: 308  Predictions above conf: 313

**Per-band recall:**
- Short: 50.0%  (TP=1, FN=1, n=2)
- Medium: 72.7%  (TP=8, FN=3, n=11)
- Long: 72.5%  (TP=214, FN=81, n=295)

---

### Frigate (zero-shot)

**COCO metrics:**
- mAP: 0.000
- mAP@50: 0.000
- mAP@75: 0.000
- mAP_s: 0.000
- mAP_m: -1.000
- mAP_l: -1.000

**P/R @ conf≥0.30:**
- Precision: 0.0%
- Recall: 0.0%
- F1: 0.0%
- TP: 0  FP: 16  FN: 377
- GT annotations: 377  Predictions above conf: 16

**Per-band recall:**
- Short: 0.0%  (TP=0, FN=377, n=377)
- Medium: —  (TP=0, FN=0, n=0)
- Long: —  (TP=0, FN=0, n=0)

---

### BrentImages Night 2 (zero-shot)

**COCO metrics:**
- mAP: 0.085
- mAP@50: 0.296
- mAP@75: 0.030
- mAP_s: -1.000
- mAP_m: 0.096
- mAP_l: 0.127

**P/R @ conf≥0.30:**
- Precision: 47.8%
- Recall: 31.9%
- F1: 38.2%
- TP: 65  FP: 71  FN: 139
- GT annotations: 204  Predictions above conf: 136

**Per-band recall:**
- Short: 0.0%  (TP=0, FN=5, n=5)
- Medium: 14.2%  (TP=17, FN=103, n=120)
- Long: 60.8%  (TP=48, FN=31, n=79)

---

### DarkMatters holdout (zero-shot)

**COCO metrics:**
- mAP: 0.564
- mAP@50: 0.720
- mAP@75: 0.606
- mAP_s: -1.000
- mAP_m: 0.242
- mAP_l: 0.581

**P/R @ conf≥0.30:**
- Precision: 71.2%
- Recall: 69.1%
- F1: 70.1%
- TP: 230  FP: 93  FN: 103
- GT annotations: 333  Predictions above conf: 323

**Per-band recall:**
- Short: 50.0%  (TP=1, FN=1, n=2)
- Medium: 64.3%  (TP=9, FN=5, n=14)
- Long: 69.4%  (TP=220, FN=97, n=317)

---
