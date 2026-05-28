# Comprehensive Evaluation Report

**Model:** DINOv3 ViT-B Multi-source (clean cold-start)  
**Checkpoint:** `/Users/robert/Argus/weights/run_clean_vitb_nodm/best_coco_bbox_mAP_epoch_15.pth`  
**Date:** 2026-05-28 16:01  
**Confidence threshold (P/R/band):** 0.30 | IoU threshold: 0.50  

## Summary

| Set | mAP | mAP@50 | mAP@75 | Precision | Recall | F1 |
|-----|-----|--------|--------|-----------|--------|----|
| Standard test (satstreaks) | 0.782 | 0.878 | 0.826 | 94.8% | 83.8% | 89.0% |

## Per-Band Recall (conf ≥ 0.30, IoU ≥ 0.50)

Bands: short < 269 px diagonal, medium 269–800 px, long > 800 px

| Set | Short recall | Short n | Medium recall | Medium n | Long recall | Long n |
|-----|-------------|---------|--------------|----------|-------------|--------|
| Standard test (satstreaks) | 100.0% | 2 | 90.9% | 11 | 83.4% | 295 |

## Detailed Results

### Standard test (satstreaks)

**COCO metrics:**
- mAP: 0.782
- mAP@50: 0.878
- mAP@75: 0.826
- mAP_s: -1.000
- mAP_m: 0.703
- mAP_l: 0.784

**P/R @ conf≥0.30:**
- Precision: 94.8%
- Recall: 83.8%
- F1: 89.0%
- TP: 258  FP: 14  FN: 50
- GT annotations: 308  Predictions above conf: 272

**Per-band recall:**
- Short: 100.0%  (TP=2, FN=0, n=2)
- Medium: 90.9%  (TP=10, FN=1, n=11)
- Long: 83.4%  (TP=246, FN=49, n=295)

---
