# Comprehensive Evaluation Report

**Model:** DINOv3 ViT-B Multi-source (clean cold-start)  
**Checkpoint:** `/Users/robert/Argus/weights/run_clean_vitb_nodm/best_coco_bbox_mAP_epoch_15.pth`  
**Date:** 2026-05-29 20:58  
**Confidence threshold (P/R/band):** 0.30 | IoU threshold: 0.50  

## Summary

| Set | mAP | mAP@50 | mAP@75 | Precision | Recall | F1 |
|-----|-----|--------|--------|-----------|--------|----|
| Standard test (satstreaks) | 0.005 | 0.014 | 0.002 | 6.2% | 5.5% | 5.8% |

## Per-Band Recall (conf ≥ 0.30, IoU ≥ 0.50)

Bands: short < 269 px diagonal, medium 269–800 px, long > 800 px

| Set | Short recall | Short n | Medium recall | Medium n | Long recall | Long n |
|-----|-------------|---------|--------------|----------|-------------|--------|
| Standard test (satstreaks) | 50.0% | 2 | 54.5% | 11 | 3.4% | 295 |

## Detailed Results

### Standard test (satstreaks)

**COCO metrics:**
- mAP: 0.005
- mAP@50: 0.014
- mAP@75: 0.002
- mAP_s: -1.000
- mAP_m: 0.233
- mAP_l: 0.004

**P/R @ conf≥0.30:**
- Precision: 6.2%
- Recall: 5.5%
- F1: 5.8%
- TP: 17  FP: 258  FN: 291
- GT annotations: 308  Predictions above conf: 275

**Per-band recall:**
- Short: 50.0%  (TP=1, FN=1, n=2)
- Medium: 54.5%  (TP=6, FN=5, n=11)
- Long: 3.4%  (TP=10, FN=285, n=295)

---
