# Zero-Shot Evaluation Report

**Scope:** run4_mmdet_test_atwood  
**Label:** Run 4 MMDet — Atwood geo-stratified test (133 imgs)  
**Checkpoint:** `weights/run4_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth`  
**Date:** 2026-05-29 20:37  
**Confidence threshold:** 0.30 | IoU threshold: 0.50  

## Decision

> FINE-TUNE ADVISED — long recall 75.0% is between 60% and 80%.  Run fine-tune with streak_dinov3_vitb_400px_ft.py; include existing-domain images at ≥1:1 ratio in the training JSON to prevent regression.

## Results vs Run 3 Baseline (standard test set)

| Metric | This scope (zero-shot) | Run 3 baseline | Delta |
|--------|------------------------|----------------|-------|
| Precision | 61.1% | 94.8% | -33.7pp |
| Recall | 55.5% | 83.8% | -28.3pp |
| F1 | 58.1% | 89.0% | -30.8pp |
| COCO mAP | 0.223 | 0.782 | -0.559 |
| COCO mAP@50 | 0.518 | 0.878 | -0.360 |

## Per-Band Recall

Bands: short < 269 px diagonal, medium 269–800 px, long > 800 px  
*(Recall is None when there are no GT annotations in that band.)*

| Band | This scope | Run 3 baseline | n (this scope) | Delta |
|------|------------|----------------|----------------|-------|
| Short | 0.0% | 100.0% | 3 | -100.0pp |
| Medium | 48.8% | 90.9% | 80 | -42.2pp |
| Long | 75.0% | 83.4% | 36 | -8.4pp |

## Detailed COCO Metrics

- mAP:     0.223
- mAP@50:  0.518
- mAP@75:  0.139
- mAP_s:   -1.000
- mAP_m:   0.284
- mAP_l:   0.297

## Detailed P/R

- Precision: 61.1%
- Recall:    55.5%
- F1:        58.1%
- TP: 66  FP: 42  FN: 53
- GT annotations: 119  Predictions above conf: 108

## Next Steps

1. Annotate Night 2 from this scope (target ≥200 images).
2. Add to manifest with `split: train` and appropriate `mix_weight`.
3. Build fine-tune JSON:
   ```
   python scripts/build_training_json.py \
       --mix-ratio {scope}:<weight> \
      --output data/annotations/all_train_ft_run4_mmdet_test_atwood.json
   ```
4. Run fine-tune:
   ```
   TRAIN_ANN_FILE=annotations/all_train_ft_{scope}.json \
   python -m training.train_dino \
       --config models/dino/streak_dinov3_vitb_400px_ft.py \
      --work-dir weights/run_ft_run4_mmdet_test_atwood
   ```
5. Re-evaluate on BOTH this scope (Night 1) AND the standard test set.
   Accept only if standard-test recall does not drop > 2pp.