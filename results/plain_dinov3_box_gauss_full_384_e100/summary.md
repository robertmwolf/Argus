# Plain DINOv3 Gaussian Center/Box Full-Dataset Run

This run tested whether the OpenMMLab-free DINOv3 detector is ready to replace
the MMDetection-backed DINOv3 path.

## Run

```text
Train cache: /private/tmp/argus_dinov3_box_gauss_cache_train_full_384
Val cache:   /private/tmp/argus_dinov3_box_gauss_cache_val_full_384
Checkpoint:  weights/run_plain_dinov3_box_gauss_full_384_e100/best.pt
Epochs:      100
Best val score: 0.506
```

Data:

```text
Train: 3,023 images / 2,957 annotations
Val:     411 images /   386 annotations
Test:    308 images /   308 annotations
```

## Test Metrics

Evaluation used `/Users/robert/Argus/data/annotations/test.json`.

| Threshold | NMS | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-----|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 7 | 319 | 0.511 | 0.529 | 0.520 | 0.325 | 0.149 | 0.318 deg |
| 0.20 | 7 | 314 | 0.513 | 0.523 | 0.518 | 0.322 | 0.149 | 0.317 deg |
| 0.30 | 7 | 308 | 0.520 | 0.520 | 0.520 | 0.320 | 0.149 | 0.318 deg |
| 0.50 | 7 | 296 | 0.530 | 0.510 | 0.520 | 0.315 | 0.146 | 0.320 deg |
| 0.50 | 3 | 308 | 0.513 | 0.513 | 0.513 | 0.314 | 0.147 | 0.318 deg |
| 0.70 | 7 | 289 | 0.533 | 0.500 | 0.516 | 0.310 | 0.146 | 0.316 deg |
| 0.80 | 7 | 286 | 0.535 | 0.497 | 0.515 | 0.308 | 0.145 | 0.315 deg |
| 0.80 | 3 | 292 | 0.524 | 0.497 | 0.510 | 0.306 | 0.144 | 0.315 deg |
| 0.90 | 7 | 281 | 0.537 | 0.490 | 0.513 | 0.304 | 0.145 | 0.318 deg |

## Comparison

Current OpenMMLab-backed reference results on the same test split:

```text
MMDetection DINOv3 ViT-B COCO eval:
  mAP@0.50 = 0.740
  mAP@0.75 = 0.606

Multi-method DINOv3 ViT-B detector:
  precision = 0.093
  recall    = 0.893
  F1        = 0.168
  mAP@0.50  = 0.755
  mAP@0.75  = 0.594

Unified Confidence Score:
  precision = 0.299
  recall    = 0.721
  F1        = 0.423
  mAP@0.50  = 0.406
  mAP@0.75  = 0.318
```

## Determination

Do not remove OpenMMLab yet.

The plain PyTorch DINOv3 path is now a serious Windows-friendly training track:
it trains end to end without MMDetection/MMCV/MMEngine, produces balanced
precision and recall around `0.52` F1, and has excellent angle accuracy around
`0.32 deg`.

However, it is still far behind the OpenMMLab DINOv3 ViT-B detector on
localization metrics:

- mAP@0.50: `0.325` plain vs `0.740-0.755` OpenMMLab DINOv3
- mAP@0.75: `0.149` plain vs `0.594-0.606` OpenMMLab DINOv3

The dependency-free model should be kept and developed as the simpler
cross-platform path, but OpenMMLab should remain the reference/production
detector until the plain model closes the localization gap.
