# Plain DINOv3 Box Gaussian Thin Full Run

This full-dataset run keeps the OpenMMLab-free DINOv3 center/box architecture
but removes the letterbox-space width floor and increases the width regression
loss weight.

## Setup

```text
Train annotations: /Users/robert/Argus/data/annotations/train.json
Val annotations:   /Users/robert/Argus/data/annotations/val.json
Test annotations:  /Users/robert/Argus/data/annotations/test.json
Backbone weights:  /Users/robert/Argus/weights/dinov3_vitb16_lvd1689m.pth
Image size:        384
Epochs:            100
Best checkpoint:   weights/run_plain_dinov3_box_gauss_thin_full_384_e100/best.pt
```

## Test Metrics

| Threshold | NMS | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-----|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 7 | 312 | 0.545 | 0.552 | 0.548 | 0.383 | 0.200 | 0.236 deg |
| 0.20 | 7 | 307 | 0.554 | 0.552 | 0.553 | 0.383 | 0.200 | 0.236 deg |
| 0.30 | 7 | 304 | 0.553 | 0.545 | 0.549 | 0.379 | 0.198 | 0.237 deg |
| 0.50 | 7 | 299 | 0.559 | 0.542 | 0.550 | 0.378 | 0.198 | 0.232 deg |
| 0.50 | 3 | 308 | 0.542 | 0.542 | 0.542 | 0.376 | 0.197 | 0.232 deg |
| 0.70 | 7 | 289 | 0.564 | 0.529 | 0.546 | 0.370 | 0.196 | 0.229 deg |
| 0.80 | 7 | 282 | 0.571 | 0.523 | 0.546 | 0.367 | 0.194 | 0.228 deg |
| 0.80 | 3 | 287 | 0.561 | 0.523 | 0.541 | 0.365 | 0.193 | 0.228 deg |
| 0.90 | 7 | 276 | 0.565 | 0.506 | 0.534 | 0.357 | 0.192 | 0.226 deg |

Best operating point by F1 is threshold `0.20`, NMS `7`.

## Diagnostics

Threshold `0.20`, NMS `7`:

```text
Predictions: 307
Ground truth: 308
Matches @ IoU 0.50: 170
Matches @ IoU 0.75: 121
Matched IoU: mean 0.800, p50 0.834, p75 0.887, p90 0.939
Center error: mean 75.1 px, p50 53.1 px, p75 100.8 px, p90 167.2 px
Length relative error: mean 0.047, p50 0.025, p75 0.069, p90 0.113
Width absolute error: mean 128.1 px, p50 81.5 px, p75 168.6 px, p90 281.3 px
Angle error: mean 0.236 deg, p50 0.172 deg, p75 0.318 deg, p90 0.484 deg
```

## Assessment

This is the strongest plain PyTorch DINOv3 result so far. It improves the
previous full Gaussian run from `F1=0.520`, `mAP@0.50=0.325`, `mAP@0.75=0.149`
to `F1=0.553`, `mAP@0.50=0.383`, `mAP@0.75=0.200`.

The result strengthens the case for continuing the OpenMMLab-free path, but it
does not justify removing OpenMMLab yet. The plain path still trails the
OpenMMLab-backed DINOv3 ViT-B COCO-eval result (`mAP@0.50=0.740`,
`mAP@0.75=0.606`) and remains below the unified confidence score on strict
localization (`mAP@0.75=0.318`).
