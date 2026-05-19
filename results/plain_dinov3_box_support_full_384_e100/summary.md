# Plain DINOv3 Box Support-Offset Full Run

This run tested whether training box regression over the high-confidence
Gaussian center neighborhood would improve strict localization. The change also
expanded center offsets from `+-0.5` cells to `+-2.0` cells so neighboring grid
cells can regress back to the true streak center.

## Setup

```text
Train annotations: /Users/robert/Argus/data/annotations/train.json
Val annotations:   /Users/robert/Argus/data/annotations/val.json
Test annotations:  /Users/robert/Argus/data/annotations/test.json
Backbone weights:  /Users/robert/Argus/weights/dinov3_vitb16_lvd1689m.pth
Image size:        384
Epochs:            100
Best checkpoint:   weights/run_plain_dinov3_box_support_full_384_e100/best.pt
```

## Test Metrics

| Threshold | NMS | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-----|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 7 | 309 | 0.560 | 0.562 | 0.561 | 0.332 | 0.185 | 0.149 deg |
| 0.10 | 3 | 324 | 0.534 | 0.562 | 0.548 | 0.331 | 0.187 | 0.149 deg |
| 0.20 | 7 | 303 | 0.568 | 0.558 | 0.563 | 0.330 | 0.185 | 0.149 deg |
| 0.30 | 7 | 300 | 0.567 | 0.552 | 0.559 | 0.326 | 0.184 | 0.149 deg |
| 0.50 | 7 | 298 | 0.567 | 0.549 | 0.558 | 0.324 | 0.184 | 0.149 deg |
| 0.50 | 3 | 305 | 0.554 | 0.549 | 0.551 | 0.324 | 0.184 | 0.149 deg |
| 0.70 | 7 | 292 | 0.568 | 0.539 | 0.553 | 0.319 | 0.181 | 0.150 deg |
| 0.80 | 7 | 291 | 0.570 | 0.539 | 0.554 | 0.319 | 0.181 | 0.150 deg |
| 0.90 | 7 | 279 | 0.574 | 0.519 | 0.545 | 0.308 | 0.176 | 0.149 deg |

Best operating point by F1 is threshold `0.20`, NMS `7`.

## Diagnostics

Threshold `0.20`, NMS `7`:

```text
Predictions: 303
Ground truth: 308
Matches @ IoU 0.50: 172
Matches @ IoU 0.75: 128
Matched IoU: mean 0.804, p50 0.838, p75 0.890, p90 0.929
Center error: mean 73.1 px, p50 59.1 px, p75 88.6 px, p90 129.4 px
Length relative error: mean 0.048, p50 0.023, p75 0.060, p90 0.126
Width absolute error: mean 127.6 px, p50 83.6 px, p75 169.2 px, p90 284.0 px
Angle error: mean 0.149 deg, p50 0.121 deg, p75 0.207 deg, p90 0.282 deg
```

## Assessment

This formulation improved F1 from `0.553` to `0.563` and improved angle error,
but it regressed the important localization metrics: `mAP@0.50` fell from
`0.383` to `0.332`, and `mAP@0.75` fell from `0.200` to `0.187`.

Conclusion: support-offset regression is not the path to closing the
OpenMMLab gap in this architecture.
