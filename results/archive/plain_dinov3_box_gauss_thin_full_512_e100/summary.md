# Plain DINOv3 Box Thin Full Run at 512px

This run tested whether increasing the plain PyTorch DINOv3 center/box input
resolution from `384` to `512` could close the strict-localization gap.

## Setup

```text
Train annotations: /Users/robert/Argus/data/annotations/train.json
Val annotations:   /Users/robert/Argus/data/annotations/val.json
Test annotations:  /Users/robert/Argus/data/annotations/test.json
Backbone weights:  /Users/robert/Argus/weights/dinov3_vitb16_lvd1689m.pth
Image size:        512
Epochs:            100
Best checkpoint:   weights/run_plain_dinov3_box_gauss_thin_full_512_e100/best.pt
```

## Test Metrics

| Threshold | NMS | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-----|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 7 | 313 | 0.537 | 0.545 | 0.541 | 0.342 | 0.212 | 0.361 deg |
| 0.20 | 7 | 309 | 0.537 | 0.539 | 0.538 | 0.338 | 0.211 | 0.357 deg |
| 0.20 | 3 | 321 | 0.520 | 0.542 | 0.531 | 0.339 | 0.210 | 0.356 deg |
| 0.30 | 7 | 307 | 0.541 | 0.539 | 0.540 | 0.338 | 0.211 | 0.357 deg |
| 0.50 | 7 | 301 | 0.545 | 0.532 | 0.539 | 0.335 | 0.208 | 0.353 deg |
| 0.50 | 3 | 307 | 0.538 | 0.536 | 0.537 | 0.336 | 0.207 | 0.353 deg |
| 0.70 | 7 | 294 | 0.548 | 0.523 | 0.535 | 0.330 | 0.206 | 0.354 deg |
| 0.80 | 7 | 288 | 0.549 | 0.513 | 0.530 | 0.324 | 0.202 | 0.356 deg |
| 0.90 | 7 | 278 | 0.558 | 0.503 | 0.529 | 0.319 | 0.200 | 0.361 deg |

## Assessment

Higher resolution did not close the gap. It slightly improved the best
`mAP@0.75` from `0.200` to `0.212`, and diagnostics showed median center error
improving from about `53 px` to `44 px`, but F1 and `mAP@0.50` regressed.

Conclusion: resolution alone is not the missing piece for replacing the
OpenMMLab detector.
