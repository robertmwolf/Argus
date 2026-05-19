# Plain DINOv3 Gaussian Center/Box Cached Run

This run tested the OpenMMLab-free DINOv3 direct detector with:

- Gaussian center targets instead of one-hot center cells
- separate center and box branches after a shared convolutional stem
- direct prediction of center offsets, angle vector, length, and width
- wider local-maximum NMS for center peaks

## Run

```text
Train cache: /private/tmp/argus_dinov3_box_gauss_cache_train512_384
Val cache:   /private/tmp/argus_dinov3_box_gauss_cache_val256_384
Checkpoint:  weights/run_plain_dinov3_box_gauss_cached_train512_384_e80/best.pt
Epochs:      80
Best val score: 0.383
```

## Test Metrics

Evaluation used `/Users/robert/Argus/data/annotations/test.json`.

| Threshold | NMS | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-----|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 3 | 374 | 0.385 | 0.468 | 0.422 | 0.266 | 0.053 | 0.510 deg |
| 0.20 | 3 | 347 | 0.415 | 0.468 | 0.440 | 0.266 | 0.053 | 0.510 deg |
| 0.30 | 3 | 340 | 0.424 | 0.468 | 0.444 | 0.266 | 0.053 | 0.510 deg |
| 0.50 | 3 | 322 | 0.441 | 0.461 | 0.451 | 0.263 | 0.053 | 0.507 deg |
| 0.50 | 5 | 305 | 0.466 | 0.461 | 0.463 | 0.268 | 0.054 | 0.507 deg |
| 0.50 | 7 | 297 | 0.478 | 0.461 | 0.469 | 0.269 | 0.054 | 0.507 deg |
| 0.70 | 3 | 297 | 0.461 | 0.445 | 0.453 | 0.255 | 0.052 | 0.489 deg |
| 0.75 | 3 | 293 | 0.464 | 0.442 | 0.453 | 0.254 | 0.052 | 0.486 deg |
| 0.80 | 3 | 283 | 0.474 | 0.435 | 0.454 | 0.251 | 0.052 | 0.487 deg |
| 0.80 | 5 | 271 | 0.495 | 0.435 | 0.463 | 0.256 | 0.053 | 0.487 deg |
| 0.80 | 7 | 267 | 0.502 | 0.435 | 0.466 | 0.256 | 0.053 | 0.487 deg |
| 0.90 | 3 | 258 | 0.496 | 0.416 | 0.452 | 0.241 | 0.049 | 0.493 deg |

## Conclusion

Gaussian center supervision plus the split head is the strongest plain-DINOv3
iteration so far. Best F1 improved from `0.422` to `0.469`, best mAP@0.50
improved from `0.238` to `0.269`, and angle error improved to about half a
degree.

The remaining weakness is strict localization. `mAP@0.75` stayed near `0.054`,
so the next bottleneck is probably center/length precision rather than angle.
