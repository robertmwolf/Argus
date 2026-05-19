# Plain DINOv3 Direct Center/Box Cached Run

This run tested the OpenMMLab-free DINOv3 spike with a cached frozen ViT-B
backbone and a direct center/box head. The head predicts:

- center heatmap logits
- center offsets `dx`, `dy`
- `cos(2 angle)` and `sin(2 angle)`
- normalized length
- normalized width

Unlike the connected-component heatmap runs, detections come directly from
local maxima in the center heatmap. This run preserves thin streak widths
instead of forcing every predicted width to at least one DINO patch.

## Run

```text
Train cache: /private/tmp/argus_dinov3_box_thin_cache_train512_384
Val cache:   /private/tmp/argus_dinov3_box_thin_cache_val256_384
Checkpoint:  weights/run_plain_dinov3_box_thin_cached_train512_384_e80/best.pt
Epochs:      80
Best val score: 0.288
```

## Test Metrics

Evaluation used `/Users/robert/Argus/data/annotations/test.json`.

| Threshold | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 444 | 0.331 | 0.477 | 0.391 | 0.238 | 0.054 | 1.013 deg |
| 0.20 | 420 | 0.350 | 0.477 | 0.404 | 0.238 | 0.054 | 1.013 deg |
| 0.30 | 399 | 0.366 | 0.474 | 0.413 | 0.237 | 0.054 | 1.001 deg |
| 0.50 | 362 | 0.384 | 0.451 | 0.415 | 0.228 | 0.054 | 0.975 deg |
| 0.70 | 327 | 0.410 | 0.435 | 0.422 | 0.222 | 0.054 | 0.964 deg |
| 0.75 | 319 | 0.414 | 0.429 | 0.421 | 0.219 | 0.053 | 0.966 deg |
| 0.80 | 310 | 0.419 | 0.422 | 0.421 | 0.216 | 0.053 | 0.970 deg |
| 0.90 | 274 | 0.445 | 0.396 | 0.419 | 0.205 | 0.049 | 0.970 deg |

## Conclusion

The direct center/box head is the strongest plain-DINOv3 spike so far. Compared
with the geometry-head connected-component run, it improves F1 from `0.329` to
`0.422` and improves best mAP@0.50 from `0.216` to `0.238`, while keeping angle
error around one degree.

Strict IoU quality remains weak. The next useful iteration is likely Gaussian
center supervision, more train samples, and better duplicate suppression or
matching around the center heatmap peaks.
