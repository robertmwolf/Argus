# Plain DINOv3 Geometry Cached Run

This run tested the OpenMMLab-free DINOv3 spike with a cached frozen ViT-B
backbone and a five-channel head:

- heatmap logits
- `cos(2 angle)`
- `sin(2 angle)`
- normalized length
- normalized width

## Run

```text
Train cache: /private/tmp/argus_dinov3_geom_cache_train512_384
Val cache:   /private/tmp/argus_dinov3_geom_cache_val256_384
Checkpoint:  weights/run_plain_dinov3_geometry_cached_train512_384_e80/best.pt
Epochs:      80
Best val Dice: 0.727
```

## Test Metrics

Evaluation used `data/annotations/test.json`.

| Threshold | Refinement | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|------------|-------------|-----------|--------|----|----------|----------|-------------|
| 0.90 | off | 362 | 0.279 | 0.328 | 0.302 | 0.221 | 0.032 | 0.232 deg |
| 0.95 | off | 363 | 0.281 | 0.331 | 0.304 | 0.215 | 0.033 | 0.240 deg |
| 0.98 | off | 346 | 0.298 | 0.334 | 0.315 | 0.216 | 0.030 | 0.254 deg |
| 0.99 | off | 331 | 0.317 | 0.341 | 0.329 | 0.216 | 0.030 | 0.265 deg |
| 0.99 | on | 331 | 0.079 | 0.084 | 0.081 | 0.012 | 0.001 | 7.529 deg |

## Conclusion

The geometry head solved the orientation failure from the previous cached
heatmap-only experiment: angle error is now under one degree without any Radon
refinement. Detection quality improved over the cached heatmap-only result, but
it is still below the earlier direct-training heatmap run and has weak mAP@0.75.

The next spike should move away from connected-component box extraction and
predict center/box parameters directly.
