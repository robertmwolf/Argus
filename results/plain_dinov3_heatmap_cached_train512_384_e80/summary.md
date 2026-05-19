# Plain DINOv3 Heatmap Cached Run — Train512 384px

Run date: 2026-05-17

## Setup

- Model: DINOv3 ViT-B frozen encoder, cached features, trainable heatmap head.
- Training cache: first 512 images from `data/annotations/train.json`.
- Validation cache: first 256 images from `data/annotations/val.json`.
- Image preprocessing: aspect-preserving 384 px letterbox.
- Training: 80 cached-head epochs, batch size 32.

Best validation Dice: 0.729 at epoch 80.

## Test Results

Held-out split: `data/annotations/test.json` (308 images, 308 annotations).
Evaluation used original-image geometry refinement.

| Threshold | Predictions | Precision | Recall | F1 | mAP@0.5 | Angle error |
|-----------|------------:|----------:|-------:|---:|--------:|------------:|
| 0.95 | 346 | 0.1503 | 0.1688 | 0.1590 | 0.0594 | 18.3 deg |
| 0.98 | 323 | 0.1641 | 0.1721 | 0.1680 | 0.0536 | 16.1 deg |
| 0.99 | 321 | 0.1526 | 0.1591 | 0.1558 | 0.0480 | 17.7 deg |

## Interpretation

Feature caching worked and made head training fast. The cached head achieved
much higher validation Dice than the first direct-training run, but that did
not transfer to better held-out OBB metrics. The likely issue is objective
alignment: coarse heatmap overlap improves while connected-component OBBs still
fragment or mis-size streak geometry.

Next iteration should train targets/outputs closer to ARGUS geometry:

- add line endpoint or angle regression,
- evaluate heatmap IoU separately from OBB IoU,
- tune connected-component merging and minimum area,
- use full train/val cache after the geometry target is improved.
