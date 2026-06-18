# Plain DINOv3 Heatmap Spike — ViT-B 384px Subset Run

Run date: 2026-05-16

## Training

```bash
python -m training.train_dinov3_heatmap \
    --train-annotations data/annotations/train.json \
    --val-annotations data/annotations/val.json \
    --weights weights/dinov3_vitb16_lvd1689m.pth \
    --model-size base \
    --image-size 384 \
    --batch-size 2 \
    --epochs 8 \
    --lr 0.001 \
    --pos-weight 20 \
    --max-samples 512 \
    --work-dir weights/run_plain_dinov3_heatmap_vitb_384_subset512_e8
```

Best checkpoint:

```text
weights/run_plain_dinov3_heatmap_vitb_384_subset512_e8/best.pt
```

Best validation Dice: 0.334 at epoch 8.

## Test Threshold Sweep

Held-out split: `data/annotations/test.json` (308 images, 308 annotations).

| Threshold | Predictions | Precision | Recall | F1 | mAP@0.5 | Angle error |
|-----------|------------:|----------:|-------:|---:|--------:|------------:|
| 0.30 | 397 | 0.0856 | 0.1104 | 0.0965 | 0.0382 | 56.7 deg |
| 0.50 | 401 | 0.0873 | 0.1136 | 0.0987 | 0.0341 | 30.2 deg |
| 0.70 | 406 | 0.1650 | 0.2175 | 0.1877 | 0.1047 | 18.8 deg |
| 0.80 | 417 | 0.2182 | 0.2955 | 0.2510 | 0.1675 | 28.2 deg |
| 0.90 | 333 | 0.3574 | 0.3864 | 0.3713 | 0.2767 | 32.1 deg |
| 0.93 | 284 | 0.4577 | 0.4221 | 0.4392 | 0.3476 | 29.4 deg |
| 0.95 | 248 | 0.5403 | 0.4351 | 0.4820 | 0.3888 | 30.0 deg |
| 0.96 | 237 | 0.5485 | 0.4221 | 0.4771 | 0.3854 | 27.9 deg |
| 0.98 | 208 | 0.5240 | 0.3539 | 0.4225 | 0.3166 | 25.2 deg |

Best F1 in this sweep: threshold 0.95.

## Interpretation

This first plain-PyTorch DINOv3 heatmap model is not yet a replacement for the
best DINOv3/MMDetection detector by recall or mAP, but it is a credible signal:
it trained without OpenMMLab and reached 54% precision / 43.5% recall / 48.2%
F1 after only 512 training samples and a very small heatmap head.

The biggest current weakness is geometry. Angle error remains high because the
first spike fits OBBs directly from coarse 24x24 heatmap components after square
resizing. Next iteration should use aspect-preserving letterbox, full training
data, and Radon/line fitting on the original image around the heatmap component.
