# DINOv3 Heatmap Spike

## Goal

Explore whether frozen DINOv3 patch features plus a lightweight heatmap head can
detect satellite streaks more directly than the current bounded-box detector.
The spike is intentionally plain PyTorch: no MMDetection, no DETR queries, and
no CUDA-only assumptions.

## Current Experiment Path

The branch already contains the core spike pieces:

- `models/plain_dinov3/streak_heatmap.py` loads a frozen DINOv3 ViT encoder and
  predicts a low-resolution patch-grid heatmap.
- `training/dinov3_heatmap_dataset.py` converts COCO annotations into streak
  occupancy heatmaps, center heatmaps, and geometry targets.
- `scripts/cache_dinov3_heatmap_features.py` caches frozen DINOv3 features so
  the head can be trained quickly without repeatedly running the backbone.
- `training/train_dinov3_heatmap_cached.py` trains a segmentation-style heatmap
  plus geometry head from cached features.
- `training/train_dinov3_box_cached.py` trains a center heatmap plus direct OBB
  regression head from the same cached features.
- `scripts/evaluate_dinov3_heatmap.py` converts connected heatmap components
  back into ARGUS OBB detections for existing metrics.
- `scripts/evaluate_dinov3_box.py` evaluates the center/box variant.

## Recommended First Run

Use small caps first to validate the workflow on the Mac, then remove
`--max-samples` for the real run.

```bash
cd /Users/robert/Argus-dinov3-heatmap-spike
export PYTORCH_ENABLE_MPS_FALLBACK=1

/Users/robert/miniconda3/envs/satid/bin/python scripts/cache_dinov3_heatmap_features.py \
  --annotations data/annotations/train.json \
  --output-dir weights/cache_plain_dinov3_train_384_smoke \
  --weights weights/dinov3_vitb16_lvd1689m.pth \
  --image-size 384 \
  --batch-size 1 \
  --max-samples 16

/Users/robert/miniconda3/envs/satid/bin/python scripts/cache_dinov3_heatmap_features.py \
  --annotations data/annotations/val.json \
  --output-dir weights/cache_plain_dinov3_val_384_smoke \
  --weights weights/dinov3_vitb16_lvd1689m.pth \
  --image-size 384 \
  --batch-size 1 \
  --max-samples 16

/Users/robert/miniconda3/envs/satid/bin/python training/train_dinov3_heatmap_cached.py \
  --train-cache weights/cache_plain_dinov3_train_384_smoke \
  --val-cache weights/cache_plain_dinov3_val_384_smoke \
  --work-dir weights/run_plain_dinov3_heatmap_cached_smoke \
  --epochs 3 \
  --batch-size 8

/Users/robert/miniconda3/envs/satid/bin/python scripts/evaluate_dinov3_heatmap.py \
  --annotations data/annotations/test.json \
  --checkpoint weights/run_plain_dinov3_heatmap_cached_smoke/best.pt \
  --output results/plain_dinov3_heatmap_smoke/metrics.json \
  --max-samples 32 \
  --no-refine-geometry
```

## Full-Scale Variant

For a serious comparison, cache `train.json`, `val.json`, and evaluate on
`test.json` without `--max-samples`. Start with `image-size=384` or `512`.
The cached training step is cheap once features exist, so sweep these first:

- `--threshold`: `0.25`, `0.35`, `0.5`, `0.65`
- `--min-pixels`: `1`, `2`, `3`
- `--geometry-weight`: `0.1`, `0.25`, `0.5`
- `--pos-weight`: `10`, `20`, `40`

## Decision Metrics

Compare against the current DINOv3 ViT-B box detector result from
`agent_docs/dinov3_plan.md`:

- test `mAP@0.5`: current box detector is about `0.74`
- precision and recall at the chosen confidence threshold
- short-streak recall, because this is where patch-grid heatmaps may help or
  hurt most
- false positives on negative-only images

## Notes

The heatmap head predicts at DINOv3 patch resolution, so very thin streaks are
still quantized to 16 px cells. If the full heatmap variant underperforms but
the center/box variant localizes well, try using the heatmap only as an object
proposal mechanism and keep the existing Radon refinement for final geometry.

## First Full 384px Run

Run directory:

```text
weights/run_plain_dinov3_heatmap_full_384_e100/
```

Feature caches:

```text
weights/cache_plain_dinov3_train_384/
weights/cache_plain_dinov3_val_384/
```

Training reached best validation Dice `0.847` at epoch 97, saved as:

```text
weights/run_plain_dinov3_heatmap_full_384_e100/best.pt
```

Test-set evaluation showed that heatmap occupancy alone is not yet enough for
ARGUS-quality detections. The best threshold in the first sweep was `0.65`:

| Threshold | Predictions | Precision | Recall | F1 | mAP@50 |
|-----------|-------------|-----------|--------|----|--------|
| 0.25 | 409 | 0.0905 | 0.1201 | 0.1032 | 0.0281 |
| 0.35 | 398 | 0.0955 | 0.1234 | 0.1076 | 0.0284 |
| 0.50 | 386 | 0.1062 | 0.1331 | 0.1182 | 0.0340 |
| 0.65 | 382 | 0.1099 | 0.1364 | 0.1217 | 0.0353 |

The failure mode is geometric: predicted streak lengths are plausible, but
component-derived widths are much too large, often hundreds of pixels after
scaling back to original resolution. The next spike should keep the heatmap as a
proposal/centerline signal and replace component width with either a thinner
target, a centerline skeleton target, or local Radon/box regression geometry.

## Orientation-Binned Centerline Variant

This is the reproduction path for the no-box detector methodology:

- Train split uses native-pixel tiles.
- Validation and holdout use full native frames.
- Model input is a bilinear resize to the DINO input size.
- Target is an 18-channel soft centerline heatmap, one channel per orientation
  bin.
- The DINOv3 backbone is frozen; only the centerline decoder is trained.

The local Mac path defaults to DINOv3 ViT-B because ViT-L is larger than the
comfortable local training envelope. The code still supports `--model-size
large` when running on a larger GPU with the ViT-L weights.

Weights linked in this worktree:

```text
weights/dinov3_vitb16_lvd1689m.pth -> /Users/robert/Argus/weights/dinov3_vitb16_lvd1689m.pth
weights/dinov3_vitl16_lvd1689m.pth -> /Users/robert/Argus/weights/dinov3_vitl16_lvd1689m.pth
```

Smoke run:

```bash
cd /Users/robert/Argus-dinov3-heatmap-spike
export PYTORCH_ENABLE_MPS_FALLBACK=1

/Users/robert/miniconda3/envs/satid/bin/python training/train_dinov3_orientation_centerline.py \
  --train-annotations data/annotations/train.json \
  --val-annotations data/annotations/val.json \
  --work-dir weights/run_dinov3_orientation_centerline_smoke \
  --weights weights/dinov3_vitb16_lvd1689m.pth \
  --image-size 128 \
  --tile-size 2560 \
  --positive-train-tiles 1 \
  --negative-train-tiles 1 \
  --max-train-samples 2 \
  --max-val-samples 2 \
  --epochs 1 \
  --batch-size 1 \
  --workers 0 \
  --decoder-channels 32 \
  --last-layers 1
```

Local ViT-B reproduction run:

```bash
cd /Users/robert/Argus-dinov3-heatmap-spike
export PYTORCH_ENABLE_MPS_FALLBACK=1

/Users/robert/miniconda3/envs/satid/bin/python training/train_dinov3_orientation_centerline.py \
  --train-annotations data/annotations/train.json \
  --val-annotations data/annotations/val.json \
  --work-dir weights/run_dinov3_vitb_orientation_centerline_tile2560_input1024 \
  --weights weights/dinov3_vitb16_lvd1689m.pth \
  --model-size base \
  --tile-size 2560 \
  --image-size 1024 \
  --positive-train-tiles 1236 \
  --negative-train-tiles 1400 \
  --orientation-bins 18 \
  --decoder-channels 192 \
  --last-layers 4 \
  --centerline-width 2.0 \
  --centerline-sigma 1.4 \
  --neighbor-bin-weight 0.35 \
  --second-neighbor-weight 0.0 \
  --epochs 5 \
  --batch-size 1 \
  --lr 5e-5 \
  --min-lr 1e-5 \
  --weight-decay 1e-4 \
  --pos-weight 120 \
  --dice-weight 1.0 \
  --bce-weight 0.35 \
  --orientation-ce-weight 0.20 \
  --manual-positive-weight 8.0 \
  --workers 0 \
  --seed 20260524 \
  --preserve-image-bit-depth
```

On CUDA, use `--workers 8 --amp`. On MPS, keep workers at `0`; Dataloader
multiprocessing is not reliable there.

Native no-OBB evaluation:

```bash
cd /Users/robert/Argus-dinov3-heatmap-spike
export PYTORCH_ENABLE_MPS_FALLBACK=1

/Users/robert/miniconda3/envs/satid/bin/python scripts/evaluate_dinov3_orientation_centerline.py \
  --checkpoint weights/run_dinov3_vitb_orientation_centerline_tile2560_input1024/best.pt \
  --annotations data/annotations/val.json \
  --output results/dinov3_vitb_orientation_centerline_val/metrics.json \
  --thresholds 0.30,0.50,0.70,0.85 \
  --target-threshold 0.05 \
  --component-iou 0.10 \
  --distance-tolerance-px 3 \
  --component-coverage 0.30 \
  --min-component-pixels 8 \
  --batch-size 1 \
  --workers 0
```

Use the same command with `--annotations data/annotations/test.json` for the
current holdout-style split.

The native evaluator intentionally does not convert to OBBs. It reports:

- soft centerline Dice
- thresholded centerline pixel precision/recall/F1
- image-level streak/no-streak precision/recall/F1/accuracy
- connected centerline component precision/recall/F1
- distance-tolerant centerline pixel precision/recall/F1
- distance-tolerant component precision/recall/F1
- orientation-bin accuracy on ground-truth centerline pixels

Replacement decision should be based on whether this model can meet the same
product need without boxes: high no-streak specificity, strong streak-frame
recall, clean centerline components, and useful orientation estimates. OBB
conversion should remain a legacy adapter only if downstream ARGUS consumers
still require it.

Visual QA overlays:

```bash
cd /Users/robert/Argus-dinov3-heatmap-spike
export PYTORCH_ENABLE_MPS_FALLBACK=1

/Users/robert/miniconda3/envs/satid/bin/python scripts/render_dinov3_orientation_centerline_overlays.py \
  --checkpoint weights/run_dinov3_vitb_orientation_centerline_tile2560_input1024/best.pt \
  --annotations data/annotations/val.json \
  --output-dir results/dinov3_vitb_orientation_centerline_val_overlays \
  --threshold 0.70 \
  --max-samples 24 \
  --workers 0
```

Each overlay PNG has five panels: source image, ground-truth centerline,
predicted heatmap, thresholded predicted mask, and predicted orientation hue.
Use these before trusting any scalar metric; they expose broad haze, shifted
lines, broken components, and bad orientation bins immediately.

## Local ViT-B 512px Pilot

The exact 1024px ViT-B run was not practical on local MPS. A single 1024px
attention pass took minutes before reaching batch-level progress. The local
pilot therefore used the same tile manifest and centerline objective at
`--image-size 512`.

Run directory:

```text
weights/run_dinov3_vitb_orientation_centerline_tile2560_input512/
```

Training summary:

| Epoch | Train Dice | Val Dice | Val Loss |
|-------|------------|----------|----------|
| 1 | 0.0055 | 0.0184 | 1.3931 |
| 2 | 0.0101 | 0.0297 | 1.3506 |
| 3 | 0.0142 | 0.0517 | 1.2903 |
| 4 | 0.0184 | 0.0592 | 1.2576 |
| 5 | 0.0221 | 0.0659 | 1.2340 |

Best checkpoint:

```text
weights/run_dinov3_vitb_orientation_centerline_tile2560_input512/best.pt
```

Native val evaluation:

```text
results/dinov3_vitb_orientation_centerline_input512_val/metrics.json
```

Key metrics:

| Metric | Value |
|--------|-------|
| Soft Dice | 0.0659 |
| Orientation accuracy on GT pixels | 0.8742 |
| Orientation within 1 bin on GT pixels | 0.8902 |
| Best exact pixel F1 | 0.0521 at threshold 0.85 |
| Best distance-tolerant pixel F1 | 0.0960 at threshold 0.85 |
| Best image-level F1 | 0.9611 at threshold 0.30 |

Threshold sweep:

| Threshold | Pixel P/R/F1 | Tol. Pixel P/R/F1 | Image P/R/F1 | Tol. Component P/R/F1 |
|-----------|--------------|-------------------|--------------|-----------------------|
| 0.30 | 0.0147 / 0.9680 / 0.0290 | 0.0277 / 0.9860 / 0.0539 | 0.9972 / 0.9275 / 0.9611 | 0.0840 / 0.2047 / 0.1191 |
| 0.50 | 0.0176 / 0.9489 / 0.0346 | 0.0332 / 0.9773 / 0.0641 | 0.9972 / 0.9197 / 0.9569 | 0.0568 / 0.1969 / 0.0882 |
| 0.70 | 0.0216 / 0.9150 / 0.0423 | 0.0407 / 0.9603 / 0.0781 | 0.9971 / 0.8886 / 0.9397 | 0.0611 / 0.1580 / 0.0882 |
| 0.85 | 0.0269 / 0.8548 / 0.0521 | 0.0506 / 0.9271 / 0.0960 | 1.0000 / 0.8264 / 0.9050 | 0.0501 / 0.0959 / 0.0658 |

Overlay output:

```text
results/dinov3_vitb_orientation_centerline_input512_val_overlays/
```

Interpretation: this first ViT-B local run learned image-level streak presence
and orientation reasonably well, but it does not yet produce clean enough
centerline masks. Pixel precision and component precision are too low, so the
next model-side change should target sharper heatmaps: lower BCE weight,
stronger Dice/positive skeleton emphasis, harder negatives, or a two-stage
image-level gate plus centerline decoder.

## Local Sharp + Image-Gate Run

Because higher-powered CUDA training is not assumed available, the strongest
local path is ViT-B at 512px with sharper heatmap loss and an auxiliary
image-level streak head. This run used:

- `--image-size 512`
- `--epochs 10`
- `--pos-weight 60`
- `--bce-weight 0.10`
- `--manual-positive-weight 3.0`
- `--image-loss-weight 0.25`

Run directory:

```text
weights/run_dinov3_vitb_orientation_centerline_input512_sharp_gate/
```

Best checkpoint:

```text
weights/run_dinov3_vitb_orientation_centerline_input512_sharp_gate/best.pt
```

Training summary:

| Epoch | Train Dice | Val Dice | Val Loss |
|-------|------------|----------|----------|
| 1 | 0.0039 | 0.0074 | 1.4278 |
| 2 | 0.0040 | 0.0087 | 1.2584 |
| 3 | 0.0047 | 0.0107 | 1.2538 |
| 4 | 0.0072 | 0.0240 | 1.2441 |
| 5 | 0.0129 | 0.0420 | 1.2336 |
| 6 | 0.0173 | 0.0581 | 1.2103 |
| 7 | 0.0204 | 0.0781 | 1.2523 |
| 8 | 0.0230 | 0.0742 | 1.1785 |
| 9 | 0.0247 | 0.0841 | 1.1661 |
| 10 | 0.0271 | 0.0807 | 1.1640 |

Native val evaluation:

```text
results/dinov3_vitb_orientation_centerline_input512_sharp_gate_val/metrics.json
```

Comparison to the earlier 512px baseline:

| Metric | Baseline 512 | Sharp + Gate |
|--------|--------------|--------------|
| Soft Dice | 0.0659 | 0.0841 |
| Orientation accuracy on GT pixels | 0.8742 | 0.9004 |
| Orientation within 1 bin | 0.8902 | 0.9266 |
| Best exact pixel F1 | 0.0521 | 0.1014 |
| Best distance-tolerant pixel F1 | 0.0960 | 0.1823 |
| Best image-level F1 | 0.9611 | 0.9616 |

Threshold sweep for sharp + gate:

| Threshold | Pixel P/R/F1 | Tol. Pixel P/R/F1 | Image P/R/F1 | Tol. Component P/R/F1 |
|-----------|--------------|-------------------|--------------|-----------------------|
| 0.30 | 0.0380 / 0.7587 / 0.0724 | 0.0714 / 0.8690 / 0.1320 | 0.9495 / 0.9741 / 0.9616 | 0.0619 / 0.2358 / 0.0981 |
| 0.50 | 0.0427 / 0.6795 / 0.0804 | 0.0803 / 0.8146 / 0.1462 | 0.9862 / 0.9275 / 0.9559 | 0.0913 / 0.2150 / 0.1282 |
| 0.70 | 0.0483 / 0.6032 / 0.0895 | 0.0906 / 0.7555 / 0.1618 | 0.9944 / 0.9171 / 0.9542 | 0.1232 / 0.2176 / 0.1573 |
| 0.85 | 0.0563 / 0.5129 / 0.1014 | 0.1053 / 0.6781 / 0.1823 | 0.9971 / 0.8938 / 0.9426 | 0.1530 / 0.2073 / 0.1760 |

Overlay output:

```text
results/dinov3_vitb_orientation_centerline_input512_sharp_gate_val_overlays/
```

Interpretation: this is the best local no-OBB result so far. It cuts down the
broad haze seen in the baseline and roughly doubles pixel/component F1, while
keeping image-level detection essentially unchanged. It still produces masks
that are too fat to call final centerlines, so the next local step should add a
thinner target or explicit skeleton/edge penalty rather than returning to box
postprocessing.

## No-OBB Segment Proposal Path

Added:

```text
scripts/propose_dinov3_centerline_segments.py
```

This is the first implementation of the no-OBB post-processing route:

1. run the orientation-binned DINOv3 centerline model,
2. threshold the heatmap into seed components,
3. reject components whose orientation bins are not locally consistent,
4. refine each seed angle with local Radon, and
5. emit line-segment proposals as endpoints, not oriented boxes.

Example local command:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 /Users/robert/miniconda3/envs/satid/bin/python scripts/propose_dinov3_centerline_segments.py \
  --checkpoint weights/run_dinov3_vitb_orientation_centerline_input512_sharp_gate/best.pt \
  --annotations data/annotations/val.json \
  --output results/dinov3_vitb_orientation_centerline_input512_sharp_gate_segments_25/proposals.json \
  --overlay-dir results/dinov3_vitb_orientation_centerline_input512_sharp_gate_segments_25_overlays \
  --max-samples 25 \
  --max-overlays 12 \
  --max-components-per-image 4
```

The JSON output contains, per image:

- `segments`: line proposals with `input_start`, `input_end`, `native_start`,
  `native_end`, score, dominant orientation bin, orientation consistency, seed
  angle, Radon-refined angle, and GT coverage diagnostics.
- `proposal_metrics`: image-level proposal recall and negative proposal rate
  for the processed slice.

Current 25-image positive slice smoke result:

| Metric | Value |
|--------|-------|
| Positive images with any segment | 25 / 25 |
| Positive image proposal recall | 1.000 |
| Positive images matched to GT coverage | 16 / 25 |
| Positive matched recall | 0.640 |
| Segments per image | 1.84 |
| Best GT coverage min / median / max | 0.000 / 0.159 / 1.000 |

Interpretation: the plumbing works, and the segment output is now inspectable.
The current model can usually produce at least one seed on positive images, but
the seed geometry is not reliable enough yet. The next model/post-processing
work should focus on reducing broad hot bands, suppressing secondary hot spots,
and training/evaluating against a wider catchment target so the segment starts
close enough for Radon and downstream streak tracing to recover the true line.

## Catchment-Band Model Training

Added a second optional training target to:

```text
training/dinov3_orientation_centerline_dataset.py
```

The dataset still returns the original orientation-binned soft centerline target
as `target`. It can now also return `catchment_target`, a wider orientation-
binned band around the same line. The trainer in:

```text
training/train_dinov3_orientation_centerline.py
```

adds `--catchment-width`, `--catchment-sigma`, and a catchment loss mix:

- `--catchment-loss-weight`
- `--catchment-pos-weight`
- `--catchment-dice-weight`
- `--catchment-bce-weight`

This keeps the output no-OBB: the model still predicts orientation-binned
heatmaps only. The catchment loss teaches the decoder to place recoverable seed
activation near the streak while the original core target keeps pressure on the
thin centerline.

Smoke verification:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 /Users/robert/miniconda3/envs/satid/bin/python training/train_dinov3_orientation_centerline.py \
  --train-annotations data/annotations/train.json \
  --val-annotations data/annotations/val.json \
  --work-dir weights/run_dinov3_orientation_centerline_catchment_smoke \
  --weights weights/dinov3_vitb16_lvd1689m.pth \
  --model-size base \
  --tile-size 2560 \
  --image-size 128 \
  --positive-train-tiles 4 \
  --negative-train-tiles 4 \
  --decoder-channels 64 \
  --last-layers 1 \
  --catchment-width 14.0 \
  --catchment-sigma 6.0 \
  --epochs 1 \
  --max-train-samples 2 \
  --max-val-samples 2 \
  --catchment-loss-weight 0.35 \
  --image-loss-weight 0.25 \
  --workers 0 \
  --preserve-image-bit-depth
```

Smoke result: completed one epoch, wrote
`weights/run_dinov3_orientation_centerline_catchment_smoke/best.pt`, and
recorded catchment metrics in `history.json`.

Local training launch script:

```text
scripts/run_centerline_catchment_local.sh
```

This is the next ViT-B 512px run to start when ready:

```bash
scripts/run_centerline_catchment_local.sh
```

Expected follow-up after training:

1. evaluate with `scripts/evaluate_dinov3_orientation_centerline.py`,
2. render overlays with `scripts/render_dinov3_orientation_centerline_overlays.py`,
3. run `scripts/propose_dinov3_centerline_segments.py`,
4. compare proposal recall and false segments against the sharp+gate baseline.

### Catchment ViT-B 512px Local Run Result

Run:

```text
weights/run_dinov3_vitb_orientation_centerline_input512_catchment/
```

Training completed cleanly in about 7 hr 18 min on Apple Silicon MPS. Best
checkpoint was epoch 9:

```text
weights/run_dinov3_vitb_orientation_centerline_input512_catchment/best.pt
```

Epoch summary:

| Epoch | Train Dice | Val Dice |
|-------|------------|----------|
| 1 | 0.0105 | 0.0501 |
| 2 | 0.0202 | 0.0694 |
| 3 | 0.0249 | 0.0776 |
| 4 | 0.0291 | 0.1056 |
| 5 | 0.0325 | 0.1070 |
| 6 | 0.0351 | 0.1125 |
| 7 | 0.0376 | 0.1206 |
| 8 | 0.0388 | 0.1175 |
| 9 | 0.0397 | 0.1217 |
| 10 | 0.0404 | 0.1174 |

Evaluation:

```text
results/dinov3_vitb_orientation_centerline_input512_catchment_val/metrics.json
results/dinov3_vitb_orientation_centerline_input512_catchment_val_overlays/
results/dinov3_vitb_orientation_centerline_input512_catchment_segments_25/proposals.json
results/dinov3_vitb_orientation_centerline_input512_catchment_segments_25_overlays/
```

Compared with the previous sharp+gate run:

| Metric | Sharp+gate | Catchment |
|--------|------------|-----------|
| Soft Dice | 0.0841 | 0.1217 |
| Orientation accuracy | 0.9004 | 0.9044 |
| Orientation within 1 bin | 0.9266 | 0.9290 |
| Best exact pixel F1 | 0.1014 | 0.1311 |
| Best distance-tolerant pixel F1 | 0.1823 | 0.2273 |
| Best image-level F1 | 0.9616 | 0.9856 |
| Best distance-tolerant component F1 | 0.1760 | 0.2798 |

The catchment model is a clear win on full validation metrics. However, the
first no-OBB segment proposal pass at threshold 0.85 became more conservative:

| 25-image proposal metric | Sharp+gate | Catchment |
|--------------------------|------------|-----------|
| Positive images with segment | 25 / 25 | 16 / 25 |
| Matched positive images | 16 / 25 | 10 / 25 |
| Segments per image | 1.84 | 0.76 |
| Best coverage min / median / max | 0.000 / 0.159 / 1.000 | 0.000 / 0.000 / 0.842 |

Interpretation: the catchment training improved the heatmap substantially, but
the segment extractor threshold/logic needs retuning for this model. The next
no-OBB step should sweep proposal thresholds and orientation-consistency
settings, then update segment extraction to use a lower catchment seed threshold
plus stricter line/refinement scoring.

### Catchment Segment Sweep

Added a cached sweep runner:

```text
scripts/sweep_dinov3_centerline_segments.py
```

The sweep runs DINOv3 once per validation image and evaluates multiple
heatmap-to-segment settings against the cached heatmap. Full validation sweep:

```text
results/dinov3_vitb_orientation_centerline_input512_catchment_segment_sweep/sweep.json
results/dinov3_vitb_orientation_centerline_input512_catchment_segment_sweep/sweep.csv
```

Best balanced setting from the focused grid:

```text
threshold=0.50
min_orientation_consistency=0.55
min_component_pixels=4
max_components_per_image=4
```

Validation proposal metrics for that setting:

| Metric | Value |
|--------|-------|
| Positive images | 386 |
| Negative images | 25 |
| Positive images with segment | 376 / 386 |
| Positive segment recall | 0.9741 |
| Positive images matched | 284 / 386 |
| Positive matched recall | 0.7358 |
| Negative images with segment | 1 / 25 |
| Negative segment rate | 0.0400 |
| Total segments | 770 |
| Segments per image | 1.8735 |
| Median best GT coverage | 0.426 |

Detailed proposal output and overlays:

```text
results/dinov3_vitb_orientation_centerline_input512_catchment_segments_t050_oc055/proposals.json
results/dinov3_vitb_orientation_centerline_input512_catchment_segments_t050_oc055_overlays/
```

Interpretation: the retuned no-OBB path is a major improvement over the initial
0.85 threshold proposal pass (`positive_matched_recall` 0.40 on the 25-image
slice vs 0.7358 across full validation). The remaining misses are frequently
geometry/extraction misses rather than raw heatmap misses: the heat is present,
but the current component-to-line fit can choose the wrong local support. Next
engineering step is to improve the line support score and tracing logic before
retraining.

### Main Pipeline Integration

The heatmap detector is now integrated as a first-class optional ARGUS detector:

```text
dinov3_heatmap_centerline
```

Implementation:

```text
inference/heatmap_detector.py
inference/pipeline.py
inference/confidence.py
```

Native detector output is a line segment:

```json
{
  "geometry_type": "line_segment",
  "line_segment": {"x1": 0, "y1": 0, "x2": 100, "y2": 10, "angle_deg": 5.7, "length_px": 100.5}
}
```

For compatibility with the current pipeline, WCS endpoint conversion, grouping,
database storage, and frontend rendering, the detector also emits:

```text
obb_compat
obb
```

These are projection fields only; they are not the native model output. This
lets the method appear in `/api/detectors` and participate in the existing
multi-detector stack without changing the no-OBB evaluation path.

The React UI treats it as a normal selectable/runnable detector in the detector
panel, confidence filters, result table badges, and canvas overlay colors.

Default checkpoint:

```text
weights/run_dinov3_vitb_orientation_centerline_input512_catchment/best.pt
```

Override knobs:

```text
HEATMAP_CENTERLINE_CHECKPOINT
HEATMAP_DINOV3_WEIGHTS
HEATMAP_IMAGE_SIZE
HEATMAP_SEGMENT_THRESHOLD
HEATMAP_MIN_ORIENTATION_CONSISTENCY
HEATMAP_MIN_COMPONENT_PIXELS
HEATMAP_MAX_COMPONENTS
```

Smoke test:

```text
run_with_array(..., models=[], enabled_detectors={"dinov3_heatmap_centerline"}, raw_mode=True)
```

returned one native line-segment detection on `data/sample/synth_streak_000.fits`.

### Heatmap-vs-OBB Comparison Harness

Added native centerline comparison utilities:

```text
eval/line_metrics.py
scripts/compare_heatmap_centerline_to_obb.py
```

The comparison converts OBB detector outputs to their centerlines and compares
both methods with the same distance-tolerant line metric. This is stricter than
the earlier proposal recall because it requires mutual support between the
predicted segment and the ground-truth centerline, not just "does the predicted
line touch at least 10% of the GT?".

Current heatmap proposal file evaluated with strict line metrics:

```text
results/dinov3_vitb_orientation_centerline_input512_catchment_segments_t050_oc055/line_metrics.json
```

Result:

| Metric | Value |
|--------|-------|
| Precision | 0.0870 |
| Recall | 0.1736 |
| F1 | 0.1159 |
| TP / FP / FN | 67 / 703 / 319 |
| Mean GT coverage on matches | 0.5496 |
| Mean predicted coverage on matches | 0.4792 |
| Mean angle error on matches | 8.183 deg |

Interpretation: the integrated detector can now be compared against OBB methods
on equal line geometry, but the current line extraction is still too fragmented
and includes many segment proposals that do not mutually support the GT
centerline. The next quality step is line-support scoring/tracing, not another
backbone training run.

### First Side-by-Side Comparison (val.json)

Ran `scripts/run_box_eval_and_compare.sh` to regenerate a plain DINOv3 box head
from the existing 384px cached features (80 epochs, val center Dice 0.517) and
compare against the heatmap centerline proposals using `eval/line_metrics.py`
(tolerance 6 px, coverage threshold 0.10):

Initial baseline (t=0.50, no line-support gate, max_components=4):

| Metric | Heatmap centerline | Plain DINOv3 box (384px) |
|--------|-------------------|--------------------------|
| Precision | 0.087 | **0.238** |
| Recall | 0.174 | **0.249** |
| F1 | 0.116 | **0.243** |
| TP / FP / FN | 67 / 703 / 319 | 96 / 307 / 290 |
| Predictions | 770 | 403 |
| Mean angle error on matches | 8.2° | **1.4°** |

Interpretation: box head won on all line-segment metrics. The heatmap was
over-generating proposals (703 FPs vs 307 for box head). The next step was to
add line-support scoring to prune low-quality proposals.

### Line-Support Scoring Analysis (val.json)

Ran `scripts/propose_dinov3_centerline_segments.py` with gates disabled
(`min_line_support=0.0`, `min_radon_snr=1.0`) on all 411 val images to collect
raw `line_support_ratio` and `radon_snr` distributions. Key findings:

**radon_snr** (peak/mean Radon column variance) is clustered at 1.0–1.2 for
all proposals — genuine streak components do NOT produce discriminative sinogram
peaks. Radon angle refinement adds noise rather than resolving angle ambiguity.

**line_support_ratio** (heat-weighted fraction of component pixels within 3 px
of the fitted line) appears anti-correlated with GT match at image-level: matched
proposals median LSR=0.14 vs unmatched median LSR=0.38. This is because genuine
streak heatmaps form large diffuse blobs (wide, low LSR) while incidental noise
components happen to be more compact/linear (high LSR). However, at
segment-level (strict bidirectional coverage), LSR>=0.50 is a strong filter:

| Config | Proposals | F1 | TP | FP | Precision | Recall |
|--------|-----------|----|----|-----|-----------|--------|
| t=0.50, lsr=0.00 | 770 | 0.116 | 67 | 703 | 0.087 | 0.174 |
| t=0.70, lsr=0.00 | 596 | 0.143 | 70 | 526 | 0.117 | 0.181 |
| t=0.80, lsr=0.50 | 176 | 0.210 | 59 | 117 | 0.335 | 0.153 |
| t=0.85, lsr=0.50 | 192 | 0.211 | 61 | 131 | 0.318 | 0.158 |
| **t=0.85, lsr=0.50, mc=2** | **175** | **0.218** | **61** | **114** | **0.349** | **0.158** |
| Box head (OBB) | 403 | **0.243** | 96 | 307 | 0.238 | 0.249 |

The sweep also confirmed that `max_components=2` (per image) reduces FPs slightly
(192→175) with no recall loss, so this was adopted as the new default.

Updated defaults in `scripts/propose_dinov3_centerline_segments.py`:
- `--threshold 0.85`
- `--min-line-support 0.50`
- `--max-components-per-image 2`

Updated comparison result (now saved to `results/heatmap_vs_obb_line_comparison.json`):

| Metric | **Heatmap (tuned)** | Plain DINOv3 box (384px) |
|--------|-------------------|--------------------------|
| Precision | **0.349** | 0.238 |
| Recall | 0.158 | **0.249** |
| F1 | 0.218 | **0.243** |
| TP / FP / FN | 61 / 114 / 325 | 96 / 307 / 290 |
| Predictions | 175 | 403 |
| Mean angle error on matches | 9.96° | **1.4°** |

The tuned heatmap now has **higher precision** than the box head (0.349 vs 0.238),
a net F1 improvement of 88% over the baseline (0.116→0.218), and is within 10%
of the box head F1. The main remaining gaps are:
- **Recall**: heatmap covers 16% of GT streaks vs 25% for box head — limited by
  fundamental heatmap sensitivity at the current model size/training
- **Angle error**: 10° vs 1.4° — Radon SNR is near 1.0 for most components,
  so Radon provides no refinement over the model's own orientation bins; the
  seed angle (model output directly) gives 7.8° error — slightly better

## Exact Split / CUDA Notes

The colleague split cannot be recreated from the currently linked local
annotations alone. Across `train.json`, `val.json`, and `test.json` there are
only 91 full-frame negative images available, but the colleague split needs
376 negatives for val+holdout (`160 + 216`).

The split builder records this explicitly:

```bash
/Users/robert/miniconda3/envs/satid/bin/python scripts/build_centerline_fullframe_splits.py \
  --output-dir data/annotations/centerline_fullframe_splits
```

A feasible local split using all 91 available negatives was created at:

```text
data/annotations/centerline_fullframe_splits_feasible/
```

CUDA 1024 launch script:

```text
scripts/run_centerline_cuda_1024.sh
```

This Mac reports `cuda=False`, `mps=True`, so the script is prepared and
syntax-checked but cannot be executed on this machine.
