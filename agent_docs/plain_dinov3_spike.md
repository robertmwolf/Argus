# Plain PyTorch DINOv3 Streak Spike

This spike tests whether ARGUS can replace the OpenMMLab/MMDetection DINO path
with a smaller native PyTorch model for streak detection.

## Goal

Train a DINOv3-based streak identifier without `mmcv`, `mmdet`, or `mmengine`.
The model predicts a low-resolution streak heatmap from a frozen DINOv3 encoder,
then converts connected heatmap components into ARGUS-style OBB detections for
comparison against existing metrics.

This is intentionally parallel to the current Phase D plan. It does not remove
or modify the MMDetection training path.

## What It Uses

| Component | File | Dependency |
|-----------|------|------------|
| Model | `models/plain_dinov3/streak_heatmap.py` | PyTorch + `dinov3` only |
| Dataset | `training/dinov3_heatmap_dataset.py` | COCO JSON, FITSLoader, PIL |
| Training | `training/train_dinov3_heatmap.py` | plain PyTorch loop |
| Feature cache | `scripts/cache_dinov3_heatmap_features.py` | one frozen DINOv3 pass |
| Cached training | `training/train_dinov3_heatmap_cached.py` | fast head-only iterations |
| Cached box training | `training/train_dinov3_box_cached.py` | direct center/box head |
| Evaluation | `scripts/evaluate_dinov3_heatmap.py` | ARGUS `eval.metrics` |
| Box evaluation | `scripts/evaluate_dinov3_box.py` | ARGUS `eval.metrics` |

The only DINOv3-specific external dependency is:

```bash
pip install git+https://github.com/facebookresearch/dinov3.git
```

No OpenMMLab package is required for this spike.

## Training

ViT-B first, because it is smaller and validates the idea quickly:

```bash
conda activate satid

python -m training.train_dinov3_heatmap \
    --train-annotations data/annotations/train.json \
    --val-annotations data/annotations/val.json \
    --weights weights/dinov3_vitb16_lvd1689m.pth \
    --model-size base \
    --image-size 512 \
    --batch-size 2 \
    --epochs 10 \
    --work-dir weights/run_plain_dinov3_heatmap
```

Smoke test:

```bash
python -m training.train_dinov3_heatmap \
    --train-annotations data/annotations/dev_subset.json \
    --val-annotations data/annotations/dev_subset.json \
    --weights weights/dinov3_vitb16_lvd1689m.pth \
    --model-size base \
    --image-size 256 \
    --smoke-test
```

## Evaluation

```bash
python scripts/evaluate_dinov3_heatmap.py \
    --annotations data/annotations/test.json \
    --checkpoint weights/run_plain_dinov3_heatmap/best.pt \
    --output results/plain_dinov3_heatmap/metrics.json
```

Outputs:

```text
results/plain_dinov3_heatmap/metrics.json
results/plain_dinov3_heatmap/predictions.json
```

## Cached Feature Workflow

Use this once the basic training path works. Caching avoids recomputing frozen
DINOv3 features every epoch.

```bash
python scripts/cache_dinov3_heatmap_features.py \
    --annotations data/annotations/train.json \
    --output-dir data/cache/plain_dinov3/train_384 \
    --weights weights/dinov3_vitb16_lvd1689m.pth \
    --model-size base \
    --image-size 384 \
    --batch-size 4

python scripts/cache_dinov3_heatmap_features.py \
    --annotations data/annotations/val.json \
    --output-dir data/cache/plain_dinov3/val_384 \
    --weights weights/dinov3_vitb16_lvd1689m.pth \
    --model-size base \
    --image-size 384 \
    --batch-size 4

python -m training.train_dinov3_heatmap_cached \
    --train-cache data/cache/plain_dinov3/train_384 \
    --val-cache data/cache/plain_dinov3/val_384 \
    --work-dir weights/run_plain_dinov3_heatmap_cached \
    --epochs 50 \
    --batch-size 32 \
    --geometry-weight 0.25
```

Cached checkpoints can be evaluated with the same evaluator:

```bash
python scripts/evaluate_dinov3_heatmap.py \
    --annotations data/annotations/test.json \
    --checkpoint weights/run_plain_dinov3_heatmap_cached/best.pt \
    --output results/plain_dinov3_heatmap_cached/metrics.json \
    --threshold 0.95
```

Use `--no-refine-geometry` when evaluating checkpoints with the geometry head.
The learned geometry channels are currently more reliable than the bounded
Radon refinement path.

## May 2026 Geometry-Head Result

The cached spike now predicts five channels: one heatmap channel plus four
geometry channels for `cos(2 angle)`, `sin(2 angle)`, normalized length, and
normalized width. This keeps the OpenMMLab-free workflow intact while testing
whether the head can learn oriented box geometry directly.

Bounded run:

```text
Train cache: /private/tmp/argus_dinov3_geom_cache_train512_384
Val cache:   /private/tmp/argus_dinov3_geom_cache_val256_384
Checkpoint:  weights/run_plain_dinov3_geometry_cached_train512_384_e80/best.pt
Results:     results/plain_dinov3_geometry_cached_train512_384_e80/
```

Best validation Dice was `0.727`, essentially tied with the previous cached
heatmap-only run (`0.729`). Test metrics without geometry refinement:

| Threshold | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-------------|-----------|--------|----|----------|----------|-------------|
| 0.90 | 362 | 0.279 | 0.328 | 0.302 | 0.221 | 0.032 | 0.232 deg |
| 0.95 | 363 | 0.281 | 0.331 | 0.304 | 0.215 | 0.033 | 0.240 deg |
| 0.98 | 346 | 0.298 | 0.334 | 0.315 | 0.216 | 0.030 | 0.254 deg |
| 0.99 | 331 | 0.317 | 0.341 | 0.329 | 0.216 | 0.030 | 0.265 deg |

A refined pass at threshold `0.99` was worse (`F1=0.081`, `mAP@0.50=0.012`,
`angle error=7.529 deg`), so refinement should remain disabled for this model.

Conclusion: the plain PyTorch DINOv3 path can learn excellent orientation
without OpenMMLab, and the feature-cache workflow makes iteration practical.
However, the current heatmap-component detector is still not competitive enough
to replace the main detector path. The likely next useful spike is a direct
center/box head, such as center heatmap plus length, width, angle, and offset,
instead of deriving detections from connected components alone.

## May 2026 Direct Center/Box Result

The next spike replaced connected-component extraction with a sparse center
heatmap and direct box regression head. The cached target format now also
includes:

- `center_heatmap`: one positive cell per streak center
- `box_target`: `dx`, `dy`, `cos(2 angle)`, `sin(2 angle)`, normalized length,
  normalized width
- `box_mask`: positive cells used for regression loss

The first direct-box attempt inherited a one-patch minimum width and produced
better F1 but weaker IoU. Preserving thin streak widths fixed that regression.

Bounded thin-width run:

```text
Train cache: /private/tmp/argus_dinov3_box_thin_cache_train512_384
Val cache:   /private/tmp/argus_dinov3_box_thin_cache_val256_384
Checkpoint:  weights/run_plain_dinov3_box_thin_cached_train512_384_e80/best.pt
Results:     results/plain_dinov3_box_thin_cached_train512_384_e80/
```

Best validation score was `0.288`. Test metrics:

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

Compared with the geometry-head connected-component run, the direct center/box
head improves F1 (`0.422` vs `0.329`) and slightly improves best mAP@0.50
(`0.238` vs `0.216`) while keeping angle error near one degree. The remaining
gap is stricter localization: mAP@0.75 is still low, so the next iteration
should focus on more data, stronger center supervision, and possibly using
Gaussian center targets or a small matching/NMS layer rather than one-hot center
cells alone.

## May 2026 Gaussian Center/Box Result

This iteration kept the direct center/box detector, but changed two pieces:

- center targets are small Gaussian blobs instead of one-hot cells,
- the trainable head has separate center and box branches after a shared stem.

The evaluator now defaults to a `7x7` local-maximum filter for center-peak NMS,
which reduced duplicate peaks on this bounded run.

Bounded Gaussian run:

```text
Train cache: /private/tmp/argus_dinov3_box_gauss_cache_train512_384
Val cache:   /private/tmp/argus_dinov3_box_gauss_cache_val256_384
Checkpoint:  weights/run_plain_dinov3_box_gauss_cached_train512_384_e80/best.pt
Results:     results/plain_dinov3_box_gauss_cached_train512_384_e80/
```

Best validation score improved from `0.288` to `0.383`. Test metrics:

| Threshold | NMS | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-----|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 3 | 374 | 0.385 | 0.468 | 0.422 | 0.266 | 0.053 | 0.510 deg |
| 0.20 | 3 | 347 | 0.415 | 0.468 | 0.440 | 0.266 | 0.053 | 0.510 deg |
| 0.30 | 3 | 340 | 0.424 | 0.468 | 0.444 | 0.266 | 0.053 | 0.510 deg |
| 0.50 | 3 | 322 | 0.441 | 0.461 | 0.451 | 0.263 | 0.053 | 0.507 deg |
| 0.50 | 5 | 305 | 0.466 | 0.461 | 0.463 | 0.268 | 0.054 | 0.507 deg |
| 0.50 | 7 | 297 | 0.478 | 0.461 | 0.469 | 0.269 | 0.054 | 0.507 deg |
| 0.70 | 3 | 297 | 0.461 | 0.445 | 0.453 | 0.255 | 0.052 | 0.489 deg |
| 0.80 | 7 | 267 | 0.502 | 0.435 | 0.466 | 0.256 | 0.053 | 0.487 deg |
| 0.90 | 3 | 258 | 0.496 | 0.416 | 0.452 | 0.241 | 0.049 | 0.493 deg |

This is the strongest OpenMMLab-free spike so far: F1 improved from `0.422` to
`0.469`, best mAP@0.50 improved from `0.238` to `0.269`, and angle error
improved from about one degree to about half a degree. Strict IoU remains the
major weakness (`mAP@0.75` around `0.054`), which points to box length/center
precision rather than angle as the next bottleneck.

## May 2026 Full-Dataset Gaussian Run

The decisive full-dataset experiment used the same Gaussian center/box head on
the full `train.json` / `val.json` split:

```text
Train cache: /private/tmp/argus_dinov3_box_gauss_cache_train_full_384
Val cache:   /private/tmp/argus_dinov3_box_gauss_cache_val_full_384
Checkpoint:  weights/run_plain_dinov3_box_gauss_full_384_e100/best.pt
Results:     results/plain_dinov3_box_gauss_full_384_e100/
```

Data:

```text
Train: 3,023 images / 2,957 annotations
Val:     411 images /   386 annotations
Test:    308 images /   308 annotations
```

Best validation score improved to `0.506`. Test metrics with `7x7` center-peak
NMS:

| Threshold | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 319 | 0.511 | 0.529 | 0.520 | 0.325 | 0.149 | 0.318 deg |
| 0.20 | 314 | 0.513 | 0.523 | 0.518 | 0.322 | 0.149 | 0.317 deg |
| 0.30 | 308 | 0.520 | 0.520 | 0.520 | 0.320 | 0.149 | 0.318 deg |
| 0.50 | 296 | 0.530 | 0.510 | 0.520 | 0.315 | 0.146 | 0.320 deg |
| 0.70 | 289 | 0.533 | 0.500 | 0.516 | 0.310 | 0.146 | 0.316 deg |
| 0.80 | 286 | 0.535 | 0.497 | 0.515 | 0.308 | 0.145 | 0.315 deg |
| 0.90 | 281 | 0.537 | 0.490 | 0.513 | 0.304 | 0.145 | 0.318 deg |

Comparison against the current OpenMMLab-backed DINOv3 ViT-B result on the same
`test.json`:

| Method | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|--------|-----------|--------|----|----------|----------|-------------|
| Plain DINOv3 Gaussian Center/Box | 0.511-0.537 | 0.490-0.529 | 0.513-0.520 | 0.325 | 0.149 | 0.315-0.320 deg |
| MMDetection DINOv3 ViT-B COCO eval | n/a | n/a | n/a | 0.740 | 0.606 | n/a |
| Multi-method DINOv3 ViT-B detector | 0.093 | 0.893 | 0.168 | 0.755 | 0.594 | 0.000 deg |
| Unified Confidence Score | 0.299 | 0.721 | 0.423 | 0.406 | 0.318 | 0.018 deg |

Determination: removing OpenMMLab entirely is not justified yet. The plain
PyTorch path now beats the unified detector on F1 and has far cleaner
operability, but it is still less than half of the OpenMMLab DINOv3 ViT-B
mAP@0.50 and about one quarter of its mAP@0.75. The dependency-free model should
be promoted to a maintained experimental detector and Windows-friendly training
track, but OpenMMLab should remain the reference/production detector path until
plain PyTorch localization closes the mAP gap.

## Success Criteria

The spike is worth promoting if it:

- trains without OpenMMLab on the target workstation,
- produces comparable or better recall than the current DINOv3 detector,
- reduces false positives enough to improve precision or F1,
- outputs useful angle/endpoint geometry after heatmap component fitting,
- can be integrated into `inference/pipeline.py` as a new detector method.

## Expected Limitations

The spike keeps DINOv3 frozen and trains only the lightweight head. The dataset
now uses aspect-preserving letterboxing, and the geometry head fixes most angle
error. The direct center/box head improves localization over connected heatmap
components, but strict IoU quality is still the main weakness.

## May 2026 Thin-Width Full-Dataset Run

The previous full run showed useful detection quality, but diagnostics on
matched boxes showed that width/box-size error was still harming strict IoU.
This iteration removed the letterbox-space width floor from targets/evaluation
and increased the width-channel regression weight:

```text
Train cache: /private/tmp/argus_dinov3_box_gauss_thin_cache_train_full_384
Val cache:   /private/tmp/argus_dinov3_box_gauss_thin_cache_val_full_384
Checkpoint:  weights/run_plain_dinov3_box_gauss_thin_full_384_e100/best.pt
Results:     results/plain_dinov3_box_gauss_thin_full_384_e100/
```

Best validation score improved to `0.513`. Test metrics with `7x7`
center-peak NMS:

| Threshold | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 312 | 0.545 | 0.552 | 0.548 | 0.383 | 0.200 | 0.236 deg |
| 0.20 | 307 | 0.554 | 0.552 | 0.553 | 0.383 | 0.200 | 0.236 deg |
| 0.30 | 304 | 0.553 | 0.545 | 0.549 | 0.379 | 0.198 | 0.237 deg |
| 0.50 | 299 | 0.559 | 0.542 | 0.550 | 0.378 | 0.198 | 0.232 deg |
| 0.70 | 289 | 0.564 | 0.529 | 0.546 | 0.370 | 0.196 | 0.229 deg |
| 0.80 | 282 | 0.571 | 0.523 | 0.546 | 0.367 | 0.194 | 0.228 deg |
| 0.90 | 276 | 0.565 | 0.506 | 0.534 | 0.357 | 0.192 | 0.226 deg |

Additional `3x3` NMS checks did not improve the result: at threshold `0.50`,
F1 was `0.542`, mAP@0.50 was `0.376`, and mAP@0.75 was `0.197`; at threshold
`0.80`, F1 was `0.541`, mAP@0.50 was `0.365`, and mAP@0.75 was `0.193`.

Best-threshold diagnostics at threshold `0.20`:

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

This is the strongest plain PyTorch result so far. Compared with the previous
full Gaussian run, F1 moved from `0.520` to `0.553`, mAP@0.50 from `0.325` to
`0.383`, mAP@0.75 from `0.149` to `0.200`, and angle error from about `0.32`
degrees to about `0.24` degrees.

Updated determination: the OpenMMLab-free path is improving quickly and now has
the strongest Windows-friendly training story in the repo, but it still should
not replace OpenMMLab as the reference detector. It is still well behind the
OpenMMLab-backed DINOv3 ViT-B COCO-eval result (`mAP@0.50=0.740`,
`mAP@0.75=0.606`) and below the unified confidence score on strict localization
(`mAP@0.75=0.318`). The next development target is no longer angle; it is
center localization and strict box overlap.

Practical note: full-dataset cached DINOv3 features require substantial scratch
space. On this run, stale `/private/tmp/argus_dinov3_*` caches filled the
filesystem and had to be removed before the full train/val cache could complete.

## May 2026 Decision-Point Iterations

After the thin-width full run, two focused localization iterations tested
whether the plain PyTorch DINOv3 path could close the mAP gap enough to justify
continued replacement work.

### 512px Full-Dataset Run

This run kept the same thin-width Gaussian center/box formulation but increased
input size from `384` to `512`:

```text
Checkpoint: weights/run_plain_dinov3_box_gauss_thin_full_512_e100/best.pt
Results:    results/plain_dinov3_box_gauss_thin_full_512_e100/
```

Best test metrics:

| Threshold | NMS | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-----|-------------|-----------|--------|----|----------|----------|-------------|
| 0.10 | 7 | 313 | 0.537 | 0.545 | 0.541 | 0.342 | 0.212 | 0.361 deg |

The 512px run improved median center error from about `53 px` to `44 px` and
raised matched detections at IoU `0.75` from `121` to `128`, but F1 and
mAP@0.50 regressed. Resolution alone is not enough.

### Support-Offset Full-Dataset Run

This run changed the target formulation: box regression is trained over the
high-confidence Gaussian neighborhood instead of only the nearest center cell,
and decoded center offsets now allow `+-2.0` patch cells.

```text
Checkpoint: weights/run_plain_dinov3_box_support_full_384_e100/best.pt
Results:    results/plain_dinov3_box_support_full_384_e100/
```

Best test metrics:

| Threshold | NMS | Predictions | Precision | Recall | F1 | mAP@0.50 | mAP@0.75 | Angle Error |
|-----------|-----|-------------|-----------|--------|----|----------|----------|-------------|
| 0.20 | 7 | 303 | 0.568 | 0.558 | 0.563 | 0.330 | 0.185 | 0.149 deg |

The support-offset run improved F1 and angle error, but it regressed both
strict localization metrics. It did not move the plain path toward the
replacement threshold.

### Decision

The plain PyTorch DINOv3 path should be kept as a useful experimental and
Windows-friendly training track, but it should not continue as the primary
OpenMMLab replacement effort in its current architecture.

The decision threshold was roughly `mAP@0.50 >= 0.50` and `mAP@0.75 >= 0.30`
to justify more replacement-focused investment. The best plain result remains
the thin-width 384px run at `F1=0.553`, `mAP@0.50=0.383`, and
`mAP@0.75=0.200`. The next two focused localization iterations did not close
that gap:

| Iteration | F1 | mAP@0.50 | mAP@0.75 | Replacement Signal |
|-----------|----|----------|----------|--------------------|
| Thin-width 384px | 0.553 | 0.383 | 0.200 | Best plain baseline |
| Thin-width 512px | 0.541 | 0.342 | 0.212 | Slight mAP@0.75 gain, overall regression |
| Support-offset 384px | 0.563 | 0.332 | 0.187 | F1/angle gain, localization regression |

Recommendation: do not remove OpenMMLab. Keep this code only if its simpler
training story is valuable for future experiments, but the replacement path
should pivot to a stronger detection formulation such as a DETR/RT-DETR-style
plain PyTorch detector, a YOLO-OBB baseline, or another architecture with a
proper matching/object-detection loss instead of continuing to tune this
heatmap-plus-local-box head.

### Archive Status

This spike is closed. No further work is planned on this specific
heatmap-plus-local-box architecture. Generated checkpoints, scratch feature
caches, Python cache directories, and bulky per-image prediction dumps were
removed after recording the aggregate metrics and summaries.
