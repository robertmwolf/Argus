# DINOv3 ViT-S Training Handoff

Date: 2026-05-28

## Goal

Train two new ViT-S models and compare each against its existing ViT-B baseline:

| New model | Architecture | Baseline to beat |
|-----------|-------------|-----------------|
| `dinov3_vits_run3` | MMDetection DINO ViT-S, OBB output | `dinov3_vitb_run3` |
| `dinov3_orientation_centerline_vits` | Orientation-centerline heatmap ViT-S, line-segment output | `dinov3_orientation_centerline_vitb` |

The two model families produce **different outputs** (OBB boxes vs line segments) and are
evaluated separately. Do not compare them to each other using the same metric.

Do not restart or retrain ViT-B for this comparison unless the existing baseline
weights are missing or invalid.

## Important Guardrail

Training must not plate solve. Before any long run, smoke-test the training path
and check logs for accidental `ASTAP`, `plate`, `solve`, or `WCS` activity.

## Expected Local Inputs

- Train annotations: `data/annotations/all_train_nodm.json`
- Validation annotations: `data/annotations/val.json`
- ViT-S weights: `weights/dinov3_vits16_lvd1689m.pth`
- ViT-B Run 3 baseline weights: `weights/run3_cold_nodm/best.pth`
- ViT-B orientation-centerline baseline weights:
  `weights/run_dinov3_vitb_orientation_centerline_input512_catchment/best.pt`

## Step 1: Preflight

```bash
test -f data/annotations/all_train_nodm.json
test -f data/annotations/val.json
test -f weights/dinov3_vits16_lvd1689m.pth
test -f weights/dinov3_vitb16_lvd1689m.pth
```

Check for stale training jobs:

```bash
ps -ef | rg "training.train_dino|train_dinov3|caffeinate"
```

## Step 2: Smoke Tests

### 2a: MMDetection DINO ViT-S

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
python -m training.train_dino \
  --config models/dino/streak_dinov3_vits_400px_run3.py \
  --work-dir weights/run3_cold_nodm_vits_smoke \
  --smoke-test
```

### 2b: Orientation-centerline ViT-S

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
python training/train_dinov3_orientation_centerline.py \
  --train-annotations data/annotations/all_train_nodm.json \
  --val-annotations data/annotations/val.json \
  --work-dir weights/run_dinov3_vits_orientation_centerline_1024_smoke \
  --weights weights/dinov3_vits16_lvd1689m.pth \
  --model-size small \
  --image-size 1024 \
  --epochs 1 \
  --batch-size 1 \
  --workers 0 \
  --positive-train-tiles 4 \
  --negative-train-tiles 4
```

Both smoke tests should complete without plate solving.

## Step 3a: Train MMDetection DINO ViT-S

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
USE_DEV_SUBSET=false \
TRAIN_ANN_FILE=annotations/all_train_nodm.json \
VAL_ANN_FILE=annotations/val.json \
ARGUS_NORM=zscore \
caffeinate -i python -m training.train_dino \
  --config models/dino/streak_dinov3_vits_400px_run3.py \
  --work-dir weights/run3_cold_nodm_vits \
  --val-interval 1 \
  --checkpoint-interval 1
```

Output weights: `weights/run3_cold_nodm_vits/best.pth`

## Step 3b: Train Orientation-Centerline ViT-S

**Use `--image-size 1024`, not 512.** The existing ViT-B baseline was trained at
512px; analysis showed that 512px fails entirely on large streaks (>1500 px native)
because the per-patch angle error × streak length exceeds the 6 px evaluation
tolerance. At 1024px the angle resolution halves and segment traces are longer.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
caffeinate -i python training/train_dinov3_orientation_centerline.py \
  --train-annotations data/annotations/all_train_nodm.json \
  --val-annotations data/annotations/val.json \
  --work-dir weights/run_dinov3_vits_orientation_centerline_1024 \
  --weights weights/dinov3_vits16_lvd1689m.pth \
  --model-size small \
  --tile-size 2560 \
  --image-size 1024 \
  --positive-train-tiles 1236 \
  --negative-train-tiles 1400 \
  --orientation-bins 18 \
  --decoder-channels 192 \
  --last-layers 4 \
  --centerline-width 2.0 \
  --centerline-sigma 1.4 \
  --catchment-width 14.0 \
  --catchment-sigma 6.0 \
  --neighbor-bin-weight 0.35 \
  --second-neighbor-weight 0.0 \
  --epochs 10 \
  --batch-size 1 \
  --lr 5e-5 \
  --min-lr 1e-5 \
  --weight-decay 1e-4 \
  --pos-weight 60 \
  --dice-weight 1.0 \
  --bce-weight 0.10 \
  --orientation-ce-weight 0.20 \
  --manual-positive-weight 3.0 \
  --catchment-loss-weight 0.35 \
  --catchment-pos-weight 20 \
  --catchment-dice-weight 1.0 \
  --catchment-bce-weight 0.20 \
  --image-loss-weight 0.25 \
  --workers 0 \
  --seed 20260524 \
  --preserve-image-bit-depth \
  --log-interval 200
```

Output weights: `weights/run_dinov3_vits_orientation_centerline_1024/best.pt`

**Resolution caveat**: the ViT-B baseline was trained at 512px so ViT-S (1024px)
vs ViT-B (512px) is not resolution-matched. If you want a fair resolution
comparison, train `--model-size base --image-size 1024` as an additional
reference run.

## Step 4: Evaluate

### 4a: MMDetection DINO — ViT-S vs ViT-B

Generate predictions for `dinov3_vits_run3` and `dinov3_vitb_run3` with the same
annotation file, IoU thresholds, confidence policy, and postprocessing caps.
The benchmark must not allow one DINO model to globally suppress the other.

```bash
python -m eval.benchmark \
  --annotations data/annotations/test.json \
  --vits-predictions results/vits_run3/predictions.json \
  --vitb-run3-predictions results/vitb_run3/predictions.json \
  --output results/ab_vits_vs_vitb/benchmark.json
```

### 4b: Orientation-centerline — ViT-S vs ViT-B

Use `propose_dinov3_centerline_segments.py` and `eval/line_metrics.py`.
Do **not** use `eval.benchmark` — the output format is line segments, not OBBs.

```bash
python scripts/propose_dinov3_centerline_segments.py \
  --annotations data/annotations/test.json \
  --weights weights/run_dinov3_vits_orientation_centerline_1024/best.pt \
  --model-size small \
  --image-size 1024 \
  --threshold 0.85 \
  --min-line-support 0.50 \
  --max-components-per-image 2 \
  --output results/orientation_centerline_vits/proposals.json

python eval/line_metrics.py \
  --annotations data/annotations/test.json \
  --predictions results/orientation_centerline_vits/proposals.json \
  --output results/orientation_centerline_vits/metrics.json
```

Compare against the ViT-B baseline at the same thresholds and gates.

**Per-size breakdown**: report recall and precision by streak length using the
`streak_length_px` field in GT annotations:

| Class  | Native length |
|--------|---------------|
| Small  | < 500 px      |
| Medium | 500–1500 px   |
| Large  | > 1500 px     |

Known ViT-B 512px baselines for orientation-centerline:
- Overall: F1=0.218, precision=0.349, recall=0.158 (TP=61, FP=114).
- Large-streak recall near 0 (fundamental resolution limit at 512px).

**LSR gate note**: the lsr≥0.50 gate hurts recall on large streaks (their
heatmap activations are diffuse blobs, lsr≈0.05–0.33). Evaluate with both
`--min-line-support 0.50` (default, optimal for small/medium) and
`--min-line-support 0.0` to isolate the gate's contribution on large streaks.

## Known State From 2026-05-28

Code changes already committed to this branch and main:

- `models/plain_dinov3/streak_heatmap.py`: `"small"` (ViT-S, embed_dim=384,
  `vit_small`) added to `_MODEL_CONFIGS`.
- `training/train_dinov3_orientation_centerline.py`: `--model-size small`
  added to choices.
- `training/dinov3_orientation_centerline_dataset.py`: FITS loading switched
  from min/max scaling to z-score (3σ clip → [0,1] float32).
- `inference/heatmap_detector.py`: optimal inference defaults applied
  (`HEATMAP_SEGMENT_THRESHOLD=0.85`, `HEATMAP_MIN_LINE_SUPPORT=0.50`,
  `HEATMAP_MAX_COMPONENTS=2`).

Verification completed in a prior worktree:

- `tests/test_model_configs.py`
- `tests/test_train_dino.py`
- `tests/test_fits_loader.py::TestASTAPPlateSolver::test_load_wcs_false_skips_astap_even_with_header_hints`
- MMDetection ViT-S smoke test passed.
- Orientation-centerline ViT-S smoke test passed (`val_dice=0.393` on tiny run).
