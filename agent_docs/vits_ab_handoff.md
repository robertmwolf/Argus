# DINOv3 ViT-S A/B Handoff

Date: 2026-05-28

## Goal

Compare apples-to-apples ViT-S replacements against the two relevant existing
ViT-B baselines:

- `dinov3_vitb_run3`: MMDetection DINO Run 3, frozen DINOv3 ViT-B backbone.
- `dinov3_heatmap_centerline_vitb`: plain PyTorch DINOv3 heatmap/centerline
  detector, frozen ViT-B encoder.

There are two new ViT-S weight tracks to produce:

1. `dinov3_vits_run3`: MMDetection DINO Run 3 settings with a frozen DINOv3
   ViT-S backbone.
2. `dinov3_heatmap_centerline_vits`: plain PyTorch heatmap/centerline head
   with a frozen DINOv3 ViT-S encoder.

Do not restart or retrain ViT-B for this comparison unless the existing
baseline weights are missing or invalid.

## Important Guardrail

Training must not plate solve. The training FITS loaders should use
`FITSLoader(load_wcs=False)` so data loading skips FITS WCS parsing, sidecar
WCS, and ASTAP plate solving. Keep inference WCS behavior unchanged.

Before any long run, smoke-test the training path and check logs for accidental
`ASTAP`, `plate`, `solve`, or `WCS` activity.

## Models in Scope

There are three model families in this comparison.

**OBB detectors (primary comparison):**

- `dinov3_vitb_run3` — MMDetection DINO ViT-B, produces OBBs.
- `dinov3_vits_run3` — MMDetection DINO ViT-S, produces OBBs.
- `dinov3_heatmap_centerline_vitb` — `train_dinov3_box_cached` ViT-B, produces
  OBBs via center-heatmap + regression.
- `dinov3_heatmap_centerline_vits` — `train_dinov3_box_cached` ViT-S, produces
  OBBs via center-heatmap + regression.

**Orientation-centerline detector (supplemental / optional track):**

- `dinov3_orientation_centerline_vitb` — `train_dinov3_orientation_centerline`
  ViT-B, produces line segments (not OBBs). Existing baseline:
  `weights/run_dinov3_vitb_orientation_centerline_input512_catchment/best.pt`.
- `dinov3_orientation_centerline_vits` — same architecture with ViT-S backbone
  (optional; see Step 3c). Produces line segments, evaluated separately with
  `eval/line_metrics.py`.

The orientation-centerline family outputs are **not directly comparable** to the
OBB family outputs. Evaluate them separately with `compare_heatmap_centerline_to_obb.py`
and `eval/line_metrics.py`, not with `eval.benchmark`.

## Expected Local Inputs

Use the DM-free Run 3 dataset unless superseded intentionally:

- Train annotations: `data/annotations/all_train_nodm.json`
- Validation annotations: `data/annotations/val.json`
- ViT-S weights: `weights/dinov3_vits16_lvd1689m.pth`
- ViT-B Run 3 baseline weights: `weights/run3_cold_nodm/best.pth`
- ViT-B heatmap baseline weights:
  `weights/run_plain_dinov3_box_gauss_thin_full_384_e100/best.pt`
- ViT-B orientation-centerline baseline weights:
  `weights/run_dinov3_vitb_orientation_centerline_input512_catchment/best.pt`

## Step 1: Preflight

Confirm the data and weights exist:

```bash
test -f data/annotations/all_train_nodm.json
test -f data/annotations/val.json
test -f weights/dinov3_vits16_lvd1689m.pth
test -f weights/dinov3_vitb16_lvd1689m.pth
```

Also check for stale training jobs:

```bash
ps -ef | rg "training.train_dino|train_dinov3|caffeinate"
```

If a ViT-B Run 3 job is already running, treat it as an unrelated existing job.
Do not queue ViT-S behind it by default; either stop it explicitly or wait until
the machine is free.

## Step 2: Smoke Tests

Run the MMDetection ViT-S smoke test:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
python -m training.train_dino \
  --config models/dino/streak_dinov3_vits_400px_run3.py \
  --work-dir weights/run3_cold_nodm_vits_smoke \
  --smoke-test
```

Run a small plain heatmap ViT-S smoke test:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
python -m training.train_dinov3_heatmap \
  --train-annotations data/annotations/all_train_nodm.json \
  --val-annotations data/annotations/val.json \
  --weights weights/dinov3_vits16_lvd1689m.pth \
  --model-size small \
  --work-dir weights/run_plain_dinov3_heatmap_vits_smoke \
  --image-size 384 \
  --smoke-test
```

Both smoke tests should complete without plate solving.

## Step 3: Train ViT-S Weights

Train the MMDetection DINO ViT-S detector:

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

For the heatmap/centerline ViT-S track, first cache ViT-S features, then train
the center/box head:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
python scripts/cache_dinov3_heatmap_features.py \
  --annotations data/annotations/all_train_nodm.json \
  --output-dir data/cache/plain_dinov3_vits/train_384 \
  --weights weights/dinov3_vits16_lvd1689m.pth \
  --model-size small \
  --image-size 384

PYTORCH_ENABLE_MPS_FALLBACK=1 \
ARGUS_NORM=zscore \
python scripts/cache_dinov3_heatmap_features.py \
  --annotations data/annotations/val.json \
  --output-dir data/cache/plain_dinov3_vits/val_384 \
  --weights weights/dinov3_vits16_lvd1689m.pth \
  --model-size small \
  --image-size 384

python -m training.train_dinov3_box_cached \
  --train-cache data/cache/plain_dinov3_vits/train_384 \
  --val-cache data/cache/plain_dinov3_vits/val_384 \
  --work-dir weights/run_plain_dinov3_box_gauss_thin_vits_384_e100 \
  --epochs 100
```

Use the same center/box head settings as the ViT-B
`plain_dinov3_box_gauss_thin_full_384_e100` baseline unless there is a specific
reason to change them.

## Step 3c: (Optional) Train ViT-S Orientation-Centerline

This track trains the orientation-binned centerline model (`train_dinov3_orientation_centerline`)
with a ViT-S backbone. It is independent of the OBB tracks and evaluated on
different metrics (line-segment F1 with `eval/line_metrics.py`, not `eval.benchmark`).

### Why and when to add this track

Analysis of the existing ViT-B orientation-centerline model (512px input,
`best.pt` above) against the ViT-B box-cached baseline shows:

- Heatmap F1 = 0.218 vs box F1 = 0.243 overall.
- **Heatmap wins on small (<500 px) and medium (500–1500 px) streaks.**
- **Heatmap fails entirely on large (>1500 px) streaks.** Root cause: at 512px
  input the per-patch angle error (~10°) combined with long streak length
  (median 3378 px native) creates a 500+ px perpendicular drift that exceeds
  the 6 px evaluation tolerance. Short heatmap segments (median ~400 px) also
  cannot cover 1500–4000 px GT streaks.
- The LSR gate (≥0.50) further rejects most large-streak candidates (they are
  diffuse blobs, lsr = 0.05–0.33). Dropping the gate recovers only 3 TPs out
  of 178 large-streak GT items — the fundamental limit is resolution and
  segment length, not the gate.

**Do not run this track at 512px.** Use `--image-size 1024`. The trainer
already defaults to 1024px; the existing ViT-B script used 512px only as a
spike setting. ViT-S at 1024px gives roughly the same spatial resolution as
ViT-B at 512px but with a lighter backbone — a reasonable comparison point.

### Code prerequisites (already applied in this branch)

- `models/plain_dinov3/streak_heatmap.py`: added `"small"` entry to
  `_MODEL_CONFIGS` (embed_dim=384, `vit_small`).
- `training/train_dinov3_orientation_centerline.py`: `--model-size` choices
  now include `"small"`.

### Smoke test

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

Check logs: no `ASTAP`, `plate`, `solve`, or `WCS` activity.

### Full training run

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

The ViT-B orientation-centerline baseline to compare against is
`weights/run_dinov3_vitb_orientation_centerline_input512_catchment/best.pt`.
Be aware that the ViT-B baseline was trained at **512px** and this ViT-S run is
at **1024px** — they are not perfectly apples-to-apples on resolution. If you
want a true resolution-matched baseline, train `--model-size base --image-size
1024` first.

## Step 4: Evaluate Apples-to-Apples

### 4a: OBB track benchmark

Generate predictions for all four OBB models:

- `dinov3_vits_run3`
- `dinov3_vitb_run3`
- `dinov3_heatmap_centerline_vits`
- `dinov3_heatmap_centerline_vitb`

Then compare with the same annotation file, IoU thresholds, confidence policy,
and postprocessing caps. The benchmark should not allow one DINO model to
globally suppress another before metrics are computed.

```bash
python -m eval.benchmark \
  --annotations data/annotations/test.json \
  --vits-predictions results/vits_run3/predictions.json \
  --vitb-run3-predictions results/vitb_run3/predictions.json \
  --heatmap-centerline-predictions results/plain_dinov3_box_gauss_thin_full_384_e100/predictions.json \
  --output results/ab_vits_vs_vitb_vs_heatmap/benchmark.json
```

If adding the ViT-S heatmap method to the benchmark CLI, keep its method ID
separate from the ViT-B heatmap method.

### 4b: Per-size breakdown (all models)

Break down recall and precision by native-pixel streak length using the
`streak_length_px` field in the GT annotations. Size boundaries:

| Class  | Native length |
|--------|---------------|
| Small  | < 500 px      |
| Medium | 500–1500 px   |
| Large  | > 1500 px     |

This breakdown is critical for interpreting the results:
- OBB models tend to have uniform recall across sizes.
- The orientation-centerline model currently **fails entirely on large streaks**
  at 512px input. At 1024px the large-streak recall should improve meaningfully.

### 4c: Orientation-centerline track evaluation (if Step 3c was run)

Use `compare_heatmap_centerline_to_obb.py` and `eval/line_metrics.py` to
evaluate the orientation-centerline ViT-S model. Do **not** pass its predictions
to `eval.benchmark` (different output format).

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

Compare this against the ViT-B baseline (`proposals.json` from
`run_dinov3_vitb_orientation_centerline_input512_catchment/best.pt`) using the
same thresholds and gates. Report overall F1 plus the per-size breakdown.

Key known baselines from the ViT-B 512px analysis:
- Overall: F1=0.218, precision=0.349, recall=0.158 (TP=61, FP=114).
- Large-streak recall is near 0 at 512px (only 2 of 178 large-streak GT items
  match even without the LSR gate).
- Expected improvement with 1024px: better angle resolution and longer traced
  segments should raise large-streak recall substantially.

**LSR gate note for large streaks**: the LSR gate (≥0.50) rejects large-streak
candidates because their heatmap activations are diffuse blobs (lsr ≈ 0.05–0.33).
If large-streak recall is the priority, consider evaluating with
`--min-line-support 0.0` alongside the default `--min-line-support 0.50` to
isolate the gate's contribution. Do not change the default (0.50 is optimal for
precision on small/medium streaks).

## Known State From 2026-05-28 Worktree

The detached Codex worktree had partial implementation support for:

- `models/dino/streak_dinov3_vits_400px_run3.py`
- ViT-S support in the MMDetection DINO adapter.
- `dinov3_vits_run3` model registry and benchmark wiring.
- Per-model DINO postprocess/NMS caps.
- `FITSLoader(load_wcs=False)` for training loaders.
- Plain DINOv3 heatmap `model_size=small` support.

Targeted verification completed there:

- `tests/test_model_configs.py`
- `tests/test_train_dino.py`
- `tests/test_fits_loader.py::TestASTAPPlateSolver::test_load_wcs_false_skips_astap_even_with_header_hints`
- MMDetection ViT-S smoke test.
- Plain heatmap ViT-S smoke test (`val_dice=0.393` on the tiny smoke run).

The `claude/jolly-visvesvaraya-62540a` Claude Code worktree added:

- `models/plain_dinov3/streak_heatmap.py`: `"small"` entry in `_MODEL_CONFIGS`
  (ViT-S, embed_dim=384, `vit_small`) — enables Step 3c orientation-centerline
  training.
- `training/train_dinov3_orientation_centerline.py`: `--model-size small` added
  to choices.

Analysis of existing ViT-B orientation-centerline model (512px input) that
informed the Step 3c and Step 4c additions:

- ViT-B heatmap vs ViT-B box at the per-size level: heatmap wins small/medium
  but fails entirely on large (>1500 px native). Resolution is the bottleneck.
- `line_support_ratio` (LSR) is anti-correlated with image-level GT match but
  positively correlated with segment-level strict TPs — raising threshold to
  0.85 and applying lsr≥0.50 improved F1 from 0.116 to 0.218.
- `radon_snr` is not discriminative (clusters at 1.0–1.2 for all proposals).
- Optimal inference settings now default in `inference/heatmap_detector.py`:
  `HEATMAP_SEGMENT_THRESHOLD=0.85`, `HEATMAP_MIN_LINE_SUPPORT=0.50`,
  `HEATMAP_MAX_COMPONENTS=2`.

