# ARGUS

ARGUS detects satellite streaks in astronomical FITS images, maps them through
WCS, and cross-identifies them against a local TLE catalog using SGP4 propagation.
Every streak is a line segment defined by two image-space endpoints (`x1`, `y1`,
`x2`, `y2`).

## Architecture

Detection uses a frozen DINOv3 ViT-S/16 backbone with a small convolutional head
trained to predict a one-channel streak centerline heatmap. Connected heatmap
components become endpoint segments. Tiled inference remaps segments to full-frame
coordinates, suppresses duplicates, and stitches collinear fragments.

The production model is `vits_v9_asl_cldice` — trained with ASL + clDice loss on
400 px tiles with z-score normalisation. See
[`docs/loss_ablation_v9_v10_postmortem.md`](docs/loss_ablation_v9_v10_postmortem.md)
for the methodology behind this choice.

```
inference/          — FITS loading, heatmap detection, post-processing, WCS, cross-ID
models/plain_dinov3/— DINOv3 ViT heatmap model definition
training/           — dataset and cached-feature trainers
eval/geometry_metrics.py — canonical segment evaluator
scripts/            — dataset preparation, caching, evaluation, operations
api/, db/, frontend/— FastAPI backend, SQLite persistence, React frontend
src/                — astrometry and satellite matching components
```

## Local development

Create the Python environment from the repository root:

```bash
conda env create -f environment.yml
conda activate satid
python -m pip check
```

Copy `.env.example` to `.env` and fill in Space-Track credentials before starting
the API. **Never commit `.env`.**

Download the published model weights from
[`lonewolfman22/argus-weights`](https://huggingface.co/lonewolfman22/argus-weights):

```bash
python scripts/sync_hf.py --download --weights-only --weights-dir weights
```

The repository is public, so setup normally requires no Hugging Face token. If
authentication is required, run `hf auth login` or export `HF_TOKEN` first.
The download includes the DINOv3 backbone weights and the published `run15_vits`
and `run17_vitb` heads. It does not currently include the production
`weights/vits_v9_asl_cldice/best.pt` head; supply that checkpoint separately and
set `VITS_V9_HEATMAP_CHECKPOINT` in `.env` when production-v9 inference is needed.

Weights are stored under the gitignored `weights/` directory and must never be
committed.

Start the API:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

In a second terminal (Node.js 20.19+ or 22.12+):

```bash
cd frontend && npm ci && npm run dev
```

## Data setup

Training data and annotations live outside the repository on an external drive.
Set `ARGUS_DATA_ROOT` to point at the durable dataset tree before running any
training or evaluation script:

```bash
export ARGUS_DATA_ROOT=/Volumes/External/TrainingData
```

The canonical evaluation annotation is
`$ARGUS_DATA_ROOT/annotations/val_balanced_v1.json`.

See [`agent_docs/datasets.md`](agent_docs/datasets.md) for the full data layout
and staging contract.

## Tests

```bash
python -m pytest tests/ -q
```

Tests run offline and do not require production checkpoints or Space-Track
credentials.

## Training a new model

```bash
# 1. Build the window dataset
python scripts/build_atwood_window_dataset.py \
  --version <N> --source "$ARGUS_DATA_ROOT/annotations/all_train_run17_merged.json" \
  --val-frac 0.08 --neg-frac 0.42 --bg-per-frame 3 --seed 42

# 2. Cache ViT-S features (backbone forward pass runs once, not every epoch)
python scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ARGUS_DATA_ROOT/train_atwood_synth_window_<N>/annotation.json" \
  --output-dir /Volumes/External/argus_caches/vits_window_<N>/train \
  --backbone vit --model-size small --weights weights/dinov3_vits16_lvd1689m.pth \
  --image-size 518 --native-tile-size 400 --tile-overlap 0.0 --norm-mode none

# 3. Train (ASL + clDice — the winning loss combination)
python training/train_dinov3_heatmap_cached.py \
  --train-cache /Volumes/External/argus_caches/vits_window_<N>/train \
  --val-cache   /Volumes/External/argus_caches/vits_window_<N>/val \
  --work-dir weights/vits_window_<N> \
  --epochs 40 --lr 1e-3 --batch-size 32 --hidden-channels 256 \
  --lr-scheduler cosine --early-stopping-patience 10 \
  --loss-mode asl_cldice \
  --asl-gamma-neg 4.0 --asl-gamma-pos 0.0 --asl-margin 0.05 --cldice-iters 3

# 4. Evaluate
python -m eval.geometry_metrics \
  --predictions results/<tag>/pf85/predictions_t070.json \
  --annotations "$ARGUS_DATA_ROOT/annotations/val_balanced_v1.json" \
  --output results/<tag>/pf85/geometry_eval.json
```

See [`agent_docs/ml_pipeline.md`](agent_docs/ml_pipeline.md) for the full
training guide and [`docs/loss_ablation_v9_v10_postmortem.md`](docs/loss_ablation_v9_v10_postmortem.md)
for the loss-function rationale.

## TLE catalog

Runtime identification uses only locally stored catalog data. Bootstrap before
processing historical images:

```bash
# Recent coverage (last 90 days) — required for live pipeline use:
export ARGUS_ENV=production
python scripts/bootstrap_recent_tles.py

# Prior years (annual zip bundles):
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2025
```

See [`agent_docs/spacetrack.md`](agent_docs/spacetrack.md) for the full TLE
management guide including rate limits and Space-Track API policy.

## Evaluating results

```bash
python scripts/compare_geometry_evals.py       # plain table across all models
python scripts/compare_geometry_evals.py --md  # GitHub-flavored Markdown
```

Canonical parameters: threshold=0.70, peak\_floor=0.85, `val_balanced_v1.json`.
