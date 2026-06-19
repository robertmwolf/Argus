# ARGUS Developer Guide

Setup, operations, and development reference for ARGUS. For project overview and
results see [`README.md`](README.md).

---

## Environment Setup

Requires Python 3.10+ and Conda. From the repository root:

```bash
conda env create -f environment.yml
conda activate satid
python -m pip check
```

The Python interpreter path is `/Users/robert/miniconda3/envs/satid/bin/python`.

Copy `.env.example` to `.env` and fill in credentials before starting the API:

```bash
cp .env.example .env
# Edit .env: Space-Track credentials, model checkpoint paths, ARGUS_DATA_ROOT
```

**Never commit `.env`.**

---

## Model Weights

Weights are stored under the gitignored `weights/` directory. Download the
published bundle from [`lonewolfman22/argus-weights`](https://huggingface.co/lonewolfman22/argus-weights):

```bash
python scripts/sync_hf.py --download --weights-only --weights-dir weights
```

The public repository requires no Hugging Face token under normal conditions.
If authentication is required:

```bash
hf auth login       # interactive
# or
export HF_TOKEN=<token>
```

The download includes:
- `weights/dinov3_vits16_lvd1689m.pth` — frozen ViT-S/16 backbone
- `weights/dinov3_vitb16_lvd1689m.pth` — frozen ViT-B/16 backbone
- `weights/run15_vits/` and `weights/run17_vitb/` — older published heads

**The production head `weights/vits_v9_asl_cldice/best.pt` is not in the public
bundle.** Supply it separately and set in `.env`:

```
VITS_V9_HEATMAP_CHECKPOINT=weights/vits_v9_asl_cldice/best.pt
```

Never commit weights to the repository.

---

## Raw Image Data

> **Note:** A public raw-data source is not yet configured. This section will be
> updated when a source (e.g. a Hugging Face dataset repository or a hosted
> archive) is available.

Training data is FITS observations from Atwood Observatory organized as:

```
$ARGUS_DATA_ROOT/
  Img_YYYYMMDD_Atwood/   — raw FITS frames per capture night
  annotations/           — COCO-format JSON annotation files
```

Set `ARGUS_DATA_ROOT` in `.env` or export it before running any training or
evaluation script:

```bash
export ARGUS_DATA_ROOT=/Volumes/External/TrainingData
```

The canonical validation annotation is:

```
$ARGUS_DATA_ROOT/annotations/val_balanced_v1.json
```

See [`agent_docs/datasets.md`](agent_docs/datasets.md) for the full data layout,
naming conventions, and file-staging contract.

---

## API and UI

Start the FastAPI backend:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Do not `source .env` before running uvicorn — the shell will mangle JSON values
in `ARGUS_MODEL_CONFIGS`. Use `python-dotenv` loading (already wired in
`api/main.py`).

In a second terminal (Node.js 20.19+ or 22.12+):

```bash
cd frontend && npm ci && npm run dev
```

The UI is available at `http://localhost:5173`.

See [`agent_docs/tle_confidence.md`](agent_docs/tle_confidence.md) for the TLE
cross-identification confidence formula, `/api/result` fields, units, sign
conventions, and how the factor breakdown is presented in the UI.

---

## Tests

```bash
python -m pytest tests/ -q
```

Tests run offline and do not require production checkpoints or Space-Track
credentials. Mock network access and heavyweight model loading when adding new
tests.

---

## Annotating New Data

New FITS frames from Atwood Observatory arrive as `Img_YYYYMMDD_Atwood/` directories
under `ARGUS_DATA_ROOT`.

1. **Review frames in Frigate.** Mark frames as positive (streak present) or
   negative (clear sky).

2. **Annotate positives with oriented bounding boxes.** Annotations are
   automatically converted to endpoint pairs (`x1, y1, x2, y2`) at ingestion.

3. **Merge new annotations** into the master training pool:

   ```bash
   # Merge new annotation file into all_train_run17_merged.json
   # (use the merge script or manually concatenate under the same ARGUS_DATA_ROOT)
   ```

4. **Rebuild the dataset** (see Training below).

Keep annotation `file_name` values relative to `ARGUS_DATA_ROOT`. Legacy absolute
paths are supported only when they are beneath the data root.

---

## Training a New Model

### 1. Build a window dataset

```bash
python scripts/build_atwood_window_dataset.py \
  --version <N> \
  --source "$ARGUS_DATA_ROOT/annotations/all_train_run17_merged.json" \
  --val-frac 0.08 --neg-frac 0.42 --bg-per-frame 3 --seed 42
```

### 2. Stage source files to local scratch (optional, for fast I/O)

```bash
python scripts/stage_dataset_files.py \
  "$ARGUS_DATA_ROOT/train_atwood_synth_window_<N>/annotation.json" \
  "$ARGUS_DATA_ROOT/train_atwood_synth_window_<N>/val_annotation.json" \
  --output-dir /tmp/argus_scratch
export ARGUS_SCRATCH_ROOT=/tmp/argus_scratch
```

### 3. Cache ViT-S features (run once per dataset)

Feature caches go to the external drive and are deleted after training completes.

```bash
python scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ARGUS_DATA_ROOT/train_atwood_synth_window_<N>/annotation.json" \
  --output-dir /Volumes/External/argus_caches/vits_window_<N>/train \
  --backbone vit --model-size small \
  --weights weights/dinov3_vits16_lvd1689m.pth \
  --image-size 518 --native-tile-size 400 --tile-overlap 0.0 --norm-mode none

python scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ARGUS_DATA_ROOT/train_atwood_synth_window_<N>/val_annotation.json" \
  --output-dir /Volumes/External/argus_caches/vits_window_<N>/val \
  --backbone vit --model-size small \
  --weights weights/dinov3_vits16_lvd1689m.pth \
  --image-size 518 --native-tile-size 400 --tile-overlap 0.0 --norm-mode none
```

### 4. Train the conv head

```bash
python training/train_dinov3_heatmap_cached.py \
  --train-cache /Volumes/External/argus_caches/vits_window_<N>/train \
  --val-cache   /Volumes/External/argus_caches/vits_window_<N>/val \
  --work-dir weights/vits_window_<N> \
  --epochs 40 --lr 1e-3 --batch-size 32 --hidden-channels 256 \
  --lr-scheduler cosine --early-stopping-patience 10 \
  --loss-mode asl_cldice \
  --asl-gamma-neg 4.0 --asl-gamma-pos 0.0 --asl-margin 0.05 --cldice-iters 3
```

Delete the feature cache after training:

```bash
rm -rf /Volumes/External/argus_caches/vits_window_<N>
```

### 5. Evaluate

```bash
# Run inference on the validation set
python scripts/run_geometry_eval.py \
  --model-tag vits_window_<N> \
  --checkpoint weights/vits_window_<N>/best.pt \
  --annotations "$ARGUS_DATA_ROOT/annotations/val_balanced_v1.json" \
  --output-dir results/window_<N>/vits_window_<N>/pf85 \
  --threshold 0.70 --peak-floor 0.85

# Score the predictions
python -m eval.geometry_metrics \
  --predictions results/window_<N>/vits_window_<N>/pf85/predictions_t070.json \
  --annotations "$ARGUS_DATA_ROOT/annotations/val_balanced_v1.json" \
  --output results/window_<N>/vits_window_<N>/pf85/geometry_eval.json

# Compare across all runs
python scripts/compare_geometry_evals.py --md
```

Commit only `geometry_eval.json`. Raw `predictions_t*.json` and `metrics_t*.json`
files are regenerable and gitignored.

**Important notes:**
- Never launch a second heavy MPS job while one is training (single GPU).
- Do not enable Radon refinement (T3) — T2 raw OBB geometry is more accurate.
- Keep only one backbone feature cache on the internal drive at a time (100 GB
  budget on a shared machine).

See [`agent_docs/ml_pipeline.md`](agent_docs/ml_pipeline.md) for additional
detail and [`docs/loss_ablation_v9_v10_postmortem.md`](docs/loss_ablation_v9_v10_postmortem.md)
for the loss-function rationale.

---

## TLE Catalog

Runtime satellite identification uses only locally stored catalog data. Bootstrap
before processing historical images:

```bash
# Recent coverage (last 90 days) — required for live pipeline use:
export ARGUS_ENV=production
python scripts/bootstrap_recent_tles.py

# Prior years (annual zip bundles):
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2025
```

Space-Track credentials must be set in `.env` for catalog updates. See
[`agent_docs/spacetrack.md`](agent_docs/spacetrack.md) for the full TLE
management guide including rate limits and Space-Track API policy.

---

## FAST_MODE

Setting `FAST_MODE=true` in `.env` silently disables cross-identification. Check
this first if detections are returning 0 satellite IDs.

---

## Evaluation Reference

Canonical parameters across all experiments:

| Parameter | Value |
|---|---|
| Validation set | `val_balanced_v1.json` |
| Heatmap threshold | 0.70 |
| Peak floor | 0.85 |
| Native tile size | 400 px |
| Image size | 518 px |
| Normalization | zscore |

Length bands: short < 50 px, medium 50–400 px, long > 400 px.
Geometry compatibility threshold: 10 px perpendicular distance.
