# ARGUS

ARGUS detects satellite streaks in astronomical FITS images, maps them through
WCS, and cross-identifies them against a local TLE catalog. Every streak is a
line segment defined by `x1`, `y1`, `x2`, and `y2`.

## Architecture

- `models/plain_dinov3/`: one-channel DINOv3 heatmap models
- `training/`: endpoint datasets and heatmap trainers
- `inference/`: FITS loading, tiled detection, segment post-processing, WCS,
  confidence, and cross-identification
- `eval/geometry_metrics.py`: canonical segment evaluator
- `db/`, `api/`, `frontend/`: persistence and product surface
- `src/`: FITS ingestion, astrometry, and satellite matching components

Historical source annotations are normalized once by
`training.annotation_endpoints`. Working annotations, predictions, database
records, API payloads, and UI rendering all use endpoints.

## Local development

Create the Python environment from the repository root:

```bash
conda env create -f environment.yml
conda activate satid
python -m pip check
```

The environment file is the tested macOS development path. CUDA users should
instead create a Python 3.11 environment, install the PyTorch build recommended
for their CUDA platform, and then install `requirements-dev.txt` with pip.

Start the API:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

In a second terminal, using Node.js 20.19+ or 22.12+:

```bash
cd frontend
npm ci
npm run dev
```

## Tests

```bash
python -m pytest tests/ -q
```

Tests are offline and do not require production checkpoints or Space-Track
credentials.

## Training and evaluation

Feature caches are created from endpoint annotation JSON:

```bash
python scripts/cache_dinov3_heatmap_features.py \
  --annotations "$ARGUS_DATA_ROOT/annotations/train.json" \
  --output-dir /tmp/argus-cache \
  --backbone vits --image-size 400
```

Use the cached or end-to-end DINOv3 heatmap trainer appropriate to the run. Models
must emit one centerline channel. Evaluate predictions with:

```bash
python -m eval.geometry_metrics \
  --predictions results/<run>/predictions.json \
  --annotations "$ARGUS_DATA_ROOT/annotations/val.json" \
  --output results/<run>/geometry_eval.json
```

Record the dataset version, coordinate frame, tile size, normalization,
checkpoint, and threshold for every run. See `docs/training_methods.md`.

## TLE catalog

Runtime identification uses only locally stored catalog data. Bootstrap and
maintenance are explicit operator actions:

```bash
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2026
python scripts/update_tle_catalog.py
```

See `agent_docs/assistant_guide.md` for the canonical contributor instructions.
