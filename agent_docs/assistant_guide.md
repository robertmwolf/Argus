# ARGUS Assistant Guide

This is the canonical instruction file for coding assistants working on ARGUS.
Both `AGENTS.md` and `CLAUDE.md` point here.

## What ARGUS is

ARGUS detects satellite streaks in astronomical FITS images, represents every
streak as two image-space endpoints, optionally resolves those endpoints through
WCS, and cross-identifies the result against the local TLE catalog. FastAPI serves
results to a React frontend.

## Canonical streak contract

A streak is defined only by `x1`, `y1`, `x2`, and `y2`. Derived values such as
centre, angle, and length may be cached, but endpoints remain authoritative.
Training targets, model outputs, post-processing, evaluation, persistence, API
responses, and frontend rendering must preserve this contract.

Historical source annotations may arrive in an older rectangle-shaped schema.
Convert them exactly once through `training.annotation_endpoints`; do not let that
schema propagate into datasets, targets, predictions, or public interfaces.

## Active ML path

- DINOv3 ViT heatmap models predict a one-channel streak centerline heatmap.
- Connected heatmap components become endpoint segments.
- Tiled inference remaps endpoints, suppresses duplicates, and stitches compatible
  collinear segments.
- `eval.geometry_metrics` is the standard evaluator.
- `inference.pipeline` is the production orchestrator.

Do not add detector heads that predict rectangles, widths, corners, or polygons.
Do not add rectangle overlap metrics. Geometry quality is measured using segment
angle, perpendicular offset, along-track overlap, and endpoint error.

## Repository map

- `inference/`: FITS loading, heatmap detection, endpoint post-processing, WCS,
  confidence, and cross-identification.
- `models/plain_dinov3/`: active one-channel heatmap models.
- `training/`: endpoint datasets and heatmap trainers.
- `eval/geometry_metrics.py`: endpoint evaluation.
- `api/`, `db/`, `frontend/`: product surface.
- `scripts/`: dataset preparation, caching, evaluation, and operations.
- `src/`: classical astronomy and catalog components.

## Data and naming

- Full-frame annotations use source image coordinates; materialized crops use
  local coordinates. Never pair full-frame pixels with crop-local endpoints.
- Keep train, validation, and test splits deterministic and leakage-free.
- Keep raw FITS, annotations, derived datasets, and feature caches outside the
  repository. Do not create dataset symlinks under `data/`.
- `ARGUS_DATA_ROOT` = `/Volumes/External/TrainingData` (durable, external drive).
  `ARGUS_SCRATCH_ROOT` = disposable local mirror under `/tmp`.
- Never hard-code `/Volumes/...`, `/tmp/...`, or `data/annotations/...` in new
  training and evaluation code.
- Canonical training source: `annotations/all_train_run17_merged_no_sattrains.json`
- Canonical eval set: `annotations/val_balanced_v1_no_sattrains.json`
- Satellite-train frames (≥2 near-parallel streaks, angle_diff < 5°, perp_dist
  < 30 px) are excluded from both training and evaluation. Exclusion manifest:
  `annotations/sat_train_excluded.json`.

See `agent_docs/datasets.md` for the path-resolution contract and current drive
layout. See `docs/data_strategy.md` for the data integration workflow.

## Environment and tests

The usual local environment is `/Users/robert/miniconda3/envs/satid`.

After creating the environment, retrieve the published Hugging Face weights:

```bash
python scripts/sync_hf.py --download --weights-only --weights-dir weights
```

The public bundle requires no token under normal conditions. If authentication
is required, use `hf auth login` or `HF_TOKEN`. The bundle contains the
`vits_v11_asl_cldice` head. DINOv3 pretrained backbones are not in the bundle —
they must be downloaded separately from Meta Research at
https://github.com/facebookresearch/dinov3 and placed in `weights/`.
Never commit downloaded weights.

```bash
/Users/robert/miniconda3/envs/satid/bin/python -m pytest tests/ -q
```

Tests are expected to run offline. Mock network access and heavyweight model
loading. Use `inference.device` for device selection; do not hard-code CUDA.

## Working rules

- Read this file before modifying the repository.
- Preserve unrelated user changes in a dirty worktree.
- Prefer small, typed functions and NumPy-style docstrings.
- Keep imports side-effect free and avoid network calls at import time.
- Validate endpoint coordinates at boundaries and derive angle/length from them.
- Add or update tests whenever behavior changes.
- Never commit weights, credentials, FITS datasets, caches, or generated results.
- Runtime satellite matching uses the local database. Space-Track access is only
  for explicit catalog maintenance.

## Current production commands

```bash
# API
/Users/robert/miniconda3/envs/satid/bin/uvicorn api.main:app --reload --port 8000

# Endpoint geometry evaluation
/Users/robert/miniconda3/envs/satid/bin/python -m eval.geometry_metrics \
  --predictions results/<run>/predictions.json \
  --annotations "$ARGUS_DATA_ROOT/annotations/<split>.json" \
  --output results/<run>/geometry_eval.json
```
