# ARGUS Datasets

ARGUS uses FITS observations from Atwood Observatory plus selected synthetic
centerline examples. A canonical annotation has `x1`, `y1`, `x2`, and `y2` in
the coordinate frame of its referenced image.

## Coordinate frame rules

Full-frame annotation files contain full-frame endpoint coordinates. Materialized
crop or tile datasets contain local endpoint coordinates and must reference the
matching cropped pixels. Never mix those frames.

Historical source files may encode annotations as OBBs or polygons. Dataset
loaders and builders must normalize them through `training.annotation_endpoints`
before any split, crop, augmentation, cache, target, or evaluation step.

Every generated annotation file should record its builder, seed, source files,
and coordinate frame in a `provenance` key in the JSON root.

---

## Data locations and local staging

Do not create dataset symlinks inside `data/`. Point ARGUS at the durable dataset
tree instead:

```bash
export ARGUS_DATA_ROOT=/Volumes/External/TrainingData
export ARGUS_SCRATCH_ROOT=/tmp/argus-data
```

The same keys may be placed in the project `.env`; command-line flags override
`.env`, and exported variables override it as well.

The roots have deliberately different lifetimes:

- `ARGUS_DATA_ROOT` is authoritative, durable user data. It lives on an external
  drive and must not be modified or removed by training workflows.
- `ARGUS_SCRATCH_ROOT` is a path-preserving local mirror used to avoid training
  directly from a slower external drive. It is a cache, not a source of truth.
- Repository `data/` contains only small versioned fixtures and application data.
  It is not a mount point for training datasets.

Store image `file_name` values in annotation JSON relative to `ARGUS_DATA_ROOT`.
Before training or validation, copy only referenced files to the local scratch
mirror:

```bash
python scripts/stage_dataset_files.py \
  --annotations \
  "$ARGUS_DATA_ROOT/annotations/train.json" \
  "$ARGUS_DATA_ROOT/annotations/val.json"
```

Resolution precedence (for each image the loader checks in order):
1. An explicit `--data-root` or `--scratch-root` command-line value.
2. The exported `ARGUS_DATA_ROOT` or `ARGUS_SCRATCH_ROOT` value.
3. The corresponding project `.env` value.

Staging can be repeated safely; files whose sizes already match are skipped
unless `--refresh` is supplied.

---

## Current external drive layout (`/Volumes/External/TrainingData`)

This is the live `ARGUS_DATA_ROOT` as of June 2026.

```
TrainingData/
  raw/
    BrentImages/                   # Source FITS from Atwood Observatory (ongoing)
      Img_20260412_Atwood/         # Each batch has its own COCO annotation JSON
      Img_20260515_Atwood/
      Img_20260527_Atwood/
      Img_20260528_Atwood/
      Geo_20260520_Atwood/
      20260530_Atwood/
  annotations/
    all_train_run17_merged_no_sattrains.json  # Canonical training source (active)
    all_train_run17_merged.json               # Pre-exclusion archive; do not use for new runs
    sat_train_excluded.json                   # Exclusion manifest (53 satellite-train frames)
    val_balanced_v1_no_sattrains.json         # Canonical eval set (241 images, 247 annotations)
    val_balanced_v1.json                      # Pre-exclusion eval archive
    hard_negatives_vits_window_v4.json        # Mined FP tiles from vits_window_v4
    hard_negatives_vits_window_v5.json        # Mined FP tiles from vits_window_v5 (t=0.85)
    hard_negatives_vits_window_v5_t075.json
  train_atwood_synth_window_v10/   # Active training tile dataset (rebuilt June 2026)
  val_atwood_window_v10/           # Active internal val tile dataset (for early stopping)
  train_atwood_synth_window_v9/    # Prior training dataset (kept for reference/comparison)
    annotation.json                # Original v9 annotation
    annotation_no_sattrains.json   # Filtered annotation used for v10 comparison runs
  val_atwood_window_v9/
    annotation.json
    annotation_no_sattrains.json
```

**Ground truth lives in the per-batch COCO JSONs** inside each `BrentImages/`
subfolder. The merged annotation files (`all_train_run17_merged*.json`) are
derived and can be regenerated. The ground truth JSONs are authoritative.

**Window datasets are self-contained.** Each `train_atwood_synth_window_vN/`
directory contains pre-rendered float32 `.npy` tile crops (already per-frame
z-score normalized) plus an `annotation.json` with relative paths. Build with:

```bash
python scripts/build_atwood_window_dataset.py \
  --version N \
  --source annotations/all_train_runM_merged_no_sattrains.json \
  --eval-frames-json annotations/val_balanced_v1_no_sattrains.json \
  --val-frac 0.08 --neg-frac 0.42 --bg-per-frame 3 --seed 42
```

**Feature caches are ephemeral.** Training pipelines build a ViT feature cache,
train, then delete the cache. Eval pipelines build a heatmap cache, evaluate,
then delete it. Do not store durable data under `/Volumes/External/argus_caches/`.
Cache names use the model tag (e.g. `vits_v10_no_sattrains`) for disambiguation.

---

## Canonical annotation files

| File | Images | Annotations | Use |
|------|--------|-------------|-----|
| `all_train_run17_merged_no_sattrains.json` | 6052 | 12 329 | Training source (active) |
| `val_balanced_v1_no_sattrains.json` | 241 | 247 | Eval / geometry metrics (active) |
| `all_train_run17_merged.json` | 6105 | 12 647 | Archive; pre-exclusion |
| `val_balanced_v1.json` | 246 | 252 | Archive; pre-exclusion |

When in doubt, use the `_no_sattrains` variants. The non-suffixed files exist for
historical comparison only.

---

## Feature-cache lifecycle

1. Stage the FITS/NPY files referenced by both train and validation manifests
   with `scripts/stage_dataset_files.py`.
2. Build frozen feature caches with `scripts/cache_dinov3_heatmap_features.py`.
   Its `--output-dir` is explicit and points outside the repository.
3. For cached-head training, pass the completed cache directories through
   `--train-cache` and `--val-cache`.
4. Remove caches only after the consuming command has completed and only as part
   of an explicitly authorized workflow.

Do not encode scratch paths in annotation JSON, replace a scratch directory with
a symlink, or rely on the current working directory to locate source images.

---

## Adding new data

See [`docs/data_strategy.md`](../docs/data_strategy.md) for the step-by-step
workflow for integrating a new batch of BrentImages, including the satellite-train
exclusion check and dataset rebuild procedure.
