# ARGUS Datasets

ARGUS uses FITS observations from Atwood Observatory, Frigate, and other reviewed
sources, plus selected synthetic centerline examples. A canonical annotation has
`x1`, `y1`, `x2`, and `y2` in the coordinate frame of its referenced image.

Historical source files may encode annotations differently. Dataset loaders and
builders must normalize them through `training.annotation_endpoints` before any
split, crop, augmentation, cache, target, or evaluation step.

Keep source data and large generated datasets outside git. Preserve deterministic
split seeds and provenance metadata. Never use validation or test images as
training negatives.

## Data locations and local staging

Do not create dataset symlinks inside `data/`. Point ARGUS at the durable dataset
tree instead:

```bash
export ARGUS_DATA_ROOT=/Volumes/MyDrive/argus-data
export ARGUS_SCRATCH_ROOT=/tmp/argus-data
```

The same keys may be placed in the project `.env`; command-line flags override
`.env`, and exported variables override it as well.

The roots have deliberately different lifetimes:

- `ARGUS_DATA_ROOT` is authoritative, durable user data. It can live on an
  external drive and must not be modified or removed by training workflows.
- `ARGUS_SCRATCH_ROOT` is a path-preserving local mirror used to avoid training
  directly from a slower external drive. It is a cache, not a source of truth.
- Repository `data/` contains only small versioned fixtures and application
  data. It is not a mount point for training datasets.

Store image `file_name` values in annotation JSON relative to
`ARGUS_DATA_ROOT`. Before training or validation, copy only referenced files to
the local scratch mirror:

```bash
python scripts/stage_dataset_files.py \
  --annotations \
  "$ARGUS_DATA_ROOT/annotations/train.json" \
  "$ARGUS_DATA_ROOT/annotations/val.json"
```

Training, feature caching, and heatmap evaluation look in the scratch mirror
first and fall back to the durable root. The directory layout is identical in
both places, so annotations remain unchanged and portable. Existing absolute
paths remain readable when they are beneath `ARGUS_DATA_ROOT`; new manifests
must use relative paths.

Resolution precedence is:

1. An explicit `--data-root` or `--scratch-root` command-line value.
2. The exported `ARGUS_DATA_ROOT` or `ARGUS_SCRATCH_ROOT` value.
3. The corresponding project `.env` value.

For each image, the loader checks `<scratch-root>/<file_name>` first, followed
by `<data-root>/<file_name>`. If a scratch copy is absent, reading from durable
storage remains valid. Staging can therefore be repeated safely; files whose
sizes already match are skipped unless `--refresh` is supplied.

## Feature-cache lifecycle

Raw-file staging and model feature caching are separate layers:

1. Stage the FITS/NPY files referenced by both train and validation manifests
   with `scripts/stage_dataset_files.py`.
2. Build frozen feature caches with `scripts/cache_dinov3_heatmap_features.py`.
   Its `--output-dir` is explicit and should point outside the repository.
3. For cached-head training, copy the completed train and validation feature
   cache directories to local scratch and pass those exact directories through
   `--train-cache` and `--val-cache`.
4. Remove scratch copies only after the consuming command has completed and
   only as part of an explicitly authorized workflow. Durable raw data and
   durable feature caches remain untouched.

Do not encode scratch paths in annotation JSON, replace a scratch directory with
a symlink, or rely on the current working directory to locate source images.
