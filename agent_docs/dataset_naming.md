# ARGUS Dataset Naming & Versioning

## Guiding Principle

Dataset names encode **what the data is**, not **which training run produced it**.
Run-scoped names like `all_train_run17_merged.json` make reusable datasets look
ephemeral.  Content-based names make the relationship between dataset and model
transparent and let datasets survive across runs.

---

## Name Format

### Full-frame datasets  (COCO JSON only, no pixel files)

```
{split}_{source}[_{filter}]_v{N}.json
```

Stored under `/Volumes/External/TrainingData/annotations/`  
and mirrored to `data/annotations/` for small files used by local scripts.

### Tiled datasets  (COCO JSON + NPY tile files)

```
{split}_{source}_{method}[_{params}]_v{N}/
├── annotation.json       ← file_names are relative paths into tiles/
└── tiles/
    └── *.npy
```

Stored as self-contained directories under `/Volumes/External/TrainingData/`.
The directory name **is** the dataset name.  The annotation uses **relative
paths** so the directory is portable (copy or move it without breaking anything).

---

## Fields

### `split`

| Value | Meaning |
|---|---|
| `train` | Training data |
| `val` | Validation split (used for hyperparam tuning and mid-run checks) |
| `test` | Held-out test set — **do not use for hyperparam tuning** |

### `source`

| Value | Meaning |
|---|---|
| `atwood` | Atwood observatory FITS (BrentImages capture) |
| `frigate` | Frigate camera manually-annotated frames |
| `synth` | Synthetically generated streak tiles |
| `atwood_frigate` | Merged Atwood + Frigate |
| `atwood_frigate_synth` | Merged Atwood + Frigate + synthetic |

### `method`  (tiled datasets only, omitted for full-frame)

| Value | Meaning | Script |
|---|---|---|
| `near_ctx` | Annotation tile + N nearest-neighbour context tiles, ordered by proximity to annotation centre | `scripts/build_tiled_val_annotation.py` |
| `rand_neg` | Annotation tiles + randomly sampled negatives to a target ratio | `scripts/build_tiled_val_annotation.py` (removed in favour of `near_ctx`) |
| `tiled` | Uniform grid — every tile at the given stride (no selection) | `scripts/cache_dinov3_heatmap_features.py` |
| `window` | Streak-centred crops (one materialised window per annotation cluster) + clean negative-sky crops; the cacher tiles each window at run time | `scripts/build_atwood_window_dataset.py` |

### `params`  (method-specific, appended after method)

| Token | Meaning | Example |
|---|---|---|
| `t{N}` | Tile size in pixels | `t400` |
| `c{N}` | Context neighbour count (`near_ctx` only) | `c4` |
| `neg{N}` | Negative tile target percentage (`rand_neg` only) | `neg38` |
| `ov{N}` | Tile overlap percentage, if non-zero | `ov25` |

### `v{N}`

Integer version starting at 1.

**Bump the version when:**
- Source images change (new observations added, images removed or corrected)
- The processing algorithm is revised in a way that changes the output
- A data quality issue (mislabelled annotation, bad normalisation) is corrected

**Do not bump the version when:**
- Regenerating the exact same dataset from the same source + algorithm
- Re-running evaluation against an existing dataset

Keep at least one prior version around until two newer versions exist.

---

## Storage Locations

| Content | Path |
|---|---|
| Full-frame COCO JSONs | `/Volumes/External/TrainingData/annotations/` |
| Local copies (small files, gitignored data/) | `data/annotations/` |
| Tiled dataset directories | `/Volumes/External/TrainingData/{dataset_name}/` |

Tiled datasets are **never** stored under `data/annotations/` — they are too
large for the repo machine's local disk.

---

## Current Datasets

### Full-frame sources  (canonical, full 6248 × 4176 FITS paths)

| File | Annotations | Notes |
|---|---|---|
| `val_run17_fits.json` | 1 156 | Current canonical val — richer labelling than `val_atwood.json`. Rename to `val_atwood_v2.json` at next edit. |
| `val_atwood.json` | 228 | Older val annotation, same 240 images. Effectively `val_atwood_v1.json`. |
| `test_atwood.json` | 228 | Held-out test. Rename to `test_atwood_v1.json` at next edit. |

### Tiled eval datasets

| Directory | Source | Method | Tile | Context | Notes |
|---|---|---|---|---|---|
| `val_atwood_near_ctx_t400_c4_v1/` | `val_run17_fits.json` | `near_ctx` | 400 px | 4 | **Current recommended fast eval** |

### Heatmap training datasets (window crops)

Self-contained dirs under TrainingData root; `annotation.json` uses **relative**
`tiles/*.npy` paths (raw float32; zscore applied at cache time). obb coords are
**window-local** because each window is a materialised crop with `tile_origin=[0,0]`.

| Directory | Source | Contents | Built by |
|---|---|---|---|
| `train_atwood_synth_window_v1/` | `all_train_run17_merged.json` | real Atwood streak windows + synthetic-short + neg-sky crops | `scripts/build_atwood_window_dataset.py` |
| `val_atwood_window_v1/` | same, held-out frames | real Atwood streak windows + neg-sky crops | same |

**Supersedes** `data/annotations/{train,val}_run18.json` (run-scoped name + a
coordinate-frame bug — see lesson below). Delete those once v1 is validated.

> **Coordinate-frame lesson (2026-06-13).** `build_run18_split.py` emitted
> annotations whose `file_name` was the FULL 6248×4176 frame but whose obb
> coords were **window-local** (offset by `tile_origin`), and the heatmap cacher
> tiles the whole `file_name` image — so every target landed ~1800 px off the
> streak, on empty sky. Both ViT-S and ViT-B trained to val_dice ~0.12 on it
> (Run 20 control). **Rule: an image's pixels and its obb coords must share one
> frame.** If `file_name` is a full frame, obb must be full-frame; if obb is
> window-local, materialise the crop and set `tile_origin=[0,0]`. The `window`
> builder always materialises the crop, so the two can never disagree.

### Legacy training files  (run-scoped, not for reuse)

These exist from before this naming convention.  Do not create new files
following these patterns.  Delete after the run that produced them is complete.

| File | Legacy purpose |
|---|---|
| `all_train_run17_merged.json` | Run 17 merged train annotation — ephemeral |
| `all_train_run{N}_*.json` | Older run-scoped train merges |
| `val_run17_fits.json` | Run 17 de-tiled val — promote to canonical name at next edit |
| `val_run12_1800_npy.json` | Run 12 NPY val — superseded |
| `val_atwood_tiled_ts1800.json` | Pre-run17 tiled val — superseded |
| `synth_run{N}_*.json` | Synthetic NPY tiles scoped to a run |

---

## `near_ctx` Method — Detail

**Script:** `scripts/build_tiled_val_annotation.py`

**Algorithm:**
1. Tile each full-frame image at `tile_size` with 0.0 overlap.
2. For each GT annotation, identify the tile whose bounds contain the OBB
   centre  → **positive tile**.
3. From the positive tile's up-to-8 grid neighbours, select the `--context N`
   closest to the annotation centre (Euclidean distance from annotation centre
   to neighbour tile centre).  These are the **context tiles**.
4. Deduplicate across all annotations in the image.
5. Write tile crops as single-channel float32 NPY files (raw pixel values,
   no normalisation baked in).  Normalisation is applied at eval time via
   `--norm-mode zscore`.

**Why proximity-ordered neighbours?**  The tiles the streak is heading toward
(in the direction of its nearest tile boundary) naturally rank first.  These
are where bleed-over false positives first appear.  With `--context 4`, the
four highest-priority neighbours are selected, covering the streak's forward
and lateral directions without inflating the tile count with distant background.

**Tile counts for val_run17_fits.json  (240 images, 1 156 annotations):**

| `--context` | Total tiles | Neg ratio | vs full tiling |
|---|---|---|---|
| 0 | ~1 344 | 0 % | 31× faster |
| 2 | ~3 744 | 64 % | 11× faster |
| **4** | **~5 424** | **75 %** | **8× faster** |
| 8 | ~6 912 | 81 % | 6× faster |
| — | 42 240 | 97 % | 1× (current baseline) |

**Evaluate a `near_ctx` dataset:**
```bash
ARGUS_NORM=zscore \
python scripts/evaluate_dinov3_heatmap.py \
    --annotations /Volumes/External/TrainingData/val_atwood_near_ctx_t400_c4_v1/annotation.json \
    --checkpoint  weights/<run>/best.pt \
    --output      results/<run>/fast_eval/metrics.json \
    --norm-mode   zscore
# No --tiled flag needed — each file is already a 400-px crop.
```

**Rebuild command:**
```bash
python scripts/build_tiled_val_annotation.py \
    --annotations /Volumes/External/TrainingData/annotations/val_run17_fits.json \
    --output-dir  /Volumes/External/TrainingData/val_atwood_near_ctx_t400_c4_v1 \
    --tile-size 400 \
    --context 4
```

---

## Examples

```
# Full-frame (annotations/ directory)
val_atwood_v2.json               ← canonical val, 1 156 annotations (rename from val_run17_fits.json)
train_atwood_v1.json             ← canonical train, full-frame FITS
train_atwood_frigate_v1.json     ← merged Atwood + Frigate

# Tiled (self-contained directories at TrainingData root)
val_atwood_near_ctx_t400_c4_v1/  ← default fast eval
val_atwood_near_ctx_t400_c2_v1/  ← faster, fewer context tiles
train_atwood_tiled_t400_v1/      ← uniform-grid train tiles (if ever needed)
```
