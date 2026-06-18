# ARGUS Script Inventory

This directory contains a mix of active operator tools, dataset builders,
training/evaluation drivers, manual annotation utilities, and research
provenance scripts. Do not delete or move scripts just because they are not on
the main inference path: several preserve how committed annotations, results, or
model checkpoints were produced.

Status labels:

- **Active**: expected to remain runnable and documented.
- **Manual**: human-in-the-loop annotation or review workflow.
- **Provenance**: documents/recreates an experiment or generated artifact.
- **Spike**: exploratory research path; preserve until results are fully
  archived elsewhere.
- **Maintenance**: operational upkeep script, usually credentialed or scheduled.

## Active Operator And Maintenance Scripts

| Script | Status | Purpose | Notes |
|---|---:|---|---|
| `bootstrap_tle_catalog.py` | Active | One-time local TLE catalog bootstrap from Space-Track annual bundles. | Referenced by README, assistant guide, architecture, and Space-Track docs. |
| `bootstrap_recent_tles.py` | Maintenance | Idempotent historical TLE backfill and daily keep-up via `gp_history`. | Schedule off-hour; requires Space-Track credentials. |
| `download_tle_bundle.py` | Maintenance | Helper for Space-Track annual TLE bundle downloads/listing. | Used during catalog setup. |
| `update_tle_catalog.py` | Maintenance | Explicit current GP-class update into the local catalog. | Operator-run only; inference should not call it. |
| `celestrak_client.py` | Active | CelesTrak fallback/live TLE edge for `src.matching.tle_manager`. | Imported by tests and runtime code. |
| `download_weights.py` | Active | Downloads Swin-T/Swin-L DINO pretrain weights based on `MODEL_SIZE`. | Does not handle DINOv3 portal weights. |
| `prepare_cloud_training.py` | Active | Pre-flight checklist before GPU training. | Main go/no-go gate before paid or remote GPU runs. |
| `cloud_setup.sh` | Active | WSL2/Linux cloud setup for GPU training. | Bootstrap only; run `prepare_cloud_training.py` afterward as the true go/no-go gate. |
| `fetch_weights.sh` | Active | Fetch trained weights/logs back from remote machine. | Companion to cloud/workstation training runs. |

## Dataset Build And Merge Scripts

Canonical training annotation JSONs live on the external drive under
`/Volumes/External/TrainingData/annotations/`. Dataset builders default there
via `ARGUS_ANNOTATIONS_DIR`; repo-local `data/annotations` is a compatibility
path for older commands and active runs.

| Script | Status | Purpose | Notes |
|---|---:|---|---|
| `convert_gtimages.py` | Active | Converts GTImages `.strk`/FITS metadata into COCO-style annotations. | Covered by tests. |
| `merge_annotations.py` | Active | Builds current SatStreaks + GTImages train/val/test splits. | Main split builder in README. |
| `merge_fits_annotations.py` | Provenance | Merges FITS-native annotation sources such as GTImages + Frigate. | |
| `prepare_atwood_holdout.py` | Active | Exports reviewed Atwood annotator output into positive and negative COCO holdout files. | Use before `zero_shot_eval.py`; excludes rejected and still-pending frames. |
| `augment_gtimages_synthetic.py` | Provenance | Builds real and synthetic GTImages tracks for StreakMind reproduction. | Keep with methodology/results provenance. |
| `augment_short_medium.py` | Provenance | Generates short/medium synthetic augmentation data. | Experiment-specific; archive only after outputs and rationale are captured elsewhere. |

## Training And Evaluation Drivers

| Script | Status | Purpose | Notes |
|---|---:|---|---|
| `evaluate_dino_checkpoint.py` | Active | Evaluates a DINO checkpoint with project metrics. | Useful for ad hoc checkpoint checks. |
| `phase_e_compare.py` | Active | Head-to-head DINOv3 vs Swin-T comparison table/JSON. | Documented in training handoff. |
## Plain DINOv3 Spike Scripts

These are tied to the archived/plain DINOv3 heatmap and box experiments. They
are not the primary production model path, but the results in `results/` depend
on them.

| Script | Status | Purpose | Notes |
|---|---:|---|---|
| `cache_dinov3_heatmap_features.py` | Spike | Caches frozen DINOv3 features for heatmap/box heads. | Referenced by `training/train_dinov3_box_cached.py`. |
| `evaluate_dinov3_heatmap.py` | Spike | Evaluates the plain DINOv3 heatmap head. | Referenced by `agent_docs/plain_dinov3_spike.md`. |
| `evaluate_dinov3_box.py` | Spike | Evaluates the plain DINOv3 box head. | Referenced by `agent_docs/plain_dinov3_spike.md`. |

## Manual Annotation And Review Tools

These scripts are intentionally kept even when they look one-off. They encode
human review workflows, Hough suggestion settings, and dataset-specific file
handling that will likely be useful for future labeling passes.

| Script | Status | Purpose | Notes |
|---|---:|---|---|
| `annotate.py` | Manual | Unified Tkinter OBB streak annotation tool for FITS, PNG, and JPEG inputs. | Handles Frigate priority lists, generic image directories, BrentImages `.strk` write-back via `--night-dir`, pending-only filtering, and explicit Go-to navigation. |
| `annotate_frigate.py` | Manual | Builds Frigate unreviewed background-candidate COCO files. | Output is not a true-negative label set unless reviewed. |
| `screen_frigate.py` | Manual | Ranks Frigate frames by short-streak likelihood. | Produces priority JSON/contact sheet for annotation. |

Archived superseded entry points live under `scripts/archive/` for provenance:
`annotate_streaks.py`, `annotate_frigate_streaks.py`, and
`annotate_brentimages_streaks.py`.

## Test And Development Support

| Script | Status | Purpose | Notes |
|---|---:|---|---|
| `make_test_fits.py` | Active | Synthetic FITS generator and helper functions for tests. | Imported by tests and `training/make_dev_subset.py`; do not move casually. |
| `__init__.py` | Active | Makes `scripts` importable for tests/runtime helpers. | Required because some modules import helpers from scripts. |

## Cleanup Policy

Use this order for future cleanup:

1. Update this inventory first when adding or retiring a script.
2. Before moving a script, run `rg "<script_name>" README.md agent_docs tests training src api eval`.
3. Prefer moving inactive research code to `scripts/archive/` over deleting it.
4. Do not archive scripts that generated committed annotations, weights, or
   results until the exact workflow is documented in `agent_docs/` or
   `METHODOLOGY.md`.
5. Keep manual annotation tools available unless a replacement can read the same
   inputs, write the same COCO fields, and preserve existing suggestion caches.
