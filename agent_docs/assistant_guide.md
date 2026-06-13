# ARGUS Assistant Guide

This is the canonical instruction file for coding assistants working on ARGUS.
Both Codex (`AGENTS.md`) and Claude Code (`CLAUDE.md`) should read this file.
Keep durable assistant instructions here; keep deeper reference material in the
topic-specific files under `agent_docs/`.

## ARGUS — Satellite Streak Detection & Identification Pipeline

## What This Is
An end-to-end pipeline that detects satellite streaks in FITS telescope images
using a Co-DINO transformer model (Swin-L backbone), refines streak angle via
the Radon transform, and cross-identifies detected objects against a local TLE
catalog (sourced once from Space-Track, stored in the ARGUS database) using SGP4
propagation and multi-factor confidence scoring. Results are served through a
FastAPI backend and React frontend.

## Current Phase
**Run 15 active (2026-06-07/08). ViT-S heatmap is the production detector.**

Progress:
- ✅ Phase 0 (Classical baseline): `src/` — fits_parser, classical_detector, plate_solver, SGP4 matching
- ✅ Phase 1 (Data pipeline): `inference/fits_loader.py`, `training/convert_labels.py`, `training/dataset.py`, `training/augmentations.py`
- ✅ Phase 2 (DINO model): `inference/device.py`, `models/dino/` configs, `training/train_dino.py`
- ✅ Phase 3 (Inference pipeline): `inference/pipeline.py`, `inference/postprocess.py`, `inference/crossid.py`
- ✅ Phase 4 (Database): `db/schema.sql`, `db/models.py` — SQLAlchemy async ORM
- ✅ Phase 5 (API): `api/main.py`, `api/storage.py`, `api/queue.py`, `api/worker.py`
- ✅ Phase 6 (Frontend): `frontend/` — React 18 + Vite + Tailwind; **line-segment rendering (not OBBs)**
- ✅ Phase 8 (Evaluation): `eval/metrics.py`, `eval/benchmark.py`
- ✅ DINOv3 ViT-B backbone integrated — `models/dino/dinov3_adapter.py`
- ✅ Adaptive tiling — `inference/tiled_pipeline.py`
- ✅ **Run 3 complete** — cold-start ViT-B OBB, 15 epochs. mAP=0.782, P=94.9%, R=83.8%.
  Weights: `weights/run3_cold_nodm/best.pth`. See `docs/training_methods.md §3.2`.
- ✅ **Run 4 complete (2026-05-29)** — two ViT-S OBB models on geometry-stratified Atwood + Frigate.
  - **OBB MMDet ViT-S (Run 5 cache):** `weights/run5_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth`
    Val: mAP@50=0.611. Test: short-recall=0%, medium-recall=49%, long-recall=75%.
  - **Centerline ViT-S:** val dice=0.2327 (plateau from epoch 1). Superseded by heatmap approach.
- ✅ **Runs 5–10 complete (2026-05-31 – 2026-06-06)** — ViT-S/ConvNeXt-S frozen heatmap grid search.
  - ConvNeXt-S definitively eliminated at Run 10: zero medium-streak recall at all configs.
  - ViT-S 10a (38% neg, cosine LR): best frozen result — medium_recall=9.09%, recall=1.75%, F1=1.54%.
  - **Root cause:** frozen ImageNet features cannot distinguish FITS streak signal from background.
    Gate (recall ≥ 60%) requires backbone adaptation.
  - See `docs/training_methods.md §3.7–§3.13` for full results.
- ✅ **Run 11 (2026-06-05)** — synthetic streak augmentation on negative FITS tiles.
  Script: `scripts/generate_synthetic_streaks.py`. Results documented in training_methods.md.
- ✅ **Run 12 (2026-06-06)** — ViT-S heatmap at 1800 px tiles (full-frame Atwood scale).
  - 0% short-streak recall: 150 px streak ≈ 3 ViT patches at 1800 px — undetectable.
  - Norm mismatch: trained autostretch but inference defaulted to zscore.
  - Root cause documented in memory: `project_medium_streak_finding.md`.
- ✅ **Runs 13, 14, 16 — AstroPT-89M exploration (2026-06-07). Abandoned.**
  - Three configurations (1800px downscale, 256px native, 720px + zscore) all achieved 0% medium recall.
  - Root cause: AstroPT's fixed 16×16 feature grid (256 cells) cannot resolve medium-streak geometry.
  - Post-mortem: `docs/astropt_exploration_postmortem.md`.
- ✅ **Run 15 complete (2026-06-07/08)** — ViT-S heatmap at **400 px tiles, zscore norm**.
  - Fixes Run 12's 1800 px scale mismatch and autostretch norm mismatch.
  - Weights: `weights/run15_vits/best.pt`
  - Pipeline: `scripts/run15_pipeline.sh`
  - Results at t=0.85 (OBB IoU metric, val_atwood): F1=21.9%, long_recall=20.9%, medium_recall=8.3%, short_recall=0%
  - Without collinear stitch (streak-match metric): recall=84.7%, F1=22.1% at t=0.50
  - Known issue: `stitch_collinear_fragments` transitivity absorbs short streaks into long chains (ratio up to 8.5×). Fix: `max_growth_ratio=3.0` parameter added to stitch.
  - **Registered as `vits_heatmap` in pipeline.py — primary production detector.**
  - See `results/run15_vits/` for threshold sweep, eval results, streak match comparison.

## Next Steps

### Run 17 — YOLO11s-OBB detector (ready to train, 2026-06-10)

**Goal:** Train a YOLO11s-OBB streak detector on the Run 15 distribution (real Atwood
FITS + synthetic short/medium NPY) for fast single-stage inference.

**Dataset:** `/Volumes/External/TrainingData/yolo_run17_dataset/`
- 103,516 PNG tiles (400px, zscore→uint8 3ch) + DOTA polygon OBB labels
- Label format: `class x1 y1 x2 y2 x3 y3 x4 y4` (9 fields, corners normalised [0,1])
- Active yaml: `dataset_10pct_bg.yaml` — 21,432 positive + 2,143 background tiles (10% bg)
- Export script: `scripts/export_yolo_obb_dataset.py` (DOTA format, OBB clipping)

**Pretrained weights:** `weights/yolo11s-obb.pt` (move from repo root after first run)

**Training command (run from repo root):**
```bash
# Stage images to internal SSD first for speed (~10GB, fits in 100GB budget)
python scripts/stage_yolo_dataset_ssd.py   # TODO: create this helper

# Then train
nohup /Users/robert/miniconda3/envs/satid/bin/python -c "
from ultralytics import YOLO
model = YOLO('weights/yolo11s-obb.pt')
model.train(
    task='obb',
    data='/tmp/yolo_run17_ssd.yaml',      # points to /tmp/yolo_run17/images/
    imgsz=416, epochs=50, batch=4, device='mps', workers=2,
    cos_lr=True, degrees=10, hsv_s=0, hsv_h=0,
    project='weights/yolo_run17', name='run',
    exist_ok=True, patience=15, save_period=5, cache=True,
)
" > /tmp/yolo_run17_train.log 2>&1 &
```

**Critical setup notes (do not skip):**

1. **SSD staging required.** Images on external drive give ~4.5h/epoch; SSD gives ~80min/epoch.
   Staging structure must have `/images/` as a path component — ultralytics auto-detects
   labels by replacing `/images/` with `/labels/` in the path.
   Correct: `/tmp/yolo_run17/images/train/` → `/tmp/yolo_run17/labels/train/`
   Wrong: `/tmp/yolo_run17_images/train/` (labels never found, instances=0 every batch)

2. **MPS bug patch (torch 2.11).** `batch_idx.unique(return_counts=True)` on MPS returns
   garbage counts. Patch already applied to ultralytics loss.py:
   ```python
   # loss.py ~line 996 — already patched, verify before training:
   _, counts = batch_idx.cpu().unique(return_counts=True)  # .cpu(): MPS unique returns garbage
   max_count = int(counts.max().item())
   counts = counts.to(dtype=torch.int32).to(device=self.device)
   out = torch.zeros(batch_size, max_count, 6, device=self.device)
   ```
   Verify with: `grep -A3 "cpu().unique" .../ultralytics/utils/loss.py`
   If patch is missing (e.g. after ultralytics upgrade), re-apply before training.

3. **imgsz must be divisible by 32.** Use 416, not 400.

4. **workers=0 causes ~4× slowdown** vs workers=2 on MPS. Use workers=2.

**Expected performance:** ~80 min/epoch on M3 MPS with SSD staging + workers=2 + cache=True.
20 epochs ≈ 27h. Consider running 50 epochs overnight.

**Success gate:** mAP50 > 0.50 on val set. Compare short/medium/long recall against Run 15
ViT-S heatmap baseline (short=0%, medium=8.3%, long=20.9%).

**Cleanup after training:**
- Move `yolo11s-obb.pt` from repo root → `weights/yolo11s-obb.pt`
- Delete SSD staging dir `/tmp/yolo_run17/`

### Run 17 — ViT-B frozen heatmap (in progress, 2026-06-09)

**Goal:** Establish whether ViT-B's richer 768-dim features improve on ViT-S frozen
(Run 15, F1=21.9%).  All hyperparameters held constant vs Run 15; backbone size is the
only variable.

**Pipeline:** `scripts/run17_vitb_pipeline.sh`
Builds merged FITS-direct annotations via `scripts/build_run17_annotations.py`, then
caches ViT-B features at 400px tiles / zscore norm and trains the heatmap head.
Weights will land at `weights/run17_vitb/best.pt`.

**Success gate:** F1 > 21.9% AND medium_recall > 8.3% on val set (OBB IoU metric).
If ViT-B frozen does not beat ViT-S frozen, the frozen-backbone ceiling is size-independent
and backbone unfreeze (Track B) is the only remaining lever.

### Run 18 — backbone unfreeze (next after Run 17 results)

Two options depending on Run 17 outcome:
- **If ViT-B frozen beats ViT-S:** unfreeze last 2 ViT-B blocks — richer features +
  domain adaptation. Requires end-to-end training script (not cached), ~$3–5 on A10.
- **If no improvement:** unfreeze last 2 ViT-S blocks (cheaper, same domain-adaptation
  benefit). Use `train_dinov3_heatmap.py` (non-cached) with two param groups:
  backbone LR = 1e-5, head LR = 1e-4.

**Success gate (either variant):** recall ≥ 60% AND precision > 1% (OBB IoU metric).

### Stitch fix validation (pending)

The `max_growth_ratio=3.0` guard was added to `stitch_collinear_fragments` in
`inference/tiled_pipeline.py` but the post-fix threshold sweep on
`results/run15_vits/threshold_sweep_stitchfix/` was never completed because the
`t0.05_nostitch` predictions needed to finish first.

To complete this:
```bash
python scripts/run_posthoc_threshold_analysis.py \
  --predictions results/run15_vits/t0.05_nostitch/predictions.json \
  --annotations /Volumes/External/TrainingData/annotations/val_run17_fits.json \
  --stitch --stitch-max-growth-ratio 3.0 \
  --output results/run15_vits/threshold_sweep_stitchfix/
```

Compare short_recall to OBB baseline (75%) and Run 12 (0%).

### Dead detector cleanup (pending)

The following detectors were removed from active use but their code is still present:
- `_run_classical_detector` (OpenCV) — defined in `pipeline.py` but not wired into `_run_all_detectors`
- Duplicate `vits_heatmap` block in `_run_all_detectors` (merge conflict residue — the sidecar block and the non-sidecar block both exist)
- `dinov3_vits_run5` in `_model_registry()` — weights still present but detector is superseded

Clean up these before the next major feature.

### Canonical evaluation metric (applies to every run)

`eval/geometry_metrics.py` is the **standard evaluation tool for all ARGUS models**.
Run it for every trained model and save results to `results/<run_name>/geometry_eval.json`.

```bash
python -m eval.geometry_metrics \
    --predictions results/<run>/t0.50/predictions.json \
    --annotations data/annotations/val_atwood.json \
    --output results/<run>/geometry_eval.json
```

Three tiers:
- **Tier 1** — Did the model find the streak? (strict centerline match, no buffer at ends)
- **Tier 2** — Angle and endpoint accuracy of the raw model output
- **Tier 3** — Same metrics after Radon angle + endpoint refinement (pass `--images-dir`)

See `agent_docs/test_strategy.md §Canonical Model Evaluation Standard` for full details.

### Evaluation rules (apply to every heatmap eval)

**Standard heatmap training strategy (all future runs):**
- Build merged annotations with `scripts/build_run17_annotations.py` (or a successor
  script). The cache script (`cache_dinov3_heatmap_features.py`) loads FITS natively,
  so NPY pre-conversion is not required to start a run.
- **For faster future runs: pre-convert FITS to NPY once, store durably on external drive.**
  FITS loading via astropy is ~100× slower than `np.load` — the Run 17 train cache took
  ~10 hours for 5446 images from FITS vs ~1 hour from NPY. One-time conversion command:
  ```bash
  python scripts/convert_tiles_to_npy.py \
    --annotations /Volumes/External/TrainingData/annotations/all_train_run5_tiled_ts1800.json \
    --output-dir  /Volumes/External/TrainingData/argus_fits_npy/atwood_train \
    --output-ann  /Volumes/External/TrainingData/annotations/train_real_atwood_npy.json \
    --norm-mode zscore
  # repeat for val_atwood_tiled_ts1800.json → val_real_atwood_npy.json
  ```
  Store at a stable external drive path (never `/tmp`). Update `build_run17_annotations.py`
  to prefer these NPY annotations when they exist, falling back to FITS if missing.
  **Do not delete NPY tiles after a run** — they are reusable across backbone experiments.
- The Run 17 failure root cause was not NPY itself but fragile `/tmp` symlinks and
  hardcoded paths. Durable NPY on the external drive + the 0-tile abort guard is safe.
- Cache features with `--native-tile-size 400 --tile-overlap 0.0` (Run 15/17 standard).
  Each 6248×4176 Atwood image tiles into ~4–6 annotation-covered 400 px crops.
- **Always add a 0-tile abort guard** after each cache step:
  ```bash
  TILES=$(python3 -c "import json; d=json.load(open('$CACHE/manifest.json')); print(len(d['manifest']))")
  [ "$TILES" -gt 0 ] || { echo "ERROR: cache wrote 0 tiles"; exit 1; }
  ```
- Write cache to `/tmp/argus_<run>_cache/` (internal SSD, 100 GB budget).
  ViT-B features are 768-dim (2× ViT-S), so expect ~2× the cache size of a ViT-S run.
- Train with `train_dinov3_heatmap_cached.py` on the tiled cache.
- Evaluate with `--tiled` in `evaluate_dinov3_heatmap.py`.
- **Threshold sweep (standard pattern):** load the model once at `--threshold 0.05` (keeps
  all candidates) and pass `--threshold-sweep 0.2 0.3 0.4 0.5 0.6 0.7`.  The script runs
  inference once per image, then re-filters by each sweep value in memory and writes
  `metrics_t020.json`, `metrics_t030.json`, … alongside the `--output` path.
  **Never launch N parallel processes for N thresholds** — each process reloads the model.
  Example:
  ```bash
  python scripts/evaluate_dinov3_heatmap.py \
    --checkpoint weights/runN_vits/best.pt \
    --annotations data/annotations/val.json \
    --output results/runN/sweep/metrics_placeholder.json \
    --tiled --stitch --norm-mode zscore \
    --threshold 0.05 \
    --threshold-sweep 0.2 0.3 0.4 0.5 0.6 0.7
  ```
- For stitch eval: always pass `--stitch-max-growth-ratio 3.0` to prevent short streaks
  from being absorbed into long false-positive chains.
- **Heatmap cache (fastest threshold sweeps):** run `scripts/cache_heatmap_maps.py` once to
  save per-image feature-resolution probability maps (float32 NPY, ~400 KB each).
  Then pass `--heatmap-cache DIR --tiled` to skip GPU inference entirely on future sweeps —
  re-thresholding runs from cached NPY maps in seconds.
  ```bash
  # Step 1: build cache once (GPU, ~same time as a single eval)
  python scripts/cache_heatmap_maps.py \
    --annotations data/annotations/val_run17_fits.json \
    --checkpoint weights/run15_vits/best.pt \
    --output-dir /tmp/argus_run15_heatmap_cache \
    --norm-mode zscore

  # Step 2: sweep thresholds without GPU (repeatable, seconds per run)
  python scripts/evaluate_dinov3_heatmap.py \
    --annotations data/annotations/val_run17_fits.json \
    --output results/run15_vits/cache_sweep/metrics_placeholder.json \
    --tiled --stitch --stitch-max-growth-ratio 3.0 \
    --heatmap-cache /tmp/argus_run15_heatmap_cache \
    --threshold 0.50 \
    --threshold-sweep 0.60 0.70 0.80 0.85 0.90
  ```

**Full-image caching (no `--native-tile-size`) must not be used for Atwood-scale images.**
Medium streaks span <2 feature patches at full-frame 384 px — undetectable.
See Run 12 postmortem in `memory/project_medium_streak_finding.md` and `docs/training_methods.md §6`.

**FITS-domain eval only.** Do not evaluate heatmap models on the SatStreaks JPEG benchmark.
The heatmap models are trained on raw FITS (Atwood); JPEG compression changes pixel
distributions in ways that break the threshold logic.

## Dataset & File Naming Convention

**Full reference:** `agent_docs/dataset_naming.md` — read it before creating any new dataset.

**Rule:** A `runN` prefix belongs only on artifacts that are *ephemeral to that specific run*
and will not be reused — cached feature tiles, model weights, eval results.
Stable, reusable datasets get content-based names with no run number.

### Full-frame source datasets  (`annotations/` directory)

| Current file | Annotations | Canonical name (use at next edit) |
|---|---|---|
| `val_run17_fits.json` | 1 156 | `val_atwood_v2.json` ← **use this for all current val work** |
| `val_atwood.json` | 228 | `val_atwood_v1.json` (older labelling, same 240 images) |
| `test_atwood.json` | 228 | `test_atwood_v1.json` |

Training source files have not yet been migrated to canonical names.
Current run-scoped files and their canonical equivalents (create at **Run 18**, not before):

| Current file | Canonical name |
|---|---|
| `all_train_run17_merged.json` | `train_atwood_frigate_synth_v1.json` |
| `all_train_run5_tiled_ts1800.json` (de-tiled Atwood source) | `train_atwood_v1.json` |
| `frigate_streaks.json` | `train_frigate_v1.json` |
| `synth_run13_short_npy.json` | `train_synth_short_v1.json` |
| `synth_run13_medium_npy.json` | `train_synth_medium_v1.json` |

The merge script (`build_run17_annotations.py`) should become a non-run-scoped
`build_train_annotation.py` that accepts `--output` and writes to the canonical name.
Do this at Run 18 — the current run-scoped file is fine for Run 17.

### Tiled eval datasets  (self-contained directories at `/Volumes/External/TrainingData/`)

| Directory | Source | Context tiles | Total tiles | Notes |
|---|---|---|---|---|
| `val_atwood_near_ctx_t400_c4_v1/` | `val_run17_fits.json` | 4 | ~5 424 | **Current recommended fast eval** |

Build with `scripts/build_tiled_val_annotation.py`.  Evaluate **without** `--tiled`
(each NPY is already a 400-px crop).  See `agent_docs/dataset_naming.md` for full
tile-count table and rebuild command.

### Canonical NPY directories  (external drive) — synthetic streak generation only

| Directory | Contents |
|---|---|
| `argus_synth_npy/short/` | Synthetic short-streak base tiles (float32 NPY, 1800×1800) |
| `argus_synth_npy/medium/` | Synthetic medium-streak base tiles |

**NPY pre-conversion is NOT needed for heatmap feature caching.**
`cache_dinov3_heatmap_features.py` loads FITS files natively. Only
`generate_synthetic_streaks.py` requires NPY input (it reads normalised tiles to
composite synthetic streaks). `scripts/convert_tiles_to_npy.py` is therefore
**legacy for pipeline use** — keep it only to support synthetic data generation.

### Run-scoped names are correct for

- `weights/runN_*/` — model checkpoints
- `results/runN_*/` — eval metrics and threshold sweeps
- `/tmp/argus_runN_cache/` — feature tile caches (deleted after training)
- `yolo_runN_dataset/` — YOLO-format exports for a specific training run
- `all_train_runN_npy.json` etc. — intermediate merged annotations generated during a run
  (replace with canonical sources for the next run; delete when stale)

### When starting a new training run

1. Draw from canonical source annotations, not from `*_runN_*` intermediate files.
2. Name any new merged/filtered annotation after its *content*, not the run.
3. If you need to add synthetic data, create it under `argus_synth_npy/<type>/`
   and register it in a canonical annotation file.

**Legacy files** (`all_train_run13_npy.json`, `synth_run13_short_npy.json`, etc.) exist from
before this convention was established.  Do not create new files following the old pattern.
Migrate a legacy file to the canonical name the first time you need to edit it.

## Hardware
- **Dev / CI:** MacBook Air M3 — CPU or MPS. Use `MODEL_SIZE=tiny` (Swin-T).
- **Phase D Route 1:** Colleague's RTX 5070 Ti 16 GB (Windows WSL2) — CUDA. Use `MODEL_SIZE=dinov3_vitl`.
- **Phase D Route 2:** Cloud GPU rental, RTX 4090 24 GB (Vast.ai / RunPod) — CUDA. Use `MODEL_SIZE=dinov3_vitl`.
- **Phase F (if needed):** A100 80 GB cloud rental — for partial ViT-L backbone unfreeze only.
- **Rule:** Never hardcode `torch.device("cuda")`. Always call `get_device()` from
  `inference/device.py`. Code must run on CPU, MPS, and CUDA without changes.

## Read These Before Starting Any Task
Always read the relevant agent_docs file before writing code:

- `agent_docs/architecture.md`       — full system design, component map, data flow
- `agent_docs/phase1_goals.md`       — Phase 1 data pipeline (complete — reference only)
- `agent_docs/argus_phases.md`       — ARGUS Phases 2–8: model through eval
- `agent_docs/dinov3_plan.md`        — DINOv3 backbone integration plan and phase status
- `agent_docs/datasets.md`           — where to get test data, download links
- `agent_docs/dependencies.md`       — exact packages, versions, install commands
- `agent_docs/dataset_naming.md`     — naming & versioning schema for all datasets (read before creating any annotation)
- `agent_docs/test_strategy.md`      — how to measure and record baseline accuracy
- `agent_docs/spacetrack.md`         — Space-Track API policy, TLE catalog setup, rate limits
- `agent_docs/Training_Handoff.md`   — Phase D training: Route 1 (RTX 5070 Ti workstation) and Route 2 (RTX 4090 cloud rental)
- `docs/cloud_training_preparation.md` — cloud rental readiness, reproducibility checklist, run manifest, transfer/sync plan

## Stack
- Python 3.11, conda environment named `satid`
  (use `/Users/robert/miniconda3/envs/satid/bin/python`)
- Core ML: PyTorch ≥ 2.2, MMDetection ≥ 3.3 (Co-DINO)
- Astronomy: astropy, astride, sgp4, skyfield, spacetrack
- Image: opencv-python-headless<4.10, scikit-image (Radon), albumentations, Shapely
- API: FastAPI, SQLAlchemy async, asyncpg/aiosqlite, Pydantic v2
- Frontend: React 18 + Vite + Tailwind CSS
- Testing: pytest
- numpy must stay < 2.0 (sep and astride are compiled against numpy 1.x)

## Project Structure
```
Argus/
├── AGENTS.md                ← Codex compatibility shim; points here
├── CLAUDE.md                ← Claude Code compatibility shim; points here
├── README.md
├── agent_docs/              ← canonical assistant guide + reference docs
├── src/                     ← Phase 0: classical baseline (complete, do not modify)
│   ├── ingest/fits_parser.py
│   ├── detection/classical_detector.py
│   ├── astrometry/plate_solver.py
│   └── matching/            ← scorer, spacetrack_query, tle_store, spatial_filter, propagator, matcher
├── inference/               ← ML inference modules
│   ├── fits_loader.py       ← FITS→tensor, normalisation + FITS/sidecar WCS (Phase 1 ✅)
│   ├── device.py            ← get_device() helper — CPU/MPS/CUDA (Phase 2, next)
│   ├── pipeline.py          ← main inference orchestrator (Phase 2)
│   ├── postprocess.py       ← Radon angle refinement + extent tracing, NMS, grouping/fusion (Phase 3)
│   └── crossid.py           ← satellite ephemeris cross-matching (Phase 3)
├── training/                ← training data and model training
│   ├── convert_labels.py    ← OBB label format → COCO JSON (Phase 1 ✅)
│   ├── dataset.py           ← FITSStreakDataset (Phase 1 ✅)
│   ├── augmentations.py     ← albumentations pipeline + SyntheticStreakInject (Phase 1 ✅)
│   └── train_dino.py        ← Co-DINO training script + checkpoint/timebox CLI overrides (Phase 2)
├── models/
│   └── dino/                ← MMDetection configs: streak_codino_swin_t.py, _swin_l.py,
│                               streak_dinov3_vitb.py, streak_dinov3_vitl.py,
│                               dinov3_adapter.py (PatchToPyramid + DINOv3Backbone)
├── api/                     ← FastAPI application (Phase 5)
│   ├── main.py
│   ├── models.py
│   ├── storage.py           ← local / S3 swappable backend
│   └── queue.py             ← in-memory / SQS swappable backend
├── frontend/                ← React + Vite (Phase 6)
├── eval/                    ← metrics, benchmark, visualise (Phase 8)
├── db/                      ← schema.sql, migrations (Phase 4)
├── data/
│   ├── raw/                 ← original FITS files (gitignored)
│   ├── processed/           ← converted PNGs (gitignored)
│   ├── catalogs/            ← TLE catalog files
│   ├── tle_zips/            ← Space-Track annual TLE bundles (gitignored, one-time setup)
│   └── annotations/         ← COCO-format JSON label files
├── tests/                   ← pytest (mirrors src/ and top-level module layout)
├── results/                 ← baseline metrics JSON output
└── weights/                 ← model weights (gitignored)
```

## Academic Research Context
This project is academic research software. It builds on the following prior works:

- **ASTRiDE** — Automated Streak Detection for Astronomical Images
  (Kim et al., https://github.com/dwkim78/ASTRiDE)
- **StreakMind** — Co-DINO transformer-based satellite streak detection pipeline
  (prior work that ARGUS builds upon; cite per their published paper/repo)
- **Co-DINO** — Co-Deformable DETR object detection
  (Zong et al., 2023, https://arxiv.org/abs/2211.12860)
- **Danarianto et al. Prototype** — satellite identification prototype pipeline
  (Danarianto et al., cite per their published paper)

### Source Citation Rules
Whenever code directly implements, adapts, or is substantially derived from one
of the above works, add an inline citation comment at the function, class, or
code block level. Use this format:

```python
# Source: <AuthorOrProject> — <brief description of what was adapted>
# Ref: <DOI, URL, or "unpublished manuscript" as appropriate>
```

Apply citations to:
- Any algorithm, formula, or threshold copied or adapted from a prior work
- Any preprocessing step, model architecture, or scoring logic derived from a source
- Helper functions whose design is traceable to a specific paper or repo

Do **not** cite:
- Standard library usage or generic Python idioms
- astropy/sgp4/skyfield API calls that follow their own documentation
- Logic that is entirely original to this project

When in doubt, cite. Over-attribution is preferable to under-attribution in
academic research code.

## Code Standards
- Type hints on every function signature
- Google-style docstrings on every public function and class
- Every module has a `if __name__ == "__main__":` block for standalone testing
- Never hardcode credentials — use environment variables only
- Never hardcode `torch.device("cuda")` — use `get_device()` from `inference/device.py`
- All file paths via `pathlib.Path`, never raw strings
- Log with `logging` module, not `print()` (except __main__ blocks)
- Log all inference timings at DEBUG level:
  `fits_load_ms`, `inference_ms`, `postprocess_ms`, `crossid_ms`, `db_write_ms`

## Environment Variables Required
```bash
export DATABASE_URL=sqlite+aiosqlite:///./argus.db   # default

# ViT-S/16 heatmap detector (primary) — MUST match the training run's tile size and norm
export VITS_HEATMAP_CHECKPOINT=weights/run15_vits/best.pt
export VITS_HEATMAP_NORM=zscore
export VITS_HEATMAP_NATIVE_TILE_SIZE=400   # must match Run 15 cache tile size
export VITS_HEATMAP_TILE_OVERLAP=0.5
export VITS_HEATMAP_THRESHOLD=0.85

# Multi-model DINO ensemble (set to [] to use heatmap only)
export ARGUS_MODEL_CONFIGS=[]

# Default norm for CLI / direct pipeline.py calls (must match loaded weights)
export ARGUS_NORM=zscore

# Optional for cloud deployment:
export STORAGE_BACKEND=local   # or s3
export QUEUE_BACKEND=memory    # or sqs
export S3_BUCKET=
export AWS_REGION=

# Space-Track credentials — only needed for explicit catalog maintenance or
# diagnostics. Day-to-day inference never queries Space-Track directly.
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
# Dev/local maintenance calls use Space-Track's test site by default.
# Production uses the official site when ARGUS_ENV=production.
export ARGUS_ENV=development
export SPACETRACK_BASE_URL=https://for-testing-only.space-track.org/
```

## TLE Catalog Setup (one-time per environment)

The cross-identification pipeline reads TLEs from a local database table
(`tle_catalog` in `argus.db`) rather than querying Space-Track at inference
time.  This must be bootstrapped once per environment.

```bash
# 1. Download annual TLE bundle(s) from Space-Track's cloud storage:
#    https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu
#    Place file(s) in data/tle_zips/  (any .zip or .txt format is accepted)

# 2. Load into the database (idempotent — safe to re-run):
python scripts/bootstrap_tle_catalog.py --zip-dir data/tle_zips/ --years 2025
```

If `tle_catalog` does not contain records for an observation's time window,
ARGUS skips cross-identification and leaves the object unknown. It does not
fall back to broad `gp_history` or hourly `gp` calls during inference.

See `agent_docs/spacetrack.md` for full API policy details.

## Running Tests
```bash
conda activate satid
pytest tests/ -v
# All tests are offline (mocked) — no Space-Track credentials required
```

## Workflow Rules
- Complete and test each phase before starting the next
- Phase 1 gate is already cleared — valid COCO JSON ✅, FITSStreakDataset iterates ✅
- Write pytest tests alongside each module, not after
- Run pytest after every module is complete — fix failures before continuing
- Ask for a plan before writing code for any module over 100 lines
- Storage and queue backends must be swappable via env var with zero changes to
  `api/main.py` or `inference/pipeline.py`
- numpy must stay pinned < 2.0; do not upgrade albumentations or opencv past versions
  that require numpy ≥ 2.0
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in your shell when running on Mac

## Device Abstraction (`inference/device.py`)

`inference/device.py` must exist before any other ML code. It provides:

```python
def get_device() -> torch.device:
    """Priority: CUDA → MPS → CPU. Never hardcode 'cuda' anywhere."""

def get_device_config() -> dict:
    """Returns device-appropriate hyperparameters."""
```

Config values by device:

| Key | CUDA | MPS | CPU |
|-----|------|-----|-----|
| `batch_size` | 2 | 1 | 1 |
| `num_workers` | 4 | 0 | 2 |
| `pin_memory` | True | False | False |
| `image_size` | 800 | 400 | 400 |
| `mixed_precision` | True | False | False |
| `gradient_checkpointing` | True | True | True |

**MPS-specific rules** (apply everywhere):
1. `num_workers` must be 0 — always use `get_device_config()["num_workers"]`
2. Wrap AMP: `use_amp = device.type == "cuda"` then `torch.autocast(device_type=device.type, enabled=use_amp)`
3. NMS/roi_align fallback: if `device.type == "mps"`, call `.cpu()` before torchvision op then `.to(device)` after
4. `pin_memory=True` crashes on MPS — always use `get_device_config()["pin_memory"]`
5. `skimage.transform.radon` and Shapely are CPU-only — expected, do not move to MPS

## Model Size Selection

Two MMDetection configs must always exist:
- `models/dino/streak_codino_swin_t.py` — development, Mac-safe (Swin-T, ~340MB weights)
- `models/dino/streak_codino_swin_l.py` — production, cloud GPU only (Swin-L, ~2.4GB weights)

`MODEL_SIZE=large` must raise `EnvironmentError` if `device.type != "cuda"`.

`scripts/download_weights.py` — downloads Swin-T (default) or Swin-L weights based on `MODEL_SIZE`; skips if file exists. Add `weights/` to `.gitignore`.

## Dev Subset Tool (`training/make_dev_subset.py`)

50-image reproducible subset for fast local iteration:
- 20 images with no streaks, 20 with short streaks (<269px), 10 with long streaks (≥269px)
- `USE_DEV_SUBSET=true` (default) → loads `data/annotations/dev_subset.json`
- `USE_DEV_SUBSET=false` → loads full annotation file (cloud training only)

## Fast Iteration Mode

`FAST_MODE=true` or `pipeline.run(image, fast=True)`:
- Keeps Radon angle refinement; skips cross-ID and DB write
- Forces `image_size=256`
- Target: <60 seconds wall time per image on Mac

## Phase Sequencing (Hardware-Aware)

| Phase | Where | Gate |
|-------|-------|------|
| 1 — Data pipeline | Mac CPU | COCO JSON valid, Dataset iterates |
| 2 — Model config | Mac (no GPU) | Both configs pass mmdet check; Swin-T weights downloaded |
| 3 — Augmentation | Mac CPU | `augmentations.py --visualize` runs clean |
| 4 — Integration | Mac MPS, tiny | `pipeline.py --fast` <60s |
| 5 — API + Frontend | Mac CPU | API starts, frontend starts, upload curl works |
| 6 — Cloud handoff | Mac | `prepare_cloud_training.py` all checks pass |
| 7 — Cloud training | Lambda A100 | val mAP >90%, fetch weights |
| 8 — Evaluation | Mac MPS | ≥94% precision, ≥97% recall |
| DINOv3 A — Probe | Mac CPU | Cosine dissimilarity > 0.05 ✅ (0.095) |
| DINOv3 B — Adapter | Mac | MMDet configs parse, pipeline smoke test ✅ |
| DINOv3 C² — ViT-B full | Mac MPS | mAP@0.5 > Swin-T baseline ✅ (0.74 vs 0.19) |
| DINOv3 D — ViT-L full | Route 1: RTX 5070 Ti (WSL2) **or** Route 2: RTX 4090 cloud | mAP@0.5 ≥ 0.74, see Training_Handoff.md ⏳ |
| DINOv3 E — Comparison | Mac | ViT-L vs ViT-B vs Swin-T table |

## DINOv3 Training (model size `dinov3_vitb` / `dinov3_vitl`)

```bash
# Mac MPS — ViT-B frozen, dev subset (smoke test):
MODEL_SIZE=dinov3_vitb USE_DEV_SUBSET=true ARGUS_NORM=autostretch \
  python -m training.train_dino --backbone dinov3_vitb --work-dir weights/dinov3_vitb_dev

# Workstation — ViT-L frozen, full dataset (Phase D):
MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false ARGUS_NORM=autostretch \
  python -m training.train_dino --backbone dinov3_vitl --work-dir weights/run_5070ti_dinov3_vitl
```

DINOv3 weights are not downloaded by `download_weights.py`.
Copy from Mac (`weights/dinov3_vitb16_pretrain_lvd1689m*.pth` or `dinov3_vitl16_lvd1689m.pth`),
or see `scripts/download_dinov3_weights.py`.

`MODEL_SIZE=dinov3_vitl` raises `EnvironmentError` if `device.type != "cuda"`.

## Cloud Training Scripts

`scripts/prepare_cloud_training.py` — validates all checklist items (annotations, dataset, configs, augmentations, pipeline fast-mode, API, split requirements files) before GPU rental; exits 1 on any failure.

`scripts/cloud_setup.sh` — run once on Lambda instance: installs deps, downloads Swin-L weights, verifies CUDA, flips `.env` to `MODEL_SIZE=large` and `USE_DEV_SUBSET=false`.

`scripts/fetch_weights.sh <user@ip>` — rsync `weights/best.pth` and training logs back to Mac after training.

## Cost Guardrails (`training/train_dino.py`)

After epoch 1 completes, print estimated total time and Lambda cost ($1.29/hr), then `sleep(30)` before epoch 2. Ctrl+C during that window aborts the run without further charges.

## Deferred Work (stub with `raise NotImplementedError`)

Do not implement until Phase 7 weights exist:
- Multi-frame tracklet association (DB schema only)
- Swin-L → Swin-T weight distillation
