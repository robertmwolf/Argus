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
**Run 7 complete (2026-06-02). Precision catastrophe persists — val metric blind to over-prediction. Run 8 must fix val set and rebalance.**

Progress:
- ✅ Phase 0 (Classical baseline): `src/` — fits_parser, classical_detector, plate_solver, SGP4 matching
- ✅ Phase 1 (Data pipeline): `inference/fits_loader.py`, `training/convert_labels.py`, `training/dataset.py`, `training/augmentations.py`
- ✅ Phase 2 (DINO model): `inference/device.py`, `models/dino/` configs, `training/train_dino.py`
- ✅ Phase 3 (Inference pipeline): `inference/pipeline.py`, `inference/postprocess.py`, `inference/crossid.py`
- ✅ Phase 4 (Database): `db/schema.sql`, `db/models.py` — SQLAlchemy async ORM
- ✅ Phase 5 (API): `api/main.py`, `api/storage.py`, `api/queue.py`, `api/worker.py`
- ✅ Phase 6 (Frontend): `frontend/` — React 18 + Vite + Tailwind
- ✅ Phase 8 (Evaluation): `eval/metrics.py`, `eval/benchmark.py`
- ✅ DINOv3 ViT-B backbone integrated — `models/dino/dinov3_adapter.py`
- ✅ Adaptive tiling — `inference/tiled_pipeline.py`
- ✅ **Run 3 complete** — cold-start ViT-B, 15 epochs. mAP=0.782, P=94.9%, R=83.8%.
  Weights: `weights/run3_cold_nodm/best.pth`. See `docs/training_methods.md §3.2`.
- ✅ Data strategy formalised — `docs/data_strategy.md`. SatStreaks excluded from training;
  geometry-based stratification adopted; multi-scope workflow implemented.
- ✅ **Run 4 complete (2026-05-29)** — two ViT-S models on geometry-stratified Atwood + Frigate.
  - **OBB MMDet ViT-S:** `weights/run4_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth`
    Val: mAP@50=0.611. Test: mAP@50=0.518, short-recall=0%, medium-recall=49%, long-recall=75%.
  - **Centerline ViT-S:** `weights/run_dinov3_vits_orientation_centerline_1024/best.pt`
    Val dice=0.2327. See `docs/training_methods.md §3.4`.
- ✅ **Run 4 FN root-cause analysis (2026-05-30)**:
  - 45% of FNs: model detected but conf < 0.30 (confidence calibration, not visibility)
  - 55% of FNs: truly not detected (data/capacity issue)
  - SNR is NOT the primary driver — 53% of FNs are bright (SNR > 20)
- ✅ **Frigate corpus analysis (2026-05-30)**:
  - 86% of annotations are near-circular blobs (<25px, AR~1) — provide no useful training signal
  - 13% (cluster-2, ≥35px, AR≥2) are genuine linear streaks — tiled at 110px for 3.64× zoom
- ✅ **Run 5 ConvNeXt-S Stage-2 HeatMap — pretiled (2026-05-31)**:
  - Backbone: DINOv3 ConvNeXt-Small, stage 2, frozen (384ch at H/16, stride 16)
  - Training data: `all_train_run5_tiled.json` (9,570 pre-tiled 400px crops, **with OBBs**)
  - Previous full-frame run (2026-05-30) had medium-band scale mismatch; this run fixes it
  - Weights: `weights/run5_convnext_small_s2_heatmap_pretiled/best.pt`
  - **val_dice=0.918** (epoch 50/50) vs 0.842 full-frame — +0.076 improvement
  - Registered as detector `convnext_heatmap` in pipeline. See `docs/training_methods.md §3.7`
  - **TILING SCRIPTS FIXED (2026-05-31):** `build_tiled_brentimages_json.py` and
    `build_tiled_frigate_json.py` now output tile-local `obb` fields. Old tiled
    annotation files built before this date are missing OBBs and must not be used
    for heatmap training. Regenerated: `all_train_run5_tiled.json`, `val_atwood_tiled_400.json`
- ✅ **Run 5 dataset built (2026-05-30)**:
  - All 5 Atwood nights re-stratified (1,475 images → 1,129 train / 240 val / 240 test)
  - Frigate replaced with cluster-2 tiled at 110px (48 annotations, 9 frames)
  - Synthetic short-band injection: 380 images, snr_scale 0.2–1.0
  - **`all_train_run5.json`**: 2,064 images, 1,956 annotations (short=23%, medium=30%, long=47%)
- ✅ **Run 5 test-set eval (2026-06-01)** — precision catastrophe on both backbones:
  - ConvNeXt-S: recall=76.3%, precision=0.05%, F1=0.001 (1,499 predictions/image)
  - ViT-S: recall=67.5%, precision=0.05%, F1=0.001 (1,211 predictions/image)
  - Root cause: only 281/9,495 tiles (3%) were negative in training; star fields trigger FP
  - Fix: add hard-negative tiles via `--hard-neg-per-pos 5` and `--neg-tiles-per-image 50`
  - See `docs/training_methods.md §3.8` for full Run 6 plan
- ✅ **Run 6 backbone comparison (2026-06-01/02)** — ConvNeXt-S vs ViT-S, 60% negative ratio:
  - ConvNeXt-S best: val_dice=0.903 (ep 49). Test: recall=36.0%, precision=0.24%, 145 preds/img
  - ViT-S best: val_dice=0.878 (ep 50). Test: recall=21.9%, precision=0.07%, 292 preds/img
  - **ConvNeXt-S wins** backbone comparison on all metrics — proceed with ConvNeXt only
  - Precision improved 5× vs Run 5 but recall collapsed: 60% negative ratio too aggressive
  - Root cause: pos_weight=20 insufficient to counteract 60% negative tile dominance in loss
  - See `docs/training_methods.md §3.9` for Run 7 plan
- ✅ **Run 7 backbone comparison (2026-06-02)** — both backbones, 27% negatives, pos_weight=50:
  - ConvNeXt-S: val_dice=0.884, test recall=28.1%, precision=0.03%, 876 preds/img
  - ViT-S: val_dice=0.856, test recall=32.5%, precision=0.05%, 654 preds/img
  - **Worse than Run 6** — pos_weight=50 + 27% negatives caused diffuse activations
  - Root cause: val_dice blind to over-prediction (val set is 97% positive tiles)
  - ViT-S beat ConvNeXt-S in this config — backbone ranking is not stable
  - See `docs/training_methods.md §3.10` for Run 8 fix

## Next Steps

### Run 8 — Fix val metric + rebalance (top priority)

The core problem: `val_dice` on a 97%-positive val set rewards any model that activates
on streak tiles, and cannot detect over-prediction on pure star-field tiles. Every run
since Run 5 has been optimising a blind metric.

**Three-part fix:**
1. **Add negative tiles to val set** so val_dice penalises star-field FPs during training
2. **Negative ratio 40–45%** (between Run 6's 60% and Run 7's 27%)
3. **pos_weight=20** (revert from Run 7's 50; Run 6's value was correct, ratio was wrong)

```bash
# Step 1: Rebuild val with negative tiles
python scripts/build_tiled_brentimages_json.py \
  --src data/annotations/val_atwood.json \
  --out /Volumes/External/TrainingData/annotations/val_atwood_tiled_400_with_neg.json \
  --native-tile-size 400 --overlap 0.5 \
  --neg-tiles-per-image 5

# Step 2: Rebuild train at 40-45% negatives
python scripts/build_tiled_brentimages_json.py \
  --src data/annotations/all_train_run5.json \
  --out /Volumes/External/TrainingData/annotations/atwood_train_run8_tiled.json \
  --native-tile-size 400 --overlap 0.5 \
  --neg-tiles-per-image 25 \
  --hard-neg-per-pos 1

# Step 3: Merge + NPY convert + cache ConvNeXt-S features + copy to local SSD
# Step 4: Train with --pos-weight 20
# Step 5: Eval with CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=400
```

Success gate: recall ≥ 60% AND precision > 1% on test_atwood.json.

### Run 6 data prep (complete)

Run 5 revealed a precision catastrophe caused by insufficient hard negatives (only 3%
of training tiles were negative). The fix is to rebuild the tiled dataset with:
- `--hard-neg-per-pos 5`: 5 annotation-free tiles per positive image (the hardest negatives)
- `--neg-tiles-per-image 50`: 50 tiles from pure-negative images (up from 2)
- Frigate fixed: re-tile from raw source to eliminate doubly-virtual path bug

Also: Frigate cluster-2 data must be included. The inclusion was already planned but the
doubly-virtual path bug in Run 5 caused 75 Frigate tiles to be silently replaced with blank
images. This is now documented in `docs/training_methods.md §3.8` (Run 6 fix).

```bash
# Step 1: Rebuild Atwood tiled annotation with hard negatives
python scripts/build_tiled_brentimages_json.py \
  --src data/annotations/all_train_run5.json \
  --out /Volumes/External/TrainingData/annotations/atwood_train_run6_tiled.json \
  --native-tile-size 400 --overlap 0.5 \
  --neg-tiles-per-image 50 \
  --hard-neg-per-pos 5

# Step 2: Fix Frigate — build from raw source (not virtual tile paths)
python scripts/build_tiled_frigate_json.py \
  --images /Volumes/External/TrainingData/raw/frigate/raw \
  --annotations <frigate_cluster2_annotations.json> \
  --output /Volumes/External/TrainingData/annotations/frigate_tiled_run6.json \
  --tile-size 400 --overlap 0.5

# Step 3: Merge + NPY convert → all_train_run6_tiled_npy.json
# Step 4: Train both backbones identically (see docs/training_methods.md §3.8)
```

### Run 6 backbone comparison

Train ConvNeXt-S and ViT-S with **identical hyperparameters** on `all_train_run6_tiled_npy.json`.
The backbone that achieves better precision at ≥60% recall on `test_atwood.json` is progressed
to ViT-B / ViT-L for the paper run.

Success gate: precision > 10% at recall ≥ 60%.

### Evaluation rules (apply to every heatmap eval)

**Standard heatmap training strategy (all future runs):**
- Cache features with `--native-tile-size 400 --tile-overlap 0.5` on the full-frame
  annotation file (e.g. `all_train_run5.json`). This tiles each 6248×4176 Atwood image
  into ~4–6 annotation-covered 400px crops — same scale as the OBB training data.
- Write cache to external drive: `/Volumes/External/TrainingData/heatmap_cache/<run>/`
- Train with `train_dinov3_heatmap_cached.py` on the tiled cache.
- Evaluate with `--tiled` in `evaluate_dinov3_heatmap.py`.

**Full-image caching (no `--native-tile-size`) must not be used for Atwood-scale images.**
Medium streaks span <2 feature patches at full-frame 384px — blob detections, not
thin-line OBBs. The eval script warns but does not abort. See `docs/training_methods.md §6`.

Applies to: `scripts/evaluate_dinov3_heatmap.py` for any checkpoint with
`cache image_size < 600 px`.

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
export MODEL_SIZE=tiny     # tiny=Swin-T (dev/MPS), large=Swin-L (A100),
                           # dinov3_vitb=ViT-B/16 (Mac MPS), dinov3_vitl=ViT-L/16 (GPU)
export ARGUS_NORM=zscore   # zscore (Swin-T/L) or autostretch (DINOv3 ViT-B/L)
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
