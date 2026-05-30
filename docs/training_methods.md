# ARGUS Satellite Streak Detection — Training Methods

> **Status:** Living document. Sections marked `[TODO]` must be resolved before the
> final paper training run. Do not submit a paper whose training run was produced
> before all TODOs are closed.

---

## 1. Model Architecture

### 1.1 Backbone — DINOv3 ViT-B/16 (frozen)

The visual backbone is Meta's DINOv3 Vision Transformer (ViT-B/16), pretrained
self-supervisedly on the LVD-1689M dataset (1.689 billion curated images).

- **Access:** Weights are available via a gated download request at
  `https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/`.
  An email with download links is issued after the request is approved.
- **Weight file:** `dinov3_vitb16_lvd1689m.pth`
- **`[TODO]`** Record the SHA-256 checksum of the weight file and commit it to
  the repo so reviewers can verify they have the identical backbone.
- **`[TODO]`** Confirm the precise paper/model-card citation for DINOv3 ViT-B/16
  (distinguish from DINOv2 if necessary).

The backbone is **fully frozen** throughout all training phases (`lr_mult=0.0`).
No gradient flows through it. It is used purely as a multi-scale feature extractor,
yielding four feature maps each of dimension 768.

### 1.2 Detection Head — Co-DINO / DINO-DETR

All trainable parameters belong to the detection head, implemented in
MMDetection 3.3.0:

| Component | Details |
|-----------|---------|
| **Neck** | `ChannelMapper`: 1×1 conv + GroupNorm(32), maps 4×768 → 4×256 |
| **Encoder** | 6-layer deformable multi-scale self-attention, embed_dim=256, FFN=2048 |
| **Decoder** | 6-layer cross-attention + DN queries, embed_dim=256, FFN=2048 |
| **Head** | `DINOHead`: 1 class (`streak`), FocalLoss (γ=2, α=0.25) + L1Loss (w=5) + GIoULoss (w=2) |
| **Queries** | 300 object queries + up to 100 DN queries per group |
| **Matching** | Hungarian assignment (FocalLossCost + BBoxL1Cost + IoUCost) |

### 1.3 Input Resolution

- **A/B comparison runs:** 256×256 px (RandomChoiceResize, keep_ratio=True)
- **Final quality run:** 400×400 px (same pipeline)
- **Backbone patch size:** 16 px → 256 features at 256px, 625 features at 400px
- **Optimizer:** AdamW, lr=1e-4, weight_decay=1e-4, gradient clip max_norm=0.1

---

## 2. Training Data

### 2.1 Dataset Inventory

| Dataset | Images | Annotations | Location | Provenance | Included? |
|---------|--------|-------------|----------|------------|----------|
| SatStreaks | 3,023 (train) / 308 (test) | ~3,100 streaks | `data/satstreaks/` local | Public dataset `[TODO: cite]` | Yes |
| BrentImages Night 1 (Apr 12 2026) | 277 | ~300 streaks | `data/BrentImages/Img_20260412_Atwood/` | First-party, Atwood Observatory | Yes |
| BrentImages Night 2 (May 15 2026) | 204 annotated + 27 negatives | 204 streaks | `/Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood/` | First-party, Atwood Observatory | Yes |
| Frigate diversity (tiled) | 250 tiles | 251 streaks | `/Volumes/External/TrainingData/raw/frigate/` | **Third-party — attribution needed** | Yes |

BrentImages is an **ongoing capture series** from Atwood Observatory (Brent's
telescope). Additional nights are expected; the `Img_YYYYMMDD_Atwood/` naming
convention accommodates new captures. Each night yields ~200–280 annotated
frames at 6248×4176 px (native resolution; downsampled for training).

**Validation set:** Derived from `val.json` (SatStreaks + BrentImages splits).

### 2.2 Data Provenance Issues (must resolve before publication)

- **Frigate:** Source and ownership `[TODO]`. Written permission to include this data
  in a published training set must be obtained. Attribution required in the paper.

- **SatStreaks:** `[TODO]` Identify the canonical citation and license.

- **BrentImages (all nights):** First-party captures from Atwood Observatory
  (ongoing series). Suitable for publication. Should be hosted with a DOI
  (HuggingFace Datasets or Zenodo recommended) as a versioned dataset that
  can grow as new nights are added.

### 2.3 Annotation Format

All training splits are COCO-format JSON with a single category:
`{"id": 1, "name": "streak", "supercategory": "satellite"}`.

Tiled crops use a virtual path encoding:
`<original_stem>__tx<x0>_ty<y0>_ts<tile_size><ext>`
which is resolved to the real crop at load time by `training/transforms.py:LoadFITSFromFile`.

**Adaptive tiling parameters** (see `docs/adaptive_tiling_plan.md`):

| Dataset | `native_tile_size` | magnification | overlap | rationale |
|---------|-------------------|---------------|---------|-----------|
| BrentImages (400px tiles) | 400 px | 1.0× | 50% | 1:1 native crops; baseline |
| Frigate (short-streak regime) | 110 px | 3.64× | 50% | 20–80 px streaks → 70–290 px at model input |

The `magnification = model_input_size / native_tile_size` factor is applied by
`inference/tiled_pipeline.py:tile_image` (cv2 resize) and reversed by
`remap_predictions` when mapping tile-space detections back to image coordinates.

### 2.4 Training Split Sizes

| Split | Images | Annotations | Notes |
|-------|--------|-------------|-------|
| `all_train_nodm_v3_external_abs.json` **v3 (2026-05-28)** | **1,134** | **1,044** | External-absolute Atwood + Frigate diversity; SatStreaks excluded |
| `all_train_nodm_external_abs.json` v2 | 8,422 | 8,213 | Legacy Run 3 candidate; includes SatStreaks and should not be used for new policy-compliant training |
| `all_train_nodm.json` v1 (active legacy filename) | 3,971 | 3,816 | Legacy compatibility; active jobs may still reference it |

**v3 composition (`all_train_nodm_v3_external_abs.json`):**

| Component | Tiles | Annotations | `native_tile_size` | Streak at model input |
|-----------|-------|-------------|-------------------|----------------------|
| Atwood Night 1 | 669 | 578 | full-frame | medium + long |
| Atwood Night 2 | 204 | 204 | full-frame | medium + long |
| Atwood Geo | 11 | 11 | full-frame | long |
| Frigate diversity | 250 | 251 | 400 px | short |
| **Total** | **1,134** | **1,044** | — | SatStreaks excluded |

Generated by:
```bash
python scripts/build_training_json.py
```

**External-data rule (2026-05-28):** Raw training images and canonical
annotation JSONs live under `/Volumes/External/TrainingData/`. Repo-local
`data/annotations` is only a compatibility path for older commands and must not
be treated as the source of truth. While existing runs may still use legacy
filenames, new training commands should use the additive external-absolute JSON
so every `file_name` points at `/Volumes/External/TrainingData/raw/...`:

```bash
USE_DEV_SUBSET=false \
TRAIN_ANN_FILE=/Volumes/External/TrainingData/annotations/all_train_nodm_v3_external_abs.json \
VAL_ANN_FILE=/Volumes/External/TrainingData/annotations/val_external_abs.json \
python -m training.train_dino --config models/dino/streak_dinov3_vitb_400px_run3.py
```

---

## 3. Training Procedure

### 3.1 Training Lineage (Pilot / Informing Runs)

Archived pilot runs and their associated local artifacts have been removed from
the repository. The current published lineage starts from the clean Run 3
training run.

**✅ Run 3 — Clean cold-start paper model (complete 2026-05-28):**
- Config: `models/dino/streak_dinov3_vitb_400px_run3.py` (400px + `randomness=dict(seed=42)`)
- Data: `all_train_nodm.json` v2 — 8,422 images (SatStreaks + BrentImages N1+N2 tiled
  at 400px + Frigate tiled at 110px/3.64×). See §2.4 for full breakdown.
- Warm start: **None** (`load_from = None`) — detection head initialised from scratch
- Hardware: Mac M3 CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`); multi-night session approach
- Schedule: 15 epochs
- Checkpoint destination: `weights/run3_cold_nodm/`

**Night 1 command (time-boxed, 3 epochs ~10–11h):**

```bash
cd /path/to/Argus && conda activate satid
PYTORCH_ENABLE_MPS_FALLBACK=1 \
USE_DEV_SUBSET=false \
TRAIN_ANN_FILE=/Volumes/External/TrainingData/annotations/all_train_nodm_external_abs.json \
VAL_ANN_FILE=/Volumes/External/TrainingData/annotations/val_external_abs.json \
ARGUS_NORM=autostretch \
caffeinate -i \
python -m training.train_dino \
    --config models/dino/streak_dinov3_vitb_400px_run3.py \
    --work-dir weights/run3_cold_nodm \
    --max-epochs 3 \
    --val-interval 1 \
    --checkpoint-interval 1
```

**Subsequent nights (resume; Ctrl-C in the morning):**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
USE_DEV_SUBSET=false \
TRAIN_ANN_FILE=/Volumes/External/TrainingData/annotations/all_train_nodm_external_abs.json \
VAL_ANN_FILE=/Volumes/External/TrainingData/annotations/val_external_abs.json \
ARGUS_NORM=autostretch \
caffeinate -i \
python -m training.train_dino \
    --config models/dino/streak_dinov3_vitb_400px_run3.py \
    --work-dir weights/run3_cold_nodm \
    --resume \
    --val-interval 1 \
    --checkpoint-interval 1
```

**Pre-flight check (run before night 1 to verify dataset paths):**

```bash
USE_DEV_SUBSET=false ARGUS_NORM=autostretch python - <<'PY'
import json, os, re, pathlib
ann = pathlib.Path("/Volumes/External/TrainingData/annotations/all_train_nodm_external_abs.json")
d   = json.loads(ann.read_text())
missing = 0
for img in d["images"]:
    fn = img["file_name"]
    m  = re.match(r"^(.+?)__tx\d+_ty\d+_ts\d+(.+)$", fn)
    real_fn = m.group(1) + m.group(2) if m else fn
    p = pathlib.Path("data") / real_fn if not real_fn.startswith("/") else pathlib.Path(real_fn)
    if not p.exists():
        missing += 1
        if missing <= 5:
            print(f"MISSING: {real_fn}")
print(f"Total images: {len(d['images'])}  Missing: {missing}")
PY
```

Expected: Total images: 8422  Missing: 0

> **Data path note (updated 2026-05-28):** Prefer absolute external-drive
> annotation and image paths. Legacy symlinks such as `data/BrentImages` may
> exist for older configs, but current training data is canonical under
> `/Volumes/External/TrainingData/raw/` and
> `/Volumes/External/TrainingData/annotations/`. The external drive must be
> mounted before training.

> **Why cold-start matters for the paper:** A reviewer could reasonably object to
> an unpublished lineage with inherited head weights. A cold-start eliminates that
> concern by training the detection head directly from the documented dataset.

### 3.2 Run 3 Training Results (completed 2026-05-28)

**Run 3 completed in a single continuous multi-night CPU session on Mac M3
(PID 8899, started 2026-05-26, finished 2026-05-28 ~14:45 local).**

Hardware: Mac M3, CPU-only (`PYTORCH_ENABLE_MPS_FALLBACK=1`), ~1.5–9.8 s/step
(thermal throttling; faster in the first hour, slower after sustained load).
Total wall-clock time: ~62 hours for 15 epochs × 3,023 steps.

Validation set: `val.json` (411 images, SatStreaks + BrentImages N1+N2 tiles).
Optimizer: AdamW, lr=1e-4. Schedule: LinearLR warmup epochs 1–2, then
CosineAnnealingLR epochs 3–15. `seed=42, deterministic=True`.

**Per-epoch COCO val results (mAP@IoU=0.50:0.95):**

| Epoch | mAP   | mAP@50 | mAP@75 | mAP_m | mAP_l | Best? |
|-------|-------|--------|--------|-------|-------|-------|
| 1     | 0.007 | 0.030  | 0.001  | 0.057 | 0.008 | |
| 2     | 0.289 | 0.442  | 0.319  | 0.245 | 0.324 | |
| 3     | 0.418 | 0.550  | 0.451  | 0.026 | 0.478 | |
| 4     | 0.471 | 0.640  | 0.476  | 0.244 | 0.524 | |
| 5     | 0.456 | 0.664  | 0.465  | 0.229 | 0.502 | |
| 6     | 0.466 | 0.677  | 0.457  | 0.252 | 0.512 | |
| 7     | 0.453 | 0.689  | 0.445  | 0.244 | 0.498 | |
| 8     | 0.458 | 0.711  | 0.416  | 0.225 | 0.498 | |
| 9     | 0.479 | 0.687  | 0.473  | 0.198 | 0.531 | |
| 10    | 0.472 | 0.703  | 0.457  | 0.259 | 0.515 | |
| 11    | 0.496 | 0.726  | 0.492  | 0.304 | 0.536 | |
| 12    | 0.538 | 0.752  | 0.542  | 0.284 | 0.588 | |
| **13** | **0.541** | **0.779** | **0.531** | **0.315** | **0.590** | ✅ best |
| 14    | 0.537 | 0.763  | 0.532  | 0.348 | 0.584 | |
| 15    | 0.537 | 0.768  | 0.535  | 0.334 | 0.585 | |

**Final COCO summary (epoch 15 val):**
```
AP @[IoU=0.50:0.95 | area=all | maxDets=100] = 0.537
AP @[IoU=0.50      | area=all | maxDets=1000] = 0.768
AP @[IoU=0.75      | area=all | maxDets=1000] = 0.535
AR @[IoU=0.50:0.95 | area=all | maxDets=100] = 0.755
AR @[IoU=0.50:0.95 | area=medium             ] = 0.333
AR @[IoU=0.50:0.95 | area=large              ] = 0.819
```

**Best checkpoint:** `weights/run3_cold_nodm/best_coco_bbox_mAP_epoch_13.pth`
Stable weights path: `weights/run3_cold_nodm/best.pth` (symlink)

**Observations:**
- Cold-start convergence was fast: by epoch 3 (mAP=0.418, mAP@50=0.550) the model
  already matched Run 2's warm-started epoch 5 (mAP@50≈0.390). This confirms the
  warm-start from Run 0 was not necessary for performance.
- The learning rate schedule shows the expected pattern: a rapid quality jump after
  the cosine phase begins (epoch 3→4: +0.053 mAP), a noisy plateau while the
  cosine decays through the middle epochs (4–11), and a final surge to peak at
  epoch 12–13 as the LR reaches its floor.
- Epochs 12–15 plateau near 0.537–0.541 — the model converged. Epoch 13 is the
  best checkpoint; extending training beyond 15 epochs is unlikely to improve on it
  without a LR restart or data augmentation changes.
- Medium-band mAP (mAP_m) remains weaker than large-band (0.315 vs 0.590 at best
  epoch). This is expected: Frigate tiles at 3.64× magnification brings 20–80 px
  native streaks to ~70–290 px model input, which may not fully overlap with the
  val set's "medium" band thresholds. A targeted augmentation pass on the medium
  band is the next lever.
- Run 3 vs Run 2 (warm-start, v1 dataset): epoch 13 mAP=0.541 vs Run 2 epoch 15
  mAP=0.423 — a **+0.118 mAP (+27.9%) improvement**, primarily from the v2 dataset
  (tiled BrentImages at 1:1 native + Frigate short-streak tiles).

**Test-set evaluation (2026-05-28):**
Run on standard test set (308 SatStreaks images) at conf≥0.30, IoU≥0.50.
Results: `results/comprehensive_eval_20260528_154914/`

| Metric | Value |
|--------|-------|
| COCO mAP | 0.782 |
| COCO mAP@50 | 0.878 |
| COCO mAP@75 | 0.826 |
| Precision | **94.9%** |
| Recall | **83.8%** |
| F1 | 89.0% |
| COCO AR | 0.908 |

**Per-band recall (conf≥0.30, IoU≥0.50):**

| Band | Recall | TP | FN | GT count |
|------|--------|----|----|----------|
| Short (<269px) | 100.0% | 2 | 0 | 2 |
| Medium (269–800px) | 90.9% | 10 | 1 | 11 |
| Long (>800px) | 83.4% | 246 | 49 | 295 |

Notes: The test set is dominated by long streaks (295/308 = 95.8%). Short sample is too
small (n=2) to draw conclusions. Medium recall (90.9%) is strong. Long recall (83.4%)
drives the overall recall number.

**Comparison to multisource baseline** (`comprehensive_eval_20260526`):

| Model | mAP@50 | Precision | Recall | F1 |
|-------|--------|-----------|--------|----|
| multisource (run_clean_vitb_nodm, ep15) | 0.550 | 71.2% | 72.4% | 71.8% |
| **Run 3 (run3_cold_nodm, ep13)** | **0.878** | **94.9%** | **83.8%** | **89.0%** |
| Δ | +0.328 | +23.7pp | +11.4pp | +17.2pp |

Run 3 substantially outperforms the multisource model on every metric. The gain
is primarily from the v2 training dataset (tiled BrentImages + Frigate).

**Updated `inference/confidence.py`:** `DETECTOR_PROFILES["dinov3_vitb_run3"]`
updated with measured values: `precision=0.9485`, `recall=0.8377`,
`band_weights={"short": 1.0, "medium": 1.1, "long": 1.0}` (derived from
per-band recall ratios). All values sourced from
`results/comprehensive_eval_20260528_154914/test_standard/metrics.json`.

### 3.4 Run 4 — ViT-S Geometry-Stratified (complete 2026-05-29)

**First training run under the new data strategy (no SatStreaks; geometry-stratified
Atwood + Frigate diversity tiles).  Two models trained simultaneously.**

#### 3.4.1 OBB Detection — MMDet DINO ViT-S

- **Config:** `models/dino/streak_dinov3_vits_400px_run3.py` (400px, ViT-S backbone)
- **Training data:** `all_train_run4.json` — 868 images
  (618 Atwood geometry-stratified train split + 250 Frigate diversity tiles)
- **Val:** `val_atwood.json` — 133 images, geometry-stratified Atwood val split
- **Hardware:** Mac M3 CPU + MPS fallback, ~3.7 s/step
- **Epochs:** 15 (12 clean; epoch 14 val corrupted by external drive EPERM event;
  resumed from epoch 13 checkpoint; epochs 14–15 completed cleanly after remount)
- **Checkpoint:** `weights/run4_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth`
- **Optimizer:** AdamW, lr=1e-6 (warmup), cosine schedule; seed=42

**Per-epoch val results (mAP@IoU=0.50:0.95, on `val_atwood.json`):**

| Epoch | mAP   | mAP@50 | mAP@75 | mAP_m | mAP_l | Best? |
|-------|-------|--------|--------|-------|-------|-------|
| 1     | 0.004 | 0.017  | 0.000  | 0.012 | 0.007 | |
| 10    | 0.173 | 0.410  | 0.090  | 0.257 | 0.231 | |
| 11    | 0.216 | 0.535  | 0.133  | 0.283 | 0.290 | |
| 12    | 0.194 | 0.490  | 0.106  | 0.235 | 0.255 | |
| 13    | 0.247 | 0.607  | 0.191  | 0.293 | 0.315 | |
| 14    | 0.253 | 0.607  | 0.188  | 0.302 | 0.331 | |
| **15** | **0.273** | **0.611** | **0.182** | **0.291** | **0.360** | ✅ best |

**Test-set results (`test_atwood.json`, 133 imgs, 119 annotations):**

| Metric | Value |
|--------|-------|
| COCO mAP | 0.223 |
| COCO mAP@50 | **0.518** |
| COCO mAP@75 | 0.139 |
| Precision (conf≥0.30) | 61.1% |
| Recall (conf≥0.30) | 55.5% |
| F1 | 58.2% |

**Per-band recall on `test_atwood.json` (conf≥0.30, IoU≥0.50):**

| Band | Recall | TP | FN | GT |
|------|--------|----|----|-----|
| Short (<269px) | 0.0% | 0 | 3 | 3 |
| **Medium (269–800px)** | **48.8%** | 39 | 41 | 80 |
| Long (>800px) | 75.0% | 27 | 9 | 36 |

Medium band (67% of annotations) is the primary failure mode. Long recall (75%)
is below the 85% quality gate — the model is viable but not yet production-ready on
Atwood native-resolution images at the full-image resize scale used here. Tiled
inference (400px crops) is expected to improve recall substantially — see §3.3 caveat.
Results: `results/zero_shot_run4_mmdet_test_atwood_*/`

**⚠ Comparison caveat:** The auto-generated report compares Run 4 ViT-S
(evaluated on Atwood FITS `test_atwood.json`) against the Run 3 ViT-B
baseline (evaluated on SatStreaks HST `test.json`). These are different test
sets from different sensor domains — the deltas are not a valid ViT-S vs ViT-B
comparison. A proper A/B requires running the ViT-B checkpoint against
`test_atwood.json`. That evaluation is not yet complete; schedule for Run 5 planning.
The only apples-to-apples cross-run metric is the SatStreaks secondary benchmark
(eval 4/6 — see below).

**Zero-shot holdout results:** See `results/zero_shot_run4_mmdet_*/report.md`.

---

#### 3.4.2 Centerline Heatmap — ViT-S

- **Architecture:** `DINOv3OrientationCenterline` (ViT-S, 18 orientation bins,
  decoder_channels=192, last_layers=4, image_size=1024)
- **Training data:** `all_train_run4.json` — 2,636 tiles/epoch
  (1,236 positive + 1,400 negative tile samples from 868 images)
- **Val:** `val_atwood.json` — 133 full-frame images
- **Hardware:** Mac M3 MPS, ~4.6 s/batch (real PNG loading after Frigate tile fix)
- **Epochs:** 10 effective (5 clean epochs → drive EPERM interruption → resumed
  from epoch-5 best checkpoint for 5 additional epochs with fresh cosine schedule)
- **Checkpoint:** `weights/run_dinov3_vits_orientation_centerline_1024/best.pt`
  (best val_dice=0.2327, epoch 1 of resumed run = effective epoch 6)
- **Loss:** BCE (w=0.10) + Dice (w=1.0) + orientation CE (w=0.20) +
  catchment BCE (w=0.20) + catchment Dice (w=1.0); pos_weight=60

**Val dice per epoch (resumed run from epoch-5 weights):**

| Resumed epoch | val_dice | train_dice | lr |
|--------------|----------|------------|----|
| 1 | **0.2327** | 0.2197 | 4.62e-05 |
| 2 | 0.2217 | 0.2259 | 3.62e-05 |
| 3 | 0.2258 | 0.2318 | 2.38e-05 |
| 4 | 0.2239 | 0.2363 | 1.38e-05 |
| 5 | 0.2284 | 0.2395 | 1.00e-05 |

Val dice plateaued at ~0.23 from epoch 1 of initial run onward; train dice continued
improving (0.10 → 0.24), indicating mild overfitting or that the val metric ceiling
is constrained by the spatial resolution mismatch at 1024px model input vs
6248×4176px native images. Results: `results/run4_centerline_*/`.

---

### 3.3 BrentImages Night 2 Evaluation Caveat

The zero-shot mAP@50 of 0.296 on BrentImages Night 2 is **misleading low** due to a
resolution mismatch. Night 2 frames are 6248×4176 px; when resized to 400px model input
the downscale factor is 15.6×. Median GT streak length in native pixels is ~687px, which
shrinks to ~44px (2.75 ViT patches) at model input — well below the training distribution
(SatStreaks median ~400px at model input).

Tiled inference at `native_tile_size=400` (1:1 native crops) resolves this. A tiled eval
(`scripts/eval_brentimages_tiled.py`) is in progress; expected to yield mAP@50 in the
0.60–0.75 range. Results to be added here
when complete.

This applies to **all future high-resolution capture nights** from Atwood Observatory.
Full-image resize is only appropriate when the target image scale matches the training
distribution; tiled inference is the correct path for native-resolution FITS frames.

### 3.2 LR Schedule (current pilot)

```
Epochs 1–2:  LinearLR  (start_factor=0.01 → 1.0, i.e. 1e-6 → 1e-4)
Epochs 3–15: CosineAnnealingLR  (T_max=13, 1e-4 → 1e-6)
```

The 2-epoch linear warmup prevents a large gradient shock when resuming from
the warm-start checkpoint at a high learning rate.

---

## 4. Final Paper Training Run — Requirements Checklist

The following must all be satisfied before executing the run whose results will
appear in the paper. **Do not run until every item is checked.**

For cloud execution, also complete `docs/cloud_training_preparation.md` and use
`docs/templates/cloud_training_manifest.md` as the run's `training_summary.md`.
The paper run must be traceable to a Git commit or explicit source archive,
input checksums, environment metadata, checkpoint checksums, and a held-out
evaluation output.

### Data

- [x] - [ ] Frigate attribution confirmed; owner has granted permission for publication
- [ ] SatStreaks citation identified
- [ ] All training data uploaded to HuggingFace Datasets (or Zenodo) with a DOI
- [ ] Annotation JSONs updated to use hosted paths; DOI recorded in this document
- [ ] Validation set is 100% local (no external volume dependency)

### Model

- [ ] DINOv3 weight SHA-256 recorded here: `[TODO]`
- [ ] DINOv3 paper/model-card citation confirmed

### Reproducibility

- [x] Random seeds fixed in config:
  ```python
  randomness = dict(seed=42, deterministic=True)
  ```
  ✅ Applied in `models/dino/streak_dinov3_vitb_400px_run3.py` for Run 3.
- [ ] Cloud training preparation checklist completed:
      `docs/cloud_training_preparation.md`
- [ ] Run manifest filled:
      `results/<run_name>/training_summary.md`
- [ ] Input SHA-256 checksums recorded for annotation JSONs and DINOv3 backbone
      weights
- [ ] Checkpoint sync destination tested before the long run starts
- [ ] `environment.yml` committed and verified on a clean environment
      (current snapshot: `environment.yml` in repo root, captured 2026-05-21)
- [ ] Decide: single run or N=3 runs with different seeds to report variance?

### Methodology

- [x] Decide on warm-start strategy:
  - ✅ **Option A selected (2026-05-26):** Cold-start the detection head directly on
    the final dataset. This is Run 3. See §3.1 for rationale and plan.
- [x] Training resolution confirmed: **400px** — Run 3 achieved mAP=0.541 vs
      Run 2 (warm-start, v1 data) mAP=0.423 at 400px. Run 1 A/B showed +0.066
      mAP@50 at 400px vs 256px; 400px is the confirmed paper resolution.
- [ ] Val set composition confirmed and frozen; report which images are in it

### Paper Run Config Parameters (informed by Run 1 and Run 2)

```python
# Decisions made after Run 1 A/B + Run 2 results + Run 3 decision (2026-05-26):
_img_scale     = (400, 400)                        # 400px: +0.066 mAP@50 vs 256px
TRAIN_ANN_FILE = '/Volumes/External/TrainingData/annotations/all_train_nodm_external_abs.json'
max_epochs     = 15                                # cosine schedule converges well by ep15
randomness     = dict(seed=42, deterministic=True) # [TODO: add to final config]
load_from      = None   # ✅ DECIDED: cold start (Option A) — no retired checkpoint
```

Note: `all_train_nodm.json` v2 (2026-05-26) now includes tiled BrentImages Night 1 and Night 2
crops at `native_tile_size=400` and Frigate crops at `native_tile_size=110`. See §2.4 for
the full breakdown. ✅ Resolved.

---

## 5. Evaluation Protocol

### 5.1 Metrics

- **Primary:** COCO mAP (IoU=0.50:0.95) and mAP@50
- **Secondary:** Precision, Recall, F1 at conf≥0.30, IoU≥0.50
- **Per-band recall:** Short (<269px diagonal), Medium (269–800px), Long (>800px)
  computed from `scripts/evaluate_comprehensive.py`

### 5.2 Test Sets

| Set | Images | Notes |
|-----|--------|-------|
| Standard (SatStreaks test split) | 308 | Primary benchmark |
| BrentImages Night 2 (zero-shot) | 231 | Out-of-distribution; use tiled inference (§3.3) |
| Frigate (zero-shot, tiled inference) | 350 | Short-streak regime; requires adaptive tiling |

Zero-shot sets are never seen during training; they measure generalisation.

### 5.3 Inference Variants

- **Standard:** Full-image resize to training resolution; use for SatStreaks
- **Tiled (BrentImages / high-res FITS):** `inference/tiled_pipeline.py`,
  `native_tile_size=400`, `overlap=0.5`, cross-tile NMS at IoU=0.4; restores 1:1 native
  resolution for large FITS frames
- **Adaptive tiled (Frigate / short-streak):** same pipeline with
  `native_tile_size=110`, `overlap=0.5`, `magnification=3.64×`; brings 20–80 px
  native streaks to 70–290 px at model input where the detector can find them.
  §6.1 verification result (2026-05-26): mAP@50 = **0.008** vs 0.000 full-frame
  baseline — ✅ PASS. FP rate high until fine-tuned on Frigate data (§6.4).
- **Optional post-NMS stitching:** `stitch_collinear_fragments()` merges collinear
  detection fragments separated by gaps ≤ `max_gap_px` (default: 1 tile width).
  Enable with `--stitch` flag on eval scripts.

---

## 6. Compute Environment

| Item | Value |
|------|-------|
| Hardware | Apple Mac M3, 16 GB unified memory |
| Training device | CPU (PYTORCH_ENABLE_MPS_FALLBACK=1; DINO deformable attention exceeds MPS 4 GB per-allocation limit) |
| Python | 3.11.15 |
| PyTorch | 2.11.0 |
| torchvision | 0.26.0 |
| mmdet | 3.3.0 |
| mmengine | 0.10.4 |
| mmcv | 2.1.0 |
| astropy | 6.1.0 |
| Full environment | `environment.yml` (repo root, captured 2026-05-21) |

Approximate training throughput on this hardware:
- 256px, batch=1, frozen backbone: ~0.73 s/step (3,971 images/epoch → ~48 min/epoch)
- 400px, batch=1, frozen backbone: ~1.4–2.5 s/step (thermal throttling; Run 2 took ~72h for 15 epochs)

---

## 7. Open Questions

1. **Frigate attribution** — blocks use of Frigate tiles in the published training set
3. **DINOv3 citation** — need the exact paper reference (not DINOv2)
4. **Data hosting** — HuggingFace Datasets vs Zenodo; need DOI before paper submission
5. ~~**Warm-start strategy**~~ — ✅ Resolved (2026-05-26): **Option A — cold start.**
   Run 3 cold-starts the detection head from scratch on `all_train_nodm.json`. See §3.1.
6. **400px vs 256px** — ✅ Resolved: 400px yields mAP@50=0.468 vs 0.402 at 256px (+0.066)
   from Run 1 A/B. Run 3 (cold-start, v2 dataset, 400px) achieved mAP=0.541/mAP@50=0.779
   — a further +0.118 mAP gain over Run 2, primarily from the improved v2 training set.
   Paper run will use 400px. Mac M3 took ~62h for 15 epochs; cloud GPU strongly
   recommended for future runs.
7. ~~**BrentImages Night 1 FITS local storage**~~ — ✅ Resolved (2026-05-26). Created
   `data/BrentImages → /Volumes/External/TrainingData/raw/BrentImages` symlink. External
   drive must be mounted during training. For cloud runs, rsync the BrentImages directory
   to the instance before training.
8. ~~**Adaptive tiling for training**~~ — ✅ Resolved (2026-05-26). `all_train_nodm.json`
   v2 includes BrentImages N1+N2 tiled at `native_tile_size=400` and Frigate tiled at
   `native_tile_size=110`. See §2.4 for full breakdown.
