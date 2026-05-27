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

| Dataset | Images | Annotations | Location | Provenance | DM-free? |
|---------|--------|-------------|----------|------------|----------|
| SatStreaks | 3,023 (train) / 308 (test) | ~3,100 streaks | `data/satstreaks/` local | Public dataset `[TODO: cite]` | Yes |
| BrentImages Night 1 (Apr 12 2026) | 277 | ~300 streaks | `data/BrentImages/Img_20260412_Atwood/` | First-party, Atwood Observatory | Yes |
| BrentImages Night 2 (May 15 2026) | 204 annotated + 27 negatives | 204 streaks | `/Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood/` | First-party, Atwood Observatory | Yes |
| ~~DarkMatters (DM)~~ | ~~149 training + 66 val holdout~~ | ~~216 streaks~~ | ~~`/Volumes/External/DarkMatters/`~~ | **Excluded — not used in this project** | N/A |
| Frigate (tiled) | 558 positive tiles + 159 neg | 655 streaks | `/Volumes/External/frigate/` | **Third-party — attribution needed** | Yes |

BrentImages is an **ongoing capture series** from Atwood Observatory (Brent's
telescope). Additional nights are expected; the `Img_YYYYMMDD_Atwood/` naming
convention accommodates new captures. Each night yields ~200–280 annotated
frames at 6248×4176 px (native resolution; downsampled for training).

**Validation set:** Derived from `val.json` (SatStreaks + BrentImages splits). DarkMatters holdout images are excluded from all evaluation sets.

### 2.2 Data Provenance Issues (must resolve before publication)

- **DarkMatters:** **Excluded from this project.** Data will not be used in any
  training run, annotation file, or checkpoint. All prior runs that used DarkMatters
  data (Runs 0–2) are archived and their weights must be deleted.

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
| `all_train_nodm.json` **v2 (2026-05-26)** | **8,422** | **8,213** | Tiled BrentImages + Frigate ts=110; use for Run 3 |
| `all_train_nodm.json` v1 (Runs 1–2) | 3,971 | 3,816 | Full-frame BrentImages (streaks ~42px at model input); superseded |

**v2 composition (`all_train_nodm.json`):**

| Component | Tiles | Annotations | `native_tile_size` | Streak at model input |
|-----------|-------|-------------|-------------------|----------------------|
| SatStreaks | 2,488 | 2,488 | full-frame (~4096px) | median ~430px |
| BrentImages Night 1 tiled | 3,110 | 2,978 | 400 px (1:1) | median ~403px |
| BrentImages Night 2 tiled | 1,309 | 1,255 | 400 px (1:1) | median ~406px |
| Frigate tiled | 1,515 | 1,492 | 110 px (3.64×) | median ~55px |
| **Total** | **8,422** | **8,213** | — | min=23 median=369 max=566 |

Generated by:
```bash
python scripts/build_tiled_brentimages_json.py \
    --src data/annotations/brentimages_night1_full.json \
    --out data/annotations/brentimages_night1_tiled_train.json \
    --native-tile-size 400 --overlap 0.5

python scripts/build_tiled_brentimages_json.py \
    --src data/annotations/brentimages_night2_full.json \
    --out data/annotations/brentimages_night2_tiled_train.json \
    --native-tile-size 400 --overlap 0.5

python scripts/build_tiled_frigate_json.py \
    --native-tile-size 110 --overlap 0.5 \
    --out data/annotations/frigate_tiled_train_ts110.json
```

---

## 3. Training Procedure

### 3.1 Training Lineage (Pilot / Informing Runs)

The following runs informed but do **not** constitute the publishable final model:

**⛔ Run 0 — ARCHIVED (May 18, 2026) — trained on DarkMatters data:**
- Config: `models/dino/streak_dinov3_vitb.py`
- Data: `dm_merged_train.json` (SatStreaks + BrentImages N1 + **DarkMatters**)
- Schedule: 4 epochs, MultiStepLR milestones=[3,4], γ=0.1, peak lr=1e-4, 256px
- Hardware: Mac M3 MPS (CPU fallback), ~8h 20m
- Checkpoint: `weights/run_gt_dm_satstreaks_dinov3_vitb/` — **DELETE, do not use**

**⛔ Run 1 — ARCHIVED (May 20–22, 2026) — warm-started from Run 0:**
- Config: `models/dino/streak_dinov3_vitb_longrun.py` (15 epochs, cosine LR, 256px)
- Phase 1A: `all_train_nodm.json` → `weights/run_15ep_nodm/` — **DELETE, do not use**
- Phase 1B: `all_train_withdm.json` → `weights/run_15ep_withdm/` — **DELETE, do not use**
- Warm start: Run 0 checkpoint (DM-contaminated)

**⛔ Run 2 — ARCHIVED (May 22–25, 2026) — warm-started from Run 0:**
- Config: `models/dino/streak_dinov3_vitb_400px.py` (15 epochs, cosine LR, 400px)
- Data: `all_train_nodm.json` (3,971 images, 3,816 annotations)
- Warm start: Run 0 checkpoint (DM-contaminated)
- Checkpoint: `weights/run_best_400px_nodm/` — **DELETE, do not use**
- Historical val results (for reference only):

| Epoch | mAP | mAP@50 |
|-------|-----|--------|
| 5     | 0.316 | 0.390 |
| 10    | 0.408 | 0.463 |
| 15    | 0.423 | 0.468 |

All four weight directories must be deleted from local disk and OneDrive. They must
not be used as warm-start sources for any future training run.

**⏳ Run 3 — Clean cold-start (pending):**
- Config: `models/dino/streak_dinov3_vitb_400px.py`
- Data: `all_train_nodm.json` v2 (8,422 images)
- Warm start: **None** — DINOv3 pretrain backbone + COCO head init only
- Work dir: `weights/run3_cold_nodm/`
- This is the first valid model in the project's history.

> **⚠️ DM contamination note:** Every checkpoint in Runs 0–2 has been exposed to
> DarkMatters data. Run 0 was trained on DM; Runs 1 and 2 warm-started from Run 0.
> The "nodm" label means DM was excluded from the fine-tuning data, but the
> detection head weights were *initialised* from a model that saw DM. This is
> sufficient for production use but **disqualifies these checkpoints as the paper
> model** without a clearly disclosed methods caveat. Run 3 (below) is the clean
> cold-start replacement.

---

**Run 3 — Cold-start DM-free paper model (planned):**
- **Decision (2026-05-26):** Start from scratch — no warm start from any DM-exposed
  checkpoint. This is **Option A** from the §4 Methodology checklist and closes
  Open Question 5.
- Config: `models/dino/streak_dinov3_vitb_400px.py` (same as Run 2)
- Data: `all_train_nodm.json` ± tiled BrentImages and Frigate crops (pending
  attribution resolution — see §4 Data checklist and `docs/adaptive_tiling_plan.md`)
- Warm start: **None** (`load_from = None`) — detection head initialised from scratch
- Hardware: Cloud GPU recommended (RTX 4090); Run 2 took 72h on Mac M3 at 400px
- Expected convergence: ~4–6h to reach Run 0 baseline quality, then continues
  improving through 15 epochs as the head learns without any DM prior
- Checkpoint destination: `weights/run3_cold_nodm/` (to be updated on completion)

> **Why cold-start matters for the paper:** A reviewer could reasonably object that
> the "no-DM" model still has DM-derived weights as its starting point. A cold-start
> eliminates that objection entirely. The mAP cost (if any) from removing the warm
> start is expected to be small — Run 1 showed DM contributes only ~0.001 mAP@50
> in the fine-tuning phase.

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

- [x] DarkMatters data excluded from this project — decision final, no consent obtained
- [ ] Frigate attribution confirmed; owner has granted permission for publication
- [ ] SatStreaks citation identified
- [ ] All training data uploaded to HuggingFace Datasets (or Zenodo) with a DOI
- [ ] Annotation JSONs updated to use hosted paths; DOI recorded in this document
- [ ] Validation set is 100% local (no external volume dependency)

### Model

- [ ] DINOv3 weight SHA-256 recorded here: `[TODO]`
- [ ] DINOv3 paper/model-card citation confirmed

### Reproducibility

- [ ] Random seeds fixed in config:
  ```python
  randomness = dict(seed=42, deterministic=True)
  ```
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
    the final dataset. No prior exposure to DM data in any checkpoint. This is Run 3.
    See §3.1 for rationale and plan.
  - ~~Option B: Accept the Run 0 warm start~~ — rejected; DM contamination in all
    existing checkpoints makes Option B require a methods footnote that weakens the
    no-DM claim.
- [ ] Training resolution confirmed (256px or 400px — 400px substantially slower
      on Mac MPS; consider whether the compute cost is justified vs mAP gain from Run 1)
- [ ] Val set composition confirmed and frozen; report which images are in it

### Paper Run Config Parameters (informed by Run 1 and Run 2)

```python
# Decisions made after Run 1 A/B + Run 2 results + Run 3 decision (2026-05-26):
_img_scale     = (400, 400)                        # 400px: +0.066 mAP@50 vs 256px
TRAIN_ANN_FILE = 'annotations/all_train_nodm.json' # no-DM: avoids consent issue, ~identical perf
max_epochs     = 15                                # cosine schedule converges well by ep15
randomness     = dict(seed=42, deterministic=True) # [TODO: add to final config]
load_from      = None   # ✅ DECIDED: cold start (Option A) — no DM-exposed checkpoint
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

1. ~~**DM consent**~~ — ✅ Resolved (2026-05-26): DarkMatters data excluded from this project entirely.
2. **Frigate attribution** — blocks use of Frigate tiles in the published training set
3. **DINOv3 citation** — need the exact paper reference (not DINOv2)
4. **Data hosting** — HuggingFace Datasets vs Zenodo; need DOI before paper submission
5. ~~**Warm-start strategy**~~ — ✅ Resolved (2026-05-26): **Option A — cold start.**
   All existing checkpoints (Runs 0–2) are contaminated by DM warm-start data; Run 3
   will cold-start the detection head from scratch on `all_train_nodm.json`. See §3.1.
6. **400px vs 256px** — ✅ Resolved: 400px yields mAP@50=0.468 vs 0.402 at 256px (+0.066).
   The gain justifies the compute cost; paper run will use 400px. For a Mac M3 (~72h),
   cloud GPU is strongly recommended for the final paper run.
7. **BrentImages Night 1 FITS local storage** — Night 1 FITS files are only on external
   drive (`/Volumes/External/TrainingData/raw/BrentImages/Img_20260412_Atwood/`). They
   should be copied to local or cloud storage so training runs don't depend on the drive.
8. ~~**Adaptive tiling for training**~~ — ✅ Resolved (2026-05-26). `all_train_nodm.json`
   v2 includes BrentImages N1+N2 tiled at `native_tile_size=400` and Frigate tiled at
   `native_tile_size=110`. See §2.4 for full breakdown.
