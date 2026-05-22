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
| SatStreaks | 3,023 (train) | ~3,100 streaks | `data/satstreaks/` local | Public dataset `[TODO: cite]` | Yes |
| BrentImages Night 1 (GTImages) | included above via `dm_merged_train.json` | — | `data/BrentImages/Img_20260412_Atwood/` | Proprietary, Atwood Observatory captures | Yes |
| BrentImages Night 2 | 204 annotated + 27 negatives | 204 streaks | `/Volumes/External/…/Img_20260515_Atwood/` | Proprietary, Atwood Observatory captures | Yes |
| DarkMatters (DM) | 149 training | ~150 streaks | `/Volumes/External/DarkMatters/` | **Third-party — consent unresolved** | No |
| Frigate (tiled) | 558 positive tiles + 159 neg | 655 streaks | `/Volumes/External/frigate/` | **Third-party — attribution needed** | Yes |

**Validation set (`dm_merged_val.json`):** 411 images (345 SatStreaks/BrentImages N1,
66 DarkMatters holdout — DM holdout images now mirrored locally to
`data/darkmatters/val_previews/`).

### 2.2 Data Provenance Issues (must resolve before publication)

- **DarkMatters:** Provided by a third party. Written consent for use in a published
  dataset and model has not been confirmed. **If consent cannot be obtained, the
  paper model must be the no-DM variant.** Results from the with-DM model may still
  be reported as an ablation if the data source is disclosed.

- **Frigate:** Source and ownership `[TODO]`. Written permission to include this data
  in a published training set must be obtained. Attribution required in the paper.

- **SatStreaks:** `[TODO]` Identify the canonical citation and license.

- **BrentImages (Night 1 & 2):** First-party captures from Atwood Observatory.
  Suitable for publication. Should be hosted with a DOI (HuggingFace Datasets or
  Zenodo recommended).

### 2.3 Annotation Format

All training splits are COCO-format JSON with a single category:
`{"id": 1, "name": "streak", "supercategory": "satellite"}`.

Tiled Frigate crops use a virtual path encoding:
`<original_stem>__tx<x0>_ty<y0>_ts<tile_size><ext>`
which is resolved to the real crop at load time by `training/transforms.py:LoadFITSFromFile`.
Tile parameters: 400×400 px, 25% overlap.

### 2.4 Training Split Sizes (current pilot runs)

| Split | Images | Annotations |
|-------|--------|-------------|
| `all_train_nodm.json` (Phase 1A) | 3,971 | 3,816 |
| `all_train_withdm.json` (Phase 1B) | 4,120 | 3,991 |

---

## 3. Training Procedure

### 3.1 Training Lineage (Pilot / Informing Runs)

The following runs informed but do **not** constitute the publishable final model:

**Run 0 — Cold start (May 18, 2026):**
- Config: `models/dino/streak_dinov3_vitb.py`
- Data: `dm_merged_train.json` (SatStreaks + GTImages/BrentImages N1 + DarkMatters)
- Schedule: 4 epochs, MultiStepLR milestones=[3,4], γ=0.1, peak lr=1e-4, 256px
- Hardware: Mac M3 MPS (CPU fallback), ~8h 20m
- Result: mAP@50 on dm_merged_val: 0.257 → 0.341 → 0.392 → **0.436** (best epoch 4)
- Checkpoint: `weights/run_gt_dm_satstreaks_dinov3_vitb/best_coco_bbox_mAP_epoch_4.pth`

**Run 1 — A/B warm-start (in progress, May 20–21, 2026):**
- Config: `models/dino/streak_dinov3_vitb_longrun.py` (15 epochs, cosine LR, 256px)
- Phase 1A: `all_train_nodm.json` → `weights/run_15ep_nodm/`
- Phase 1B: `all_train_withdm.json` → `weights/run_15ep_withdm/`
- Warm start: Run 0 checkpoint (detection head only, backbone still frozen)
- Purpose: Measure DM contribution; inform dataset choice for paper run
- **Note:** This is not a clean DM ablation — the warm-start checkpoint was itself
  trained on DM data. Both A/B branches begin from a model that has seen DarkMatters.

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

- [ ] DarkMatters consent confirmed in writing, OR decision made to exclude DM
      from the paper model
- [ ] Frigate attribution confirmed; owner has granted permission for publication
- [ ] SatStreaks citation identified
- [ ] All training data uploaded to HuggingFace Datasets (or Zenodo) with a DOI
- [ ] Annotation JSONs updated to use hosted paths; DOI recorded in this document
- [ ] Validation set is 100% local (no external volume dependency) ✅ done for
      DM holdout images

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

- [ ] Decide on warm-start strategy:
  - **Option A (recommended for clean ablation):** Cold-start the detection head
    directly on the final dataset. No prior exposure to DM data in any checkpoint.
    Longer to converge (~4–6h to reach Run 0 baseline, then continues improving).
  - **Option B:** Accept the Run 0 warm start and disclose in the paper that the
    initial 4-epoch checkpoint included DarkMatters data, then fine-tuned without it.
    Simpler but requires a clear methods footnote.
- [ ] Training resolution confirmed (256px or 400px — 400px substantially slower
      on Mac MPS; consider whether the compute cost is justified vs mAP gain from Run 1)
- [ ] Val set composition confirmed and frozen; report which images are in it

### Paper Run Config Parameters (to be filled after Run 1 results)

```python
# Fill these in after reviewing Run 1 A/B results:
_img_scale    = (???, ???)   # 256 or 400
TRAIN_ANN_FILE = '???'       # nodm or withdm (pending consent)
max_epochs    = ???           # informed by Run 1 convergence curve
randomness    = dict(seed=42, deterministic=True)
load_from     = None          # or Run 0 checkpoint if Option B above
```

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
| Standard (SatStreaks test split) | ~600 | Primary benchmark |
| BrentImages Night 2 (zero-shot) | 231 | Out-of-distribution generalisation |
| Frigate (zero-shot, tiled inference) | 350 | Short-streak regime |
| DarkMatters holdout (zero-shot) | ~83 | Only if DM consent resolved |

Zero-shot sets are never seen during training; they measure generalisation.

### 5.3 Inference Variants

- **Standard:** Full-image resize to training resolution; use for SatStreaks and DM
- **Tiled:** `inference/tiled_pipeline.py`, tile_size=400, overlap=0.5, cross-tile NMS
  at IoU=0.4; use for Frigate and any high-resolution frames

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
- 400px: not yet measured (Run 1 Phase 2 pending)

---

## 7. Open Questions

1. **DM consent** — blocks the with-DM paper model and clean A/B interpretation
2. **Frigate attribution** — blocks use of Frigate tiles in the published training set
3. **DINOv3 citation** — need the exact paper reference (not DINOv2)
4. **Data hosting** — HuggingFace Datasets vs Zenodo; need DOI before paper submission
5. **Warm-start strategy** — Option A vs B (see §4 Methodology above)
6. **400px compute** — is the mAP gain (unknown until Run 1 Phase 2) worth the
   training time on Mac MPS? If significant, may justify cloud GPU for paper run.
