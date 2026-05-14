# DINOv3 Backbone Integration Plan

Branch: `feature/dinov3-backbone`

## Objective

Replace the Swin-L backbone in the existing Co-DINO / DINO-DETR streak detector
with a DINOv3 ViT backbone. DINOv3 is Meta's self-supervised ViT foundation model
(6× larger training set than DINOv2, Gram anchoring for dense feature quality).

## Checkpoint Selection

### Primary training backbone — ViT-L/16 LVD-1689M
```
dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```
- 300 M parameters — fits a single A100 40 GB with gradient checkpointing
- LVD-1689M (1.7 B diverse web images) — domain-neutral, no harmful terrestrial
  bias unlike SAT-493M (Earth observation, wrong domain for night-sky images)
- Embed dim 1024, patch size 16

### Dev / Mac backbone — ViT-B/16 LVD-1689M
```
dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```
- 86 M parameters — fits 16 GB unified memory at 400 px
- Same ViT architecture as ViT-L, shallower depth
- Used for Phase A probe and Mac-side integration testing

### Reference — ViT-7B COCO DETR head (inspect only, do not train with)
```
dinov3_vit7b16_coco_detr_head-b0235ff7.pth
```
- Pre-trained DETR detection head (COCO2017) paired with ViT-7B backbone
- Download, inspect `state_dict` keys, and use as architecture blueprint for the
  DETR detection head attached to ViT-L at ARGUS scale
- ViT-7B itself (6.7 B params) exceeds single-A100 budget; not for training

### Fallback backbone — ConvNeXt-Large LVD-1689M
```
dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth
```
- Hierarchical `[C, H/4, H/8, H/16, H/32]` feature maps slot directly into the
  existing Co-DINO MMDetection neck without a patch-to-pyramid adapter
- Use if ViT patch-to-pyramid integration proves unstable in MMDetection

### Excluded
- SAT-493M variants: wrong domain (Earth observation ≠ night sky telescope imagery)
- ViT-7B variants: exceed single-A100 memory budget for training

## Why LVD over SAT for astronomy

Night-sky FITS images are: nearly black background, PSF-shaped point sources
(stars), thin high-aspect-ratio bright lines (streaks). SAT-493M models use
different pixel normalization (mean=(0.430,0.411,0.296)) tuned for colorful
terrestrial imagery and will produce a systematic domain mismatch on dark-field
astronomical data. LVD-1689M is domain-neutral and uses standard ImageNet
normalization (mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)).

## Architecture Plan

### Feature extraction
DINOv3 ViT is a flat (isotropic) transformer — it does not produce a native
feature pyramid. To feed the existing Co-DINO / Deformable DETR neck we need a
patch-to-pyramid adapter.

```
DINOv3 ViT-L/16
  ↓  get_intermediate_layers(x, n=4, reshape=True)
  →  4 × (B, 1024, H/16, W/16) patch feature maps at different depths
  ↓  PatchToPyramid adapter (models/dino/dinov3_adapter.py)
     - 1×1 conv projections to [256, 256, 256, 256] channels
     - Bilinear upsample to produce [H/8, H/16, H/32, H/64] strides
  →  FPN-compatible feature pyramid
  ↓  Existing Co-DINO neck + DINO-DETR head (unchanged)
```

ConvNeXt-L produces `[C, H/4, H/8, H/16, H/32]` directly → skip adapter entirely.

### New files
```
models/dino/dinov3_adapter.py         — PatchToPyramid, DINOv3Backbone (MMDet registry)
models/dino/streak_dinov3_vitb.py     — MMDet config: ViT-B dev (Mac, 400px)
models/dino/streak_dinov3_vitl.py     — MMDet config: ViT-L production (A100, 800px)
scripts/download_dinov3_weights.py    — Download ViT-B + ViT-L + COCO DETR head
scripts/probe_dinov3.py               — Phase A feasibility probe
```

### Modified files
```
training/train_dino.py     — add --backbone {swin,dinov3_vitb,dinov3_vitl,convnext_l}
inference/pipeline.py      — backbone-agnostic (already is, if adapter is correct)
agent_docs/assistant_guide.md  — update weights paths
```

### Normalization
ViT LVD models expect ImageNet normalization applied to uint8→float tensors.
The FITS loader already produces uint8 arrays; the MMDetection pipeline handles
tensor normalization via the `Normalize` transform in the data pipeline.
Update `mean` and `std` in the MMDet config to:
```python
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],   # ImageNet (0–255 scale)
    std=[58.395, 57.120, 57.375],
    to_rgb=True,
)
```
This replaces the current z-score-derived values in the Swin configs.

## Training Strategy

The DINOv3 backbone requires zero training — the downloaded weights are the
finished product of Meta's pretraining run. Only the adapter and detection head
need to learn from streak data.

### What actually trains

| Component | Params | Notes |
|-----------|--------|-------|
| PatchToPyramid adapter | ~4 M | 1×1 convs only |
| DETR detection head | ~40 M | queries, cross-attn, cls+bbox |
| DINOv3 backbone | 0 (frozen) | `requires_grad=False` throughout |

### Primary training — backbone fully frozen
- DINOv3 backbone: `requires_grad=False`, `lr_mult=0.0` for all epochs
- Adapter + DETR head: full lr from epoch 1
- Evaluate fully before considering any backbone fine-tuning

Memory with frozen backbone (no backbone gradients or optimizer states):

| Config | Memory | Runs on |
|--------|--------|---------|
| ViT-B/16, 400px, batch=1 | ~6 GB | Mac MPS (16 GB) |
| ViT-L/16, 400px, batch=1 | ~9 GB | Mac MPS (16 GB) |
| ViT-L/16, 800px, batch=1 | ~12 GB | RTX 5070 Ti (16 GB VRAM) |

### Optional Stage 2 — partial backbone unfreeze (deferred)
Only attempt if fully-frozen evaluation on the full dataset falls short of Phase 8
targets (≥94% precision, ≥97% recall). This is the only step that requires
cloud-scale GPU:
- Unfreeze last 4 ViT-L transformer blocks: `lr_mult=0.01`
- Estimated VRAM with unfrozen blocks at 800px: ~18–22 GB → A100 40 GB required
- Do not start Stage 2 without frozen-backbone eval results in hand

### Mac dev (ViT-B, 400px)
- `batch_size=1`, `num_workers=0`, `pin_memory=False`
- No AMP (MPS)
- Use `USE_DEV_SUBSET=true` (50-image subset)
- Run full 50-epoch frozen training here as dev validation

### Workstation (RTX 5070 Ti, ViT-L, 800px)
- Full SatStreaks dataset, frozen backbone, 50 epochs
- `batch_size=1`, mixed precision, gradient checkpointing on head
- Follow `agent_docs/Training_Handoff.md` for handoff procedure

## Evaluation Gates

| Gate | Metric | Target |
|------|--------|--------|
| Phase A ✅ | Cosine dissimilarity streak vs background | 0.0951 — PASS |
| Phase B | MMDet config parses with DINOv3 backbone | `mmdet.utils.check_config` |
| Phase B | `pipeline.py --fast` with frozen ViT-B <60 s on Mac | Wall time |
| Phase C | Frozen ViT-B, dev subset, 50 epochs: mAP@0.5 >0.50 | Mac MPS validation |
| Phase D | Frozen ViT-L, full dataset, 50 epochs | Workstation RTX 5070 Ti |
| Phase D | mAP@0.5 ≥0.70, precision ≥94%, recall ≥97% | Phase 8 targets |
| Phase E (optional) | Partial ViT-L unfreeze if Phase D falls short | A100 only if needed |

## Phase Sequence

| Phase | Task | Where | Status |
|-------|------|--------|--------|
| A | Feasibility probe: cosine dissimilarity + PCA heatmaps | Mac CPU | ✅ DONE (cosine dissim=0.095 > 0.05 gate) |
| B | `dinov3_adapter.py` + MMDet configs + pipeline smoke test | Mac | ✅ DONE (smoke test loss 37→30) |
| C | Full 50-epoch frozen ViT-B training, dev subset | Mac MPS | ✅ DONE (best mAP@0.5=0.274 on dev_subset, 0.002 on test — expected from 50-image train) |
| D | Full 50-epoch frozen ViT-L training, full dataset | RTX 5070 Ti | ⏳ PENDING — handoff in Training_Handoff.md |
| E | Evaluation vs Co-DINO Swin-T/L baseline on test.json | Mac | ✅ PARTIAL — Swin-T baseline 0.190; ViT-L column pending Phase D |
| F (optional) | Partial ViT-L unfreeze if Phase D targets not met | A100 | deferred — evaluate after Phase D |

### Phase E current results (test.json)

| Model | mAP | mAP@0.5 | mAP@0.75 | Training data |
|-------|-----|---------|---------|---------------|
| Co-DINO Swin-T | 0.149 | **0.190** | 0.167 | full merged (SatStreaks + GTImages) |
| DINOv3 ViT-B (Phase C) | 0.001 | 0.002 | 0.000 | 50-image dev_subset only — not a fair comparison |
| DINOv3 ViT-L (Phase D) | TBD | TBD | TBD | full merged — apples-to-apples vs Swin-T |

**Phase E gate**: ViT-L mAP@0.5 ≥ 0.190 (match Swin-T) → Phase D succeeded.
Within ±5 pp = acceptable. > 5 pp below → consider Phase F.

## Open Questions (resolve during Phase A/B)

1. Does DINOv3 hub loading work offline from local `.pth`? (test in probe)
2. Does `get_intermediate_layers(n=4)` on ViT-B/L give usable multi-scale signal,
   or are all 4 layers too similar (isotropic ViT problem)?
3. Can the PatchToPyramid adapter recover enough spatial resolution for thin
   streaks (16 px wide at 400 px image = ~1 patch wide)?
4. ConvNeXt fallback: does ConvNeXt-L LVD outperform SAT ViT-L in ablation?
