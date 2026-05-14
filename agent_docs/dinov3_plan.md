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

### Stage 1 — backbone frozen (epochs 1–20)
- DINOv3 backbone: `requires_grad=False`, `lr_mult=0.0`
- Adapter + DETR head: full lr
- Rationale: DINOv3 features are high quality; let the head learn first

### Stage 2 — partial backbone unfreeze (epochs 21–50)
- Unfreeze last 4 transformer blocks of ViT-L: `lr_mult=0.01`
- Adapter + head: `lr_mult=1.0`
- Lower backbone LR than Swin to preserve pretrained representations

### Batch / memory (A100 40 GB, ViT-L, 800px)
- `batch_size=1`, gradient accumulation steps=2 (effective batch=2)
- Gradient checkpointing: `True` (ViT-L activations are large)
- Mixed precision: `True` (CUDA only)
- Estimated VRAM: ~34 GB with gradient checkpointing

### Mac dev (ViT-B, 400px)
- `batch_size=1`, `num_workers=0`, `pin_memory=False`
- No AMP (MPS)
- Use `USE_DEV_SUBSET=true` (50-image subset)

## Evaluation Gates

| Gate | Metric | Target |
|------|--------|--------|
| Phase A | Streak patches produce visually distinct PCA heatmap | Subjective pass |
| Phase A | ViT-B runs on Mac MPS at 400px in <30 s/image | Wall time |
| Phase B | MMDet config parses with DINOv3 backbone | `mmdet.utils.check_config` |
| Phase B | `pipeline.py --fast` with ViT-B <60 s on Mac | Wall time |
| Phase C | ViT-L dev subset 50 epochs: mAP@0.5 >0.50 | vs Swin-T 0.657 |
| Phase C | Full dataset cloud: mAP@0.5 ≥0.70 | Phase 8 target ≥94% prec/rec |

## Phase Sequence

| Phase | Task | Where |
|-------|------|-------|
| A | Feasibility probe: feature visualization + memory check | Mac CPU/MPS |
| B | `dinov3_adapter.py` + MMDet configs | Mac |
| C | Update `train_dino.py` + smoke test (1 epoch, dev subset) | Mac MPS |
| D | Cloud training: ViT-L, full dataset, 50 epochs | A100 |
| E | Evaluation vs Co-DINO Swin baseline | Mac MPS |

## Open Questions (resolve during Phase A/B)

1. Does DINOv3 hub loading work offline from local `.pth`? (test in probe)
2. Does `get_intermediate_layers(n=4)` on ViT-B/L give usable multi-scale signal,
   or are all 4 layers too similar (isotropic ViT problem)?
3. Can the PatchToPyramid adapter recover enough spatial resolution for thin
   streaks (16 px wide at 400 px image = ~1 patch wide)?
4. ConvNeXt fallback: does ConvNeXt-L LVD outperform SAT ViT-L in ablation?
