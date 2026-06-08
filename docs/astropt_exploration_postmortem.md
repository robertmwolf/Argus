# AstroPT-89M Backbone Exploration — Post-Mortem

**Runs:** 13, 14, 16  
**Date:** June 2026  
**Outcome:** Abandoned — 0% medium-streak recall across all configurations

---

## Motivation

After 12 runs of hyperparameter tuning, the frozen DINOv3 ViT-S backbone plateaued at
18.18% medium-streak recall (Run 12). The hypothesis was that AstroPT v1.0, a GPT-style
transformer pretrained autoregressively on DESI full-sky optical survey data, would produce
better streak/background feature separability than the ImageNet-pretrained ViT-S.

---

## AstroPT Architecture

| Property | Value |
|---|---|
| Model | `Smith42/astroPT`, path `models/fully_trained/0089M_params` |
| Parameters | ~89M |
| Architecture | GPT-2-style, 12 layers, 12 heads, causal attention |
| Embed dim | 768 |
| Patch size | 16 × 16 px |
| Fixed input size | **256 × 256 px** (positional embeddings `nn.Embedding(1024, 768)`, not interpolatable) |
| Internal norm | Per-patch instance norm: `(patch − μ) / (σ + 1e-8)` on each 768-dim flattened patch |
| Feature output | `(B, 768, 16, 16)` after reshape — 256 cells total |

Causal attention was disabled post-load by filling the triangular bias buffers with 1.0,
converting the model to bidirectional for spatial feature extraction.

Code: `models/plain_dinov3/astro_backbone.py`, `inference/astropt_heatmap_detector.py`

---

## What We Tried

### Run 13 — Full-image downscale (1800 → 256px)

**Cell size:** 113px native/cell  
**Normalization:** autostretch  
**Dataset:** 33/33/33 short/medium/long synthetic-augmented, 9,440 tiles  

| Metric | Value |
|---|---|
| Long recall | 27.8% |
| Medium recall | 0% |
| FPs / image | 9.7 |

**Failure mode:** OBBs 2–6× oversized. Medium streaks (253–696px native) spanned only
2–6 feature cells; the geometry head predicted boxes that were too large → IoU < 0.1 → zero
recall. Long streaks were detectable because they were longer than the oversized geometry error.

---

### Run 14 — Native 256px tiling

**Cell size:** 16px native/cell  
**Normalization:** autostretch  
**Dataset:** Same 33/33/33 annotation JSON, tiled at native 256px  

| Metric | Value |
|---|---|
| Long recall | 1.5% |
| Medium recall | 0% |
| FPs / image | **262** |

**Failure mode:** At native resolution, star PSF halos (5–10px FWHM) span 1–3 adjacent
cells as high-contrast rings that the model read as elongated objects. Autostretch amplified
halo contrast further. The model fired aggressively on background astronomical structure.

---

### Run 16 — 720px native tiles + zscore

**Cell size:** 45px native/cell  
**Normalization:** zscore  
**Dataset:** `all_train_run12_atwood1800_npy.json`, 8,332 × 1800×1800 NPY images,
tiled at 720px with 50% overlap (~44,000 train tiles)  
Training: 40 epochs, batch=32, pos_weight=20, cosine LR, val_dice=0.827

| Metric | Value |
|---|---|
| Long recall | **50.7%** |
| Medium recall | **0%** |
| FPs / image | 49 |

**Result:** Best AstroPT configuration. The 720px cell size pushed PSF halos to ~0.1–0.2
cells (sub-cell), eliminating most of the Run 14 FP flood. Long recall improved significantly
over Run 13. But medium recall remained 0% at every threshold from 0.05 to 0.95 — the
model detects exactly 114/228 GT annotations (all long) and never fires on medium-length
streaks regardless of confidence threshold.

---

## Root Cause: 16×16 Feature Grid

The fundamental limitation is AstroPT's fixed 256px input and 16×16 feature map (256 cells).

| Backbone | Feature grid | Cells/image | Native px/cell | Medium streak cells |
|---|---|---|---|---|
| ViT-S Run 12 | 32 × 32 | 1,024 | 55px | 5–13 |
| AstroPT Run 16 | **16 × 16** | **256** | 45px | 5–15 |

Although the native px/cell is similar (45 vs 55), AstroPT has 4× fewer spatial cells.
The geometry head must predict OBB parameters from a coarser grid, and the training data
(68% long streaks) biases it toward long-streak predictions. Medium streaks get predicted
as long → OBB too large → IoU < 0.1 → not counted.

Raising the OBB IoU threshold from 0.1 might show some medium detections, but the
underlying mismatch between prediction and GT OBB would still limit usable precision.

---

## Comparison Table

| Run | Backbone | px/cell | Long recall | Med recall | FPs/img |
|---|---|---|---|---|---|
| Run 12 | ViT-S frozen | 55 | 57.1% | **18.2%** | 24 |
| Run 13 | AstroPT frozen | 113 | 27.8% | 0% | 10 |
| Run 14 | AstroPT frozen | 16 | 1.5% | 0% | 262 |
| Run 16 | AstroPT frozen | 45 | 50.7% | 0% | 49 |

---

## Conclusion

AstroPT does not improve on ViT-S for streak detection with a frozen backbone and heatmap
head. Three configurations spanning a 7× range of cell sizes all produced 0% medium recall.
The 16×16 feature grid is the hard ceiling: it cannot be worked around by tiling strategy
or normalization choice without changing the model architecture.

**Next step:** Partial ViT-S unfreeze on cloud GPU (last 2–4 transformer blocks).
The ViT-S 32×32 grid already achieves 18% medium recall frozen; unfreezing should allow
the features to specialize on streak detection and break the current ceiling.
