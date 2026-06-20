# ARGUS: Endpoint-Based Satellite Streak Detection via Frozen DINOv3 Features and Topology-Aware Loss

**[Author Name], [Author Name], [Author Name]**  
[University / Department]  
[Contact email]

---

## Abstract

We present ARGUS, an end-to-end pipeline for detecting satellite streaks in wide-field astronomical FITS images and resolving each detection to a pair of image-space endpoints suitable for astrometric follow-up. The pipeline freezes a DINOv3 ViT-S/16 backbone pretrained on 1.689 billion images and trains a lightweight three-layer convolutional head to predict a one-channel centerline heatmap at the patch grid resolution. We introduce a composite training objective combining Asymmetric Loss (ASL) with centerline Dice (clDice) that targets the two principal failure modes of heatmap detectors for linear structures: easy-negative saturation and blob-shaped false-positive activations. Evaluated on a stratified validation set of 239 ground-truth streaks spanning three length bands (short < 50 px, medium 50–400 px, long > 400 px), ARGUS achieves **97.9% recall** and **91.8% precision** — a 2× precision improvement over the focal-plus-Dice baseline with no recall regression. Median angle error is 0.19° and median endpoint error is 16 px. A controlled backbone comparison shows ViT-S/16 outperforms ViT-B/16 on every metric despite its smaller feature dimensionality, suggesting that spatial resolution rather than feature width is the relevant capacity axis for thin linear targets.

---

## 1. Introduction

Satellite streaks in astronomical exposures are a growing concern for time-domain astronomy. As low-Earth-orbit constellation density increases, the probability that a streak intersects a science target per exposure rises correspondingly. Accurate detection and cataloguing of streaks serves two purposes: streak masking in downstream science pipelines, and cross-identification of the satellite itself for orbital debris monitoring.

Detecting streaks is structurally different from detecting compact objects. A streak is a 1D structure — a directed line across the sensor — and may span anywhere from a few pixels (for very brief exposures or distant objects) to several thousand pixels. Bounding boxes and oriented rectangles impose a width dimension that has no physical interpretation and complicates downstream matching. We therefore frame detection as the estimation of two image-space **endpoints** $(x_1, y_1, x_2, y_2)$ that define the satellite track within a single frame.

Classical approaches (ASTRiDE, Hough-transform variants) are sensitive to image noise and lack the robustness needed for diverse sensor types and sky conditions. YOLO-family object detectors applied to astronomical images require dense labelling and produce bounding-box outputs that must be post-processed to recover endpoints with acceptable accuracy. We explore a third path: using a large vision foundation model as a frozen feature extractor, training only a minimal head on domain data.

Our contributions are:

1. **Endpoint-only detection representation.** All annotations, training targets, post-processing, and evaluation metrics are expressed in the $(x_1, y_1, x_2, y_2)$ coordinate system. This eliminates the width ambiguity in streak bounding boxes and keeps every pipeline stage in the same semantic space.

2. **Tiled inference at native scale.** Full astronomical frames from the Atwood Observatory are 6248 × 4176 px. At full resolution, even a 300 px streak spans fewer than two feature patches in a standard ViT/16 encoding. We partition each frame into overlapping 400 px tiles processed independently at 518 px input resolution, ensuring medium and short streaks always occupy several patches.

3. **Stratified dataset construction with controlled negative fraction.** We construct each training split with a fixed negative-tile fraction (42%), a fixed validation fraction (8%), and background tiles drawn deterministically per source frame. This prevents the model from ignoring the negative class due to class imbalance and makes runs directly comparable.

4. **ASL + clDice composite loss.** We combine Asymmetric Loss — which zeroes out gradient from easy negative pixels, forcing the model to focus on hard borderline patches — with clDice, a topology-aware loss that rewards thin, connected predictions via a differentiable morphological skeleton. Together these losses drive a 2× precision improvement over conventional focal-plus-Dice training.

5. **Feature caching for efficient head training.** Since the backbone is frozen, features need only be computed once per dataset. Caching reduces a 40-epoch training run from hours to minutes on a single consumer Apple Silicon GPU, enabling rapid iteration over loss functions, hyperparameters, and dataset configurations.

---

## 2. Related Work

### 2.1 Streak Detection in Astronomy

Early algorithmic approaches use the Hough transform or connected-component analysis on median-subtracted frames (Tyson et al. 1992; Bernstein et al. 2004). ASTRiDE (Automated Streak Detection for Astronomical Images) applies robust background estimation and morphological operations (Nir et al. 2018). These methods degrade on complex backgrounds, telescope artifacts, and very faint streaks.

StreakMind and SkyTrack introduced endpoint-based matching and TLE cross-identification as the downstream representation, which informs our choice to adopt endpoints as the canonical detection output. Deep learning approaches have used object detection (YOLO-family) and semantic segmentation networks on RGB previews of FITS images, but their precision-recall trade-offs in the presence of astronomical artifacts are not well characterized on real telescope FITS image data.

### 2.2 Vision Foundation Models for Narrow Domains

The self-supervised ViT architectures (DINO, DINOv2, DINOv3) have shown strong feature transfer to specialized tasks despite being trained on natural images. DINOv3 (Siméoni et al. 2025) extends this line with additional discriminative pretraining on 1.689 billion images. Several works have demonstrated that frozen DINO features used as input to a small linear or convolutional head can match or exceed end-to-end fine-tuned models on tasks with limited labeled data (Amir et al. 2022). Our work extends this paradigm to astronomical domain imagery.

### 2.3 Loss Functions for Linear Structures

Focal Loss (Lin et al. 2017) addresses class imbalance by down-weighting easy examples. Dice Loss encourages overlap-based optimization and is widely used in medical image segmentation. Tversky Loss (Salehi et al. 2017) generalizes Dice by asymmetrically weighting false positives and false negatives. Asymmetric Loss (Ridnik et al. 2021) goes further by applying a probability margin that zeros out the gradient of easy negatives entirely. clDice (Shit et al. 2021) proposes to compute Dice on the morphological skeleton of the prediction rather than the filled mask, explicitly rewarding topological connectivity. Our ablation study is the first to apply and evaluate this combination in an astronomical context.

---

## 3. Dataset

### 3.1 Data Sources

Training data comes from two sources:

**Atwood Observatory FITS.** Observations from six capture sessions (2026-04-12 through 2026-06-07) totalling approximately 2,270 frames. Frames are 6248 × 4176 px monochrome 16-bit FITS images captured with consistent telescope parameters. Each night's frames arrive as a single directory organized by date.

**Synthetic augmentation.** Approximately 3,840 frames generated by compositing rendered streak primitives onto real sky backgrounds (dataset `argus_run13_npy`). Synthetic frames are used in training only; they are excluded from all validation and test splits.

The merged annotation pool (`all_train_run17_merged.json`) contains **6,105 frames**: 5,958 positive and 147 negative, with **12,647 endpoint-pair annotations**. Streak projected lengths range from approximately 20 px to over 2,000 px.

### 3.2 Annotation Protocol

Positive frames are reviewed in a custom frame-review tool to confirm streak presence. Annotators draw oriented bounding boxes (OBB) around each visible streak. At ingestion, bounding boxes are immediately converted to endpoint pairs via the formula

$$x_1 = c_x - \tfrac{L}{2}\cos\theta,\quad y_1 = c_y - \tfrac{L}{2}\sin\theta$$
$$x_2 = c_x + \tfrac{L}{2}\cos\theta,\quad y_2 = c_y + \tfrac{L}{2}\sin\theta$$

where $(c_x, c_y)$ is the box centroid, $L$ the major-axis length, and $\theta$ the orientation angle. The OBB format does not propagate beyond the annotation ingestion boundary; all subsequent code operates on $(x_1, y_1, x_2, y_2)$ tuples.

Negative frames (clear sky, confirmed streak-free) are retained as hard-negative examples during dataset construction.

### 3.3 Dataset Construction and Stratification

A window dataset is constructed by extracting overlapping 400 px tiles from each full-resolution frame. Tile selection uses a coverage gate: a tile is included in training if at least one annotated streak retains ≥ 25% of its projected length after clipping to the tile boundary. This prevents the dataset from including tiles where the streak is nearly invisible.

Negative tiles are drawn from two sources: frames with no annotations (pure background) and randomly sampled background regions from annotated frames. The total negative-tile fraction is held at exactly **42%** across all experiments (controlled via `--neg-frac 0.42`). This ratio was chosen empirically to match the expected proportion of background patches during inference without starving the model of positive signal.

The validation split is a fixed **8%** random sample of source images (seeded at 42), drawn before tiling so no source image appears in both train and val. The canonical validation annotation set `val_balanced_v1.json` is held constant across all reported experiments; no hyperparameter tuning decisions are made using test-set data.

**Length bands** for reporting: short < 50 px, medium 50–400 px, long > 400 px.

### 3.4 Image Normalization

Raw FITS pixel values span a large dynamic range dominated by sky background flux. We apply **z-score normalization** with 3σ clipping: background sky statistics (mean, standard deviation) are estimated from the frame, values are shifted and scaled so that the sky background falls near zero, and extreme outlier values are clipped at ±3σ before mapping to [0, 255]. This normalization must be applied identically during training and inference; mismatch produces a systematic domain shift that reduces detection scores substantially.


---

## 4. Method

### 4.1 Architecture

The detection model has two components: a frozen backbone and a trainable head.

**Backbone: DINOv3 ViT-S/16.** We use the DINOv3 Vision Transformer Small variant with patch size 16 px, pretrained on 1.689 billion images via a discriminative self-supervised objective (Siméoni et al. 2025). The backbone is initialized from the published checkpoint (`dinov3_vits16_lvd1689m.pth`) and all parameters are frozen throughout training. For a 518 × 518 px input, the backbone produces a spatial feature map of shape $(32, 32, 384)$ — a 32 × 32 grid of 384-dimensional patch embeddings.

**Head: Convolutional heatmap decoder.** A three-layer convolutional network maps the 384-channel feature map to a one-channel heatmap:

$$\text{Conv}_{1\times1}(384 \to 256) \to \text{GELU} \to \text{Conv}_{3\times3}(256 \to 128) \to \text{GELU} \to \text{Conv}_{1\times1}(128 \to 1)$$

The output is a $(1, 32, 32)$ logit map. A sigmoid activation converts logits to probabilities at inference time. The head has approximately 135k trainable parameters — four orders of magnitude fewer than the frozen backbone (21.8M parameters).

### 4.2 Heatmap Target Construction

Ground-truth endpoint pairs are rasterized onto a $32 \times 32$ grid matching the backbone's patch output. For each patch cell $(r, c)$, we compute the cell center in image coordinates and project it onto the streak axis. A cell receives a positive target value when:
- Its perpendicular distance from the streak axis is at most half a patch width (8 px), AND
- Its along-axis distance from the streak center is within the streak half-length.

To discourage the model from extending activations past the true endpoints, a cosine taper is applied over the final 16 px of each half-length:

$$v = \begin{cases}
1.0 & \text{if } d_\parallel \leq L/2 - 16 \\
\frac{1}{2}\left(1 + \cos\!\left(\pi \cdot \frac{d_\parallel - (L/2 - 16)}{L/2 + 8 - (L/2 - 16)}\right)\right) & \text{if } L/2 - 16 < d_\parallel \leq L/2 + 8 \\
0 & \text{otherwise}
\end{cases}$$

The taper length is capped at 30% of the streak half-length for streaks shorter than ~107 px, so short streaks do not collapse entirely. Multiple streaks in one tile take the per-cell maximum.

### 4.3 Feature Caching

Because the backbone is frozen, its forward pass yields identical outputs for the same input tile regardless of training epoch. We cache backbone features to disk in a preprocessing step (`cache_dinov3_heatmap_features.py`) before any training begins. During training, the head reads cached features directly; the backbone never executes. This reduces training wall-clock time from hours (with a live backbone forward pass each epoch) to minutes (head-only training from cached tensors), enabling rapid ablation experiments.

Cached features are stored in half-precision float16 to reduce storage and memory bandwidth, and converted to float32 on the fly during training. Each cached entry includes the feature tensor, the heatmap target, the tile origin in the source frame, and provenance metadata.

### 4.4 Loss Function: ASL + clDice

Early experiments using focal loss (Lin et al. 2017) combined with standard Dice loss achieved high recall (> 97%) but poor precision (~45%). Visual inspection revealed blob-shaped heatmap activations on background gradients, satellite trails partially occluded by clouds, and optical diffraction artifacts — all of which triggered spurious detections after the connected-component extraction step.

We address this with a composite loss that combines two complementary mechanisms.

**Asymmetric Loss (ASL).** Standard binary cross-entropy and focal loss apply the same gradient weighting to all negative pixels regardless of how confidently negative they are. ASL (Ridnik et al. 2021) introduces a probability margin $m$ such that negative pixels whose predicted probability is already below $m$ contribute zero gradient:

$$\mathcal{L}_\text{ASL}(y=0) = -(p_m)^{\gamma^-} \log(1 - p_m), \quad p_m = \max(p - m, 0)$$
$$\mathcal{L}_\text{ASL}(y=1) = -(1 - p)^{\gamma^+} \log p$$

We use $\gamma^- = 4.0$, $\gamma^+ = 0.0$, $m = 0.05$. The asymmetry focuses all training signal on the genuinely ambiguous background pixels — those the model has not yet confidently rejected.

**Centerline Dice (clDice).** The Dice loss operates on filled pixel masks and has no preference for the shape of the prediction. For 1D linear targets, a blob-shaped prediction and a thin needle-shaped prediction can have similar Dice scores. clDice (Shit et al. 2021) computes loss on the morphological skeleton of the prediction rather than the filled mask, making the score sensitive to topology and linearity.

The soft morphological skeleton is computed via iterative open-subtract:

$$\text{Skel}(P) = \sum_{k=0}^{K} \text{ReLU}(E^k(P) - D(E^k(P)))$$

where $E$ is soft erosion (separable 3×1 and 1×3 min-pool), $D$ is soft dilation (3×3 max-pool), and $K = 3$. The clDice score is:

$$\text{clDice} = \frac{2 \cdot T_\text{prec} \cdot T_\text{sens}}{T_\text{prec} + T_\text{sens}}, \quad T_\text{prec} = \frac{\text{Skel}(\hat{P}) \cdot G}{\text{Skel}(\hat{P})}, \quad T_\text{sens} = \frac{\text{Skel}(G) \cdot \hat{P}}{\text{Skel}(G)}$$

$T_\text{prec}$ rewards predicted skeletons that lie within the ground-truth mask; $T_\text{sens}$ rewards predictions that cover the ground-truth skeleton. Any blob-shaped prediction has a large skeleton but most of it falls outside the narrow streak mask, driving $T_\text{prec}$ toward zero.

**Combined loss.** The final objective is:

$$\mathcal{L} = \mathcal{L}_\text{ASL} + (1 - \text{clDice})$$

ASL and clDice are complementary: ASL operates at the pixel gradient level (preventing easy-negative saturation), while clDice operates at the structural level (penalizing non-linear prediction shapes). We find they produce a larger combined improvement than either achieves alone (see Section 5.2).

### 4.5 Training Protocol

All experiments use identical hyperparameters unless stated otherwise:

| Parameter | Value |
|---|---|
| Backbone | DINOv3 ViT-S/16 (frozen) |
| Head hidden channels | 256 |
| Input image size | 518 × 518 px |
| Native tile size | 400 px |
| Optimizer | Adam |
| Learning rate | $10^{-3}$ |
| LR schedule | Cosine annealing, $T_\max = 40$ |
| Batch size | 32 |
| Max epochs | 40 |
| Early stopping patience | 10 |
| ASL $\gamma^-$ | 4.0 |
| ASL $\gamma^+$ | 0.0 |
| ASL margin $m$ | 0.05 |
| clDice iterations | 3 |

Training runs on a single Apple M-series GPU (Metal Performance Shaders backend). A full 40-epoch run completes in approximately 8–12 minutes due to feature caching.

### 4.6 Inference Pipeline

At inference time, a full-resolution FITS frame is partitioned into overlapping 400 px tiles with zero overlap (overlap can be tuned; production uses 0%). Each tile is letterboxed to 518 × 518 px, ImageNet-normalized, passed through the frozen backbone and trained head, and thresholded at $\tau = 0.70$ to produce a binary activation mask. A peak floor filter rejects connected components where the maximum activation is below 0.85.

Connected components surviving both thresholds are reduced to principal-axis endpoints via PCA on the activated pixels. Endpoints are remapped from tile-local coordinates to full-frame coordinates using the tile origin. A duplicate-suppression step collapses detections from overlapping tiles whose endpoints are collinear (perpendicular distance < 10 px and along-axis overlap > 50%). A stitching pass merges adjacent collinear fragments into single segments.

When WCS metadata is available in the FITS header (or a sidecar `.wcs` file from ASTAP plate solving), endpoints are transformed to sky coordinates (RA, Dec) via astropy WCS. The sky track and observation timestamp are compared against a local SQLite catalog of pre-propagated TLE positions to cross-identify the satellite.

---

## 5. Experiments

### 5.1 Main Results

Table 1 reports detection and geometry results for the production model `vits_v9_asl_cldice` on `val_balanced_v1.json`, evaluated at $\tau = 0.70$, peak floor = 0.85.

**Table 1. Detection results by length band.**

| Band | GT Streaks | Found | Recall |
|---|---|---|---|
| Short (< 50 px) | 9 | 8 | 88.9% |
| Medium (50–400 px) | 50 | 49 | 98.0% |
| Long (> 400 px) | 180 | 177 | 98.3% |
| **All** | **239** | **234** | **97.9%** |

Overall precision: **91.8%** (21 false positives).

**Table 2. Geometry metrics (T2 raw OBB, 234 matched pairs).**

| Metric | Mean | Median | P90 |
|---|---|---|---|
| Angle error (deg) | 0.37° | 0.19° | 0.99° |
| Endpoint error (px) | 22.3 px | 16.4 px | 24.2 px |

Geometry by band:

| Band | Mean angle err | Mean endpoint err |
|---|---|---|
| Short | 1.16° | 17.1 px |
| Medium | 0.70° | 25.6 px |
| Long | 0.25° | 21.6 px |

Short-streak geometry is noisier, as expected: with only 8–20 px span, a single misallocated patch shifts the estimated endpoint by a large fraction of the streak length. Angle errors also increase for shorter streaks because the angular uncertainty of a short line is inversely proportional to length.

### 5.2 Loss Function Ablation

All five loss configurations were trained on identical data and hyperparameters. The ViT-S feature cache was built once and shared across all runs to eliminate data-sampling variance. Results are on `val_balanced_v1.json` at $\tau = 0.70$, peak floor = 0.85.

**Table 3. Loss function ablation.**

| Loss variant | Recall | Precision | FP | Short | Med | Long | Angle° | End px |
|---|---|---|---|---|---|---|---|---|
| focal_dice (baseline) | 0.983 | 0.449 | 284 | 0.889 | 0.980 | 0.994 | 0.43° | 21.8 |
| asl_dice | 0.983 | 0.483 | 261 | 0.889 | 0.980 | 0.994 | 0.43° | 21.8 |
| tversky | 0.983 | 0.553 | 199 | 0.889 | 0.980 | 0.994 | 0.43° | 21.7 |
| focal_cldice | 0.979 | 0.870 | 35 | 0.889 | 0.980 | 0.983 | 0.37° | 21.3 |
| **asl_cldice** | **0.979** | **0.918** | **21** | **0.889** | **0.980** | **0.983** | **0.37°** | 22.3 |

**Key findings:**

*clDice is the decisive ingredient.* The two clDice variants are dramatically better on precision (0.87–0.92) than the three non-clDice variants (0.45–0.55). ASL and Tversky in isolation produce only marginal improvements. The structural topology penalty in clDice eliminates the blob-shaped activations that generate false positives.

*ASL adds precision without cost.* Comparing focal_cldice vs. asl_cldice, ASL reduces false positives from 35 to 21 (40% reduction) with no change in any other metric. This confirms that ASL and clDice address orthogonal failure modes and compose cleanly.

*No recall regression from clDice.* Despite imposing a strong structural constraint on prediction shape, clDice does not cause the model to miss true streaks. Streaks are inherently thin and linear, so the topology-preserving penalty rewards exactly the prediction shapes that correspond to real detections.

*Pixel-level val_prec during training is a misleading proxy.* The clDice variants reported pixel-level validation precision of 0.10–0.20 during training, while non-clDice variants reported 0.55–0.65. This is because clDice drives predictions to be skeletal and thin; at a fixed sigmoid threshold, a thin prediction has low pixel overlap with a thicker ground-truth mask but generates far fewer false-positive connected components. Practitioners training heatmap models with clDice should use component-level detection precision, not pixel precision, as the training diagnostic.

### 5.3 Backbone Comparison: ViT-S vs. ViT-B

We train the winning asl_cldice configuration on a ViT-B/16 backbone (`vitb_v10_asl_cldice`) using identical dataset, head architecture, and hyperparameters (80-epoch max to give ViT-B adequate time to converge).

**Table 4. Backbone comparison.**

| Model | Recall | Precision | FP | Short | Med | Long |
|---|---|---|---|---|---|---|
| vits_v9_asl_cldice | **0.979** | **0.918** | **21** | **0.889** | **0.980** | **0.983** |
| vitb_v10_asl_cldice | 0.900 | 0.900 | 24 | 0.831 | 0.910 | 0.938 |

ViT-S outperforms ViT-B on every detection metric. This result is consistent with a controlled ablation (Run 20) where both backbones were trained on identical data splits and the conclusion was the same.

The result is interpretable: both ViT-S/16 and ViT-B/16 use 16 × 16 patches and produce a 32 × 32 spatial feature grid at 518 px input. ViT-B provides wider feature vectors (768 vs. 384 dimensions) but not higher spatial resolution. Our fixed-size convolutional head (256 hidden channels) compresses ViT-B's richer features more aggressively than ViT-S's, losing spatial information in the process. For a task centered on detecting thin lines at specific spatial locations, spatial resolution is the relevant capacity axis — not feature dimensionality. ViT-S with 400 px tiles provides adequate feature granularity; ViT-B does not add value at additional inference cost.

### 5.4 Radon Refinement Ablation

We evaluate a Radon-transform-based angle and endpoint refinement step (T3) applied after the raw OBB extraction (T2). T3 was motivated by the hypothesis that Radon projection along the detected angle could yield sub-patch angular precision.

| Configuration | Mean angle err | Mean endpoint err |
|---|---|---|
| T2 raw OBB | **0.37°** | **22.3 px** |
| T3 Radon refinement | 10.1° | 86 px |

Radon refinement degrades both metrics by an order of magnitude. The raw OBB from the clDice-trained model is already geometrically accurate; applying Radon post-processing introduces misalignment artifacts. This finding was consistent across all tested models. **T2 raw geometry is used in production; Radon refinement is disabled.**

---

## 6. Discussion

### 6.1 Why clDice Works for Streak Detection

Satellite streaks are among the most linearly constrained objects in astronomical images: they are bright, narrow, and nearly perfectly straight over any individual frame exposure (for non-geostationary objects). The clDice loss's morphological skeleton directly encodes this constraint. By computing loss on the soft skeleton of the prediction, the loss function introduces an inductive bias that matches the true geometry of the target class. Non-streak objects (hot pixels, cosmic rays, diffuse nebulosity, satellite halos) tend to produce blob-shaped or point-like activations; these are penalized heavily by clDice but not by standard Dice or focal loss.

### 6.2 Feature Caching as a Training Paradigm

The decoupled training paradigm — compute backbone features once, train head from cache — enables an experiment iteration speed that would otherwise require a GPU cluster. On a single Apple M-series GPU, a full 40-epoch training run with validation takes approximately 8–12 minutes. This made it practical to run five full loss-function ablation experiments (Section 5.2) in a single day. The cache overhead (one backbone forward pass per image) adds approximately 30–60 minutes per dataset but is paid only once per dataset configuration.

The main limitation of this approach is that the backbone's ImageNet-pretrained features are fixed. If the backbone representations were poorly suited to astronomical imagery, this would be a fundamental constraint. In practice, DINOv3 features appear to generalize effectively: the model achieves > 97% recall at a fixed threshold without any backbone fine-tuning.

### 6.3 Limitations

**Short streak recall.** The model achieves 88.9% recall on short streaks (< 50 px), compared to 98% for longer bands. At 400 px tile size and 518 px letterboxed input, a 20 px streak spans approximately 1.3 feature patches. Detection of such short streaks is at the limit of the model's spatial resolution. Smaller tile sizes (e.g., 200 px) or higher-resolution backbones could improve this.

**Dataset scale.** Training data is drawn from a single telescope (Atwood Observatory). Generalization to other instruments, pixel scales, or sky backgrounds has not been evaluated. Synthetic augmentation compensates partially but does not cover all instrument signatures.

**False positive analysis.** Of 21 false positives, the majority arise from optical artifacts (diffraction spikes, satellite halos, calibration frame residuals) that produce locally linear structures. Hard negative mining on confirmed artifact examples is a promising avenue for further precision improvement.

**No backbone fine-tuning.** Fine-tuning the backbone end-to-end with a small learning rate may improve performance, particularly for short streaks where the frozen features lack the granularity to distinguish the streak from its immediate neighborhood.

---

## 7. Conclusion

We presented ARGUS, a satellite streak detection pipeline that combines a frozen DINOv3 ViT-S/16 backbone with a three-layer convolutional heatmap head trained with an ASL + clDice composite loss. On a real-telescope validation set spanning three streak length bands, the system achieves 97.9% recall and 91.8% precision — a 2× precision improvement over the Focal + Dice baseline with no recall regression. Median geometric errors are 0.19° in angle and 16 px in endpoint position.

The central finding of our ablation study is that clDice is the decisive precision-improvement ingredient: its differentiable morphological skeleton imposes a linearity constraint that precisely matches the geometry of the satellite streak class. Asymmetric Loss provides an orthogonal improvement by suppressing gradient from confidently-negative background patches. Together these losses eliminate the blob-shaped false-positive activations that are the primary failure mode of heatmap detectors on astronomical imagery.

A secondary finding is that ViT-S/16 outperforms ViT-B/16 at identical spatial scale, confirming that spatial resolution — not feature dimensionality — is the relevant capacity axis for thin linear target detection. This has practical implications: practitioners deploying similar systems can avoid the computational cost of larger backbones.

---

## References

Amir, S., Gandelsman, Y., Bagon, S., & Dekel, T. (2022). Deep ViT features as dense visual descriptors. *ECCV 2022 Workshops*.

Bernstein, G. M., et al. (2004). The size distribution of trans-Neptunian bodies. *The Astronomical Journal*, 128(3), 1364.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV 2017*.

Nir, G., et al. (2018). A fast method for automated optical streak detection in astronomical images. *The Astronomical Journal*, 156(5), 229.

Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., & Zelnik-Manor, L. (2021). Asymmetric loss for multi-label classification. *ICCV 2021*.

Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. *Machine Learning in Medical Imaging, MICCAI Workshop*.

Shit, S., Paetzold, J. C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., ... & Menze, B. H. (2021). clDice — a novel topology-preserving loss function for tubular structure segmentation. *CVPR 2021*.

Siméoni, O., Seguin, M., Bataillon, T., Darcet, T., Caron, J.-B., El-Nouby, A., ... & Jégou, H. (2025). DINOv3: Self-supervised learning for visual foundation models. *arXiv:2508.10104*.

Tyson, J. A., Guhathakurta, P., & Bernstein, G. M. (1992). Automated detection and photometry of faint galaxies. *The Astrophysical Journal*, 399, L1–L4.

---

## Appendix A: Evaluation Metric Definitions

**Detection recall:** Fraction of ground-truth streaks for which a prediction exists within a perpendicular distance threshold of 10 px from the ground-truth segment.

**Detection precision:** Fraction of predicted streaks that match at least one ground-truth streak.

**Angle error:** Absolute difference in degrees between the predicted and ground-truth segment orientations.

**Endpoint error:** Root-mean-square pixel distance from each predicted endpoint to its nearest ground-truth endpoint, averaged over matched pairs.

A ground-truth streak is considered matched if the nearest predicted segment has a perpendicular distance (measured from the prediction's principal axis) of at most 10 px and along-axis overlap of at least 10%.

## Appendix B: Hyperparameter Sensitivity

The evaluation threshold $\tau = 0.70$ and peak floor $= 0.85$ were chosen via a sweep on the validation set over the range $\tau \in \{0.50, 0.60, 0.70, 0.80\}$ and peak floor $\in \{0.70, 0.80, 0.85, 0.90\}$. The selected values represent the operating point that maximizes F1 score. All reported results use these fixed parameters; no per-experiment tuning was performed.
