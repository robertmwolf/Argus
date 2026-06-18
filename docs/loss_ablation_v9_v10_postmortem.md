# Loss Ablation Study: Window v9 / v10 Postmortem

**Date:** June 2026
**Outcome:** `vits_v9_asl_cldice` is the production model. ViT-B (v10) does not improve over ViT-S.

---

## Problem Statement

Through window_v8, detection recall was strong (>97%) but **precision was poor** — the
best ViT-S result had detection precision of 0.449 (focal+Dice baseline). False positives
were the primary failure mode: the model predicted blob-shaped heatmap activations that
triggered detections on noise, background gradients, and diffuse artifacts.

The hypothesis was that **the loss function, not the dataset or architecture, was the
bottleneck.** The training data and annotation quality had improved through v5–v8; the
model had enough signal to find streaks but no penalty specifically discouraging
non-linear, blob-shaped predictions.

---

## v9 Design: Loss Ablation

Five loss modes were trained on identical hyperparameters and the v9 dataset (same recipe
as v8: calibration-corrected frames, val_frac=0.08, neg_frac=0.42, seed=42). The ViT-S
feature cache was built once and shared across all five runs to eliminate any
data-sampling variance.

| Variant | Loss | Purpose |
|---|---|---|
| `focal_dice` | Focal(γ=2, α=0.85) + Dice | v8 ViT-S baseline |
| `asl_dice` | ASL(γ_neg=4, m=0.05) + Dice | Hard-negative focus; zeros out easy negatives |
| `focal_cldice` | Focal + clDice(iters=3) | Topology-aware; rewards linear connectivity |
| `tversky` | Focal + Tversky(α_fp=0.6) | Penalises FPs 1.5× more than FNs in Dice |
| `asl_cldice` | ASL + clDice | Combined precision-targeting + topology |

Fixed hyperparameters across all variants: backbone=ViT-S/16 (frozen), head
hidden_channels=256, lr=1e-3, batch=32, cosine LR (T_max=40), early_stop patience=10,
tile_size=400, image_size=518.

---

## Results

All five variants evaluated on `val_balanced_v1.json` at threshold=0.70, peak_floor=0.85.

| Variant | Recall | Precision | FP | Short | Med | Long | Angle° | EndPx |
|---|---|---|---|---|---|---|---|---|
| focal_dice (baseline) | 0.983 | 0.449 | 284 | 0.889 | 0.980 | 0.994 | 0.43 | 21.8 |
| asl_dice | 0.983 | 0.483 | 261 | 0.889 | 0.980 | 0.994 | 0.43 | 21.8 |
| tversky | 0.983 | 0.553 | 199 | 0.889 | 0.980 | 0.994 | 0.43 | 21.7 |
| focal_cldice | 0.979 | 0.870 | 35 | 0.889 | 0.980 | 0.983 | 0.37 | 21.3 |
| **asl_cldice** | **0.979** | **0.918** | **21** | **0.889** | **0.980** | **0.983** | **0.37** | **22.3** |

---

## Key Findings

### clDice is the decisive ingredient

The two clDice variants (focal_cldice, asl_cldice) are dramatically better on precision
than the three non-clDice variants — 0.87–0.92 vs 0.45–0.55. ASL and Tversky produce
only marginal improvements over the baseline.

**Why clDice works for streak detection:** clDice (Centerline Dice) computes loss via the
soft morphological skeleton of the prediction. It rewards thin, connected, linear
predictions and penalises blob-shaped activations. For satellite streak detection this is
exactly the right inductive bias — a streak is a 1D structure; any prediction that
spreads laterally or forms a disconnected blob is penalised. The non-clDice losses all
treat the heatmap as an independent-pixel binary classification problem and have no
structural penalty for blob predictions.

### val_prec during training was misleading

The clDice variants showed very low pixel-level `val_prec` during training (0.10–0.20)
while the non-clDice variants showed much higher values (0.55–0.65). This is because
clDice pushes predictions to be narrow and skeletal — at a fixed t=0.5 binarisation
threshold, a thin prediction scores low pixel-precision but generates far fewer
connected components that survive the per-component filtering step. The training-time
`val_prec` metric is **not a reliable proxy** for detection precision; use geometry_eval
on the balanced val set.

### No recall penalty

All five variants maintained 0.979–0.983 recall with no short/medium/long band
regression. The clDice penalty for non-linear predictions does not cause the model to
miss true streaks — streaks are inherently thin and linear, so the loss rewards exactly
the prediction shapes that correspond to real detections.

### Stitching is a no-op for clDice models

Pre- and post-stitch evaluation of `vits_v9_asl_cldice` produced identical results
(recall, precision, FP count, geometry all unchanged). The clDice loss rewards connected
linear predictions, so the model produces complete streak heatmaps that do not fragment
at tile boundaries. The stitcher can be left in place as a safety net with zero cost.

### Radon refinement degrades geometry

T3 (Radon angle + endpoint tracing) applied to `vits_v9_asl_cldice` worsened angle
error from 0.37° to 10.1° and endpoint error from 22px to 86px. The raw OBB output
(T2) from the model is already geometrically accurate; Radon is solving a problem
that no longer exists. **Use T2 raw geometry in production; do not enable Radon.**
This finding is consistent across all prior models (see memory: radon_degrades_geometry).

---

## v10: ViT-B Does Not Improve Over ViT-S

`vitb_v10_asl_cldice` trained the winning asl_cldice loss on a ViT-B/16 backbone
(same dataset, same head, same hyperparameters, 80-epoch max).

| Model | Recall | Precision | FP | Short | Med | Long |
|---|---|---|---|---|---|---|
| vits_v9_asl_cldice | **0.979** | **0.918** | **21** | **0.889** | **0.980** | **0.983** |
| vitb_v10_asl_cldice | 0.900 | 0.900 | 24 | 0.831 | 0.910 | 0.938 |

ViT-B is worse across every detection metric. This matches the Run 20 controlled
backbone experiment, which reached the same conclusion.

**Why ViT-B does not help:** Both ViT-S/16 and ViT-B/16 use 16×16 patches and produce
the same 32×32 spatial feature grid at 518px input. ViT-B has wider feature vectors
(768-dim vs 384-dim) but not higher spatial resolution. The same fixed-size conv head
(hidden_channels=256) is used for both, meaning ViT-B's richer features are compressed
more aggressively. For this task — detecting thin bright lines on dark backgrounds —
ViT-S's features are sufficient and the additional capacity does not translate to better
detection. To give ViT-B a fair comparison a wider head (hidden_channels=512+) would
be needed, but given the task simplicity this is unlikely to change the outcome.

---

## Production Conclusion

**`weights/vits_v9_asl_cldice/best.pt`** is the production model.

- Inference module: `inference/vits_window_v9_detector.py`
- API detector ID: `vits_heatmap_v9`
- Env namespace: `VITS_V9_*`
- Default threshold: 0.70, peak_floor: 0.85
- Do not enable Radon refinement (T2 raw geometry is better)
- Stitching can remain enabled (no-op, harmless)

If further improvement is needed, the most promising levers are:
1. **Hard negative mining**: add mined hard negatives to the training set (not yet tried
   with clDice loss)
2. **Wider head**: test hidden_channels=512 with ViT-B if capacity is the bottleneck
3. **More data**: additional BrentImages capture sessions
