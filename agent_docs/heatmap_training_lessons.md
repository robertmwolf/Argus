# Heatmap Detector — Training & Evaluation Lessons Learned

Accumulated from Runs 5 through 15. Read this before training or evaluating a
new heatmap checkpoint.

---

## 1. Tile size must match exactly between training and inference

The heatmap head learns to activate on streaks at a specific angular resolution
(pixels-per-patch). If the native tile size at inference differs from the one
used when building the training feature cache, detections collapse to near-zero.

**Run 12 failure:** model trained at `native_tile_size=400` (each streak ≈ 8–25
patches), evaluated at `native_tile_size=1800` (same streak ≈ 1–3 patches) →
0% short/medium recall.

**Rule:** Always record `native_tile_size` in the pipeline script and checkpoint
`args`. The default for `evaluate_dinov3_heatmap.py` is controlled by
`VITS_HEATMAP_NATIVE_TILE_SIZE` / `CONVNEXT_HEATMAP_NATIVE_TILE_SIZE`; verify
it matches before running eval.

---

## 2. Normalisation mode must match between training and inference

The backbone's input distribution shifts dramatically between `autostretch`,
`zscore`, and `zscale`. A mismatch causes near-zero or saturated activations.

**Rule:** Set `ARGUS_NORM` (or pass `--norm-mode`) to whatever mode was used
when building the training cache. The mode is stored in `train_cache_metadata`
inside the checkpoint; check it with:

```python
ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
print(ckpt.get("train_cache_metadata", {}))
```

Run 15 uses `zscore` throughout (cache build + training + eval).

---

## 3. Stitch `max_gap_px=400` absorbs short/medium detections into FP chains

`stitch_collinear_fragments` uses union-find with no constraint on how large a
merged detection can become. A 339 px short-streak detection was transitively
chained into a 2866 px false-positive chain (ratio 8.5×), moving the centroid
278 px from ground truth → 0% short recall.

**Fix (applied in Run 15):** `max_growth_ratio=3.0` parameter added to
`stitch_collinear_fragments`. Any merge that would produce a span more than 3×
the longer input fragment is rejected. Defaults:

```python
stitch_collinear_fragments(preds, max_gap_px=400, max_growth_ratio=3.0)
```

Both `evaluate_dinov3_heatmap.py` (`--stitch-max-growth-ratio`) and
`run_posthoc_threshold_analysis.py` (`--stitch-max-growth-ratio`) expose this
as a CLI flag.

---

## 4. Per-band recall is misleading when stitch is enabled

`eval/metrics.py:evaluate()` filters **both** predictions and ground truth by
`streak_length_px` band before computing per-band precision/recall. When stitch
merges a 267 px medium prediction with a nearby long-streak fragment, the merged
detection becomes 768 px (`streak_length_px >= 400`) and disappears from the
`medium` band's prediction pool — the medium GT goes unmatched in the per-band
metric even though the overall recall captures it.

**Rule:** Use **overall recall** as the primary metric when stitch is enabled.
Per-band recall is only reliable in no-stitch mode. The genuine medium-streak
matching rate (any-band bbox IoU ≥ 0.10) improved from 17% to 70% with stitch
at t=0.85 for Run 15.

---

## 5. geometry_weight defaults to 0.25 — geometry head IS trained

`train_dinov3_heatmap_cached.py` has `geometry_weight=0.25` as its default.
Unless you explicitly pass `--geometry-weight 0`, the geometry head (cos2θ,
sin2θ, length, width) is trained. The inference code gates geometry use on this
weight:

```python
use_geometry = float(ckpt.get("args", {}).get("geometry_weight", 0.0)) > 0.0
```

Disabling geometry for a checkpoint trained with it (or vice versa) degrades
angle estimates. Always verify before overriding.

---

## 6. Post-hoc threshold sweep — always generate predictions at t=0.05 without stitch

The canonical eval pattern is:

```bash
# Step 1: generate raw predictions (no stitch, low threshold)
python scripts/evaluate_dinov3_heatmap.py \
    --annotations data/annotations/val_run12_1800_npy.json \
    --checkpoint  weights/run15_vits/best.pt \
    --output      results/run_N/t0.05_nostitch/metrics.json \
    --tiled --threshold 0.05

# Step 2: sweep thresholds + stitch combinations on saved predictions
python scripts/run_posthoc_threshold_analysis.py \
    --predictions results/run_N/t0.05_nostitch/predictions.json \
    --annotations data/annotations/val_run12_1800_npy.json \
    --output-dir  results/run_N/threshold_sweep \
    --thresholds 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.95 \
    --iou-threshold 0.10 --stitch
```

If you generate predictions *with* `--stitch`, the stitch is baked in and the
post-hoc sweep re-stitches already-stitched detections — this is a no-op and
will not reflect any subsequent changes to the stitch parameters.

---

## 7. val_run12_1800_npy.json band distribution

For reference, with `evaluate()` defaults (`short < 150 px`, `long >= 400 px`):

| Band   | Count | Range (px) |
|--------|------:|-----------|
| short  |     6 | 70 – 125  |
| medium |   205 | 156 – 398 |
| long   |   949 | 401 – 1800|

There are only 6 true "short" GT annotations — short recall is statistically
noisy. Medium annotations (154 in the 150–200 px range) are the meaningful
short-streak proxy for this dataset.

---

## 8. Run 15 best operating point

Checkpoint: `weights/run15_vits/best.pt`  
Tile size: 400 px native, 50% overlap  
Normalisation: zscore  
Threshold: **0.85** (best F1)  

| Metric        | Value  |
|---------------|--------|
| Precision     | 77.3%  |
| Recall        | 89.2%  |
| F1            | 82.8%  |
| Long recall   | 89.1%  |
| OBB baseline F1 | 23.7% |

Stitch parameters: `max_gap_px=400`, `max_growth_ratio=3.0`.
