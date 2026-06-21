# ML Pipeline

## Current Production Model

**`vits_v9_asl_cldice`** — DINOv3 ViT-S/16 backbone with ASL + clDice loss.

- Weights: `weights/vits_v9_asl_cldice/best.pt`
- Inference: `inference/vits_window_v9_detector.py`
- API detector ID: `vits_heatmap_v9`
- Env namespace: `VITS_V9_*` (see `.env`)
- Eval against `val_balanced_v1_no_sattrains.json` (241 annotations, t=0.85, pf=0.85, ppf=0.85, perp_tol=20px):
  - **0.988 recall / 0.988 precision**, 8.9 px endpoint error (mean), 6.8 px (median), 0.50° angle error
  - Band breakdown (geometry_metrics thresholds — short < 400px, medium 400–1000px, long ≥ 1000px): short 1.000, medium 0.990, long 0.974
  - FN=3, FP=3

## Weight setup

Download the published Hugging Face bundle after creating the Python environment:

```bash
python scripts/sync_hf.py --download --weights-only --weights-dir weights
```

The public `lonewolfman22/argus-weights` repository currently provides the
DINOv3 backbone files and the `run15_vits` and `run17_vitb` heads. It does not
provide `vits_v9_asl_cldice/best.pt`; production-v9 inference additionally
requires a separately supplied checkpoint configured with
`VITS_V9_HEATMAP_CHECKPOINT`. Use `hf auth login` or `HF_TOKEN` only when Hub
authentication is required.

See `docs/loss_ablation_v9_v10_postmortem.md` for full methodology and conclusions.

---

## Architecture

Frozen DINOv3 ViT-S/16 backbone → small conv head → per-patch heatmap → tiled
stitch → OBB detections.

The backbone is always frozen; only the conv head is trained. Features are cached to
disk before training (`scripts/cache_dinov3_heatmap_features.py`) so the backbone
forward pass runs once rather than every epoch.

**Loss: ASL + clDice.** Asymmetric Loss zeroes out easy-negative gradient; clDice
rewards thin, connected, linear predictions via soft morphological skeleton. Together
they suppress false positives without hurting recall. See postmortem for details.

**Inference parameters:**
- Native tile size: 400px (must match training)
- Image normalization: zscore
- Heatmap threshold: 0.85 (better recall+precision than 0.70 on current annotation set)
- Peak floor: 0.85
- Profile peak fraction: **0.85** (`--profile-peak-fraction 0.85`) — heatmap-profile endpoint refinement; reduces endpoint error from 21 px → 6.6 px
- Do **not** enable Radon refinement — T2 raw OBB geometry is more accurate

---

## Dataset Curation Notes

For data layout, integration workflow, and the satellite-train exclusion policy,
see [`docs/data_strategy.md`](../docs/data_strategy.md) and
[`agent_docs/datasets.md`](datasets.md).

**Canonical annotation files (June 2026):**
- Training source: `annotations/all_train_run17_merged_no_sattrains.json` (6052 images)
- Eval set: `annotations/val_balanced_v1_no_sattrains.json` (241 images, 247 annotations)
- Tile dataset: `train_atwood_synth_window_v11/` + `val_atwood_window_v11/` (v11 = coordinate-validated rebuild from same source; v10 had OOB annotation contamination)
- Exclusion manifest: `annotations/sat_train_excluded.json` (53 satellite-train frames)

---

## Training a New Model

```bash
# 1. Build dataset (run the sat-train exclusion check first — see docs/data_strategy.md)
python scripts/build_atwood_window_dataset.py \
  --version <N> \
  --source /Volumes/External/TrainingData/annotations/all_train_run<M>_merged_no_sattrains.json \
  --eval-frames-json /Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json \
  --val-frac 0.08 --neg-frac 0.42 --bg-per-frame 3 --seed 42

# 2. Cache ViT-S features (once)
python scripts/cache_dinov3_heatmap_features.py \
  --annotations <train_ann> --output-dir <cache>/train \
  --backbone vit --model-size small --weights weights/dinov3_vits16_lvd1689m.pth \
  --image-size 518 --native-tile-size 400 --tile-overlap 0.0 --norm-mode none

# 3. Train
python training/train_dinov3_heatmap_cached.py \
  --train-cache <cache>/train --val-cache <cache>/val \
  --work-dir weights/<tag> \
  --epochs 40 --lr 1e-3 --batch-size 32 --hidden-channels 256 \
  --lr-scheduler cosine --early-stopping-patience 10 \
  --loss-mode asl_cldice \
  --asl-gamma-neg 4.0 --asl-gamma-pos 0.0 --asl-margin 0.05 --cldice-iters 3

# 4. Eval
python -m eval.geometry_metrics \
  --predictions results/<tag>/pf85/predictions_t070.json \
  --annotations /Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json \
  --output results/<tag>/pf85/geometry_eval.json
```

Feature caches go to `/Volumes/External/argus_caches/<tag>/` — delete after training.
Balcache (for eval) also goes there, deleted after eval. Never use `~/` for caches.

---

## Evaluation

Canonical eval: `val_balanced_v1_no_sattrains.json` (241 annotations), threshold=0.85, peak_floor=0.85, profile_peak_fraction=0.85, **perp_tol=20px** (`DEFAULT_PERP_THRESHOLD_PX` in `eval/geometry_metrics.py`).

**Band definitions** (as used in `eval/geometry_metrics.py`): short < 400 px, medium 400–1000 px, long ≥ 1000 px. These differ from the annotation-tool / build-script thresholds (50/400); be explicit when discussing band recall.

Standard 3-step eval pipeline:
```bash
# 1. Cache heatmaps
python scripts/cache_heatmap_maps.py \
  --annotations /Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json \
  --checkpoint weights/<tag>/best.pt --output-dir /Volumes/External/argus_caches/<tag>_bal \
  --norm-mode zscore

# 2. Threshold sweep + endpoint refinement
python scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
  --heatmap-cache /Volumes/External/argus_caches/<tag>_bal \
  --annotations /Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json \
  --norm-mode zscore --threshold 0.05 --threshold-sweep 0.70 0.75 0.80 0.85 0.90 \
  --peak-floor 0.85 --profile-peak-fraction 0.85 \
  --output results/<tag>/pf85/metrics.json

# 3. Geometry metrics
python -m eval.geometry_metrics \
  --predictions results/<tag>/pf85/predictions_t085.json \
  --annotations /Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json \
  --output results/<tag>/pf85/geometry_eval.json
```

```bash
python scripts/compare_geometry_evals.py        # plain table
python scripts/compare_geometry_evals.py --md   # markdown table
```

Record only `geometry_eval.json` in git. Raw `predictions_t*.json` and
`metrics_t*.json` files are regenerable and gitignored.

### Endpoint Error Analysis

```bash
python scripts/analyze_endpoint_errors.py \
  --predictions results/<tag>/pf85/predictions_t085.json \
  --annotations /Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json \
  --output results/<tag>/pf85/endpoint_error_analysis.json --top-n 20
```

Baseline v9 (no ppf): mean symmetric endpoint error = 21.2 px, 98% too long.
With ppf=0.85: 6.6 px, 73% too long.

---

## Post-Processing: Heatmap Profile Endpoint Refinement

The v9 model has a systematic "too long" endpoint bias of ~21 px (98% of predictions are too long). This arises because heatmap activation tapers off gradually past the true endpoints rather than dropping sharply, so the connected component bleeds slightly beyond the true endpoint.

**Fix:** `--profile-peak-fraction 0.85` in `evaluate_dinov3_heatmap.py` (or `profile_peak_fraction=0.85` in `_component_to_segment()`). After PCA gives the major axis, the full score_map is projected along that axis within a 1.5-patch corridor. The endpoint is placed where activation drops below 85% of the component peak, rather than at the extreme binary-mask pixel.

| Configuration | Recall | Prec | EndPx (mean) | EndPx (med) |
|---|---|---|---|---|
| Binary mask extremes (no ppf) | 0.988 | 0.988 | ~21 px | — |
| **ppf=0.85** | **0.988** | **0.988** | **8.9 px** | **6.8 px** |

At perp_tol=20px, ppf=0.85 gives ~58% endpoint error reduction with no recall or precision change.

**Endpoint taper (training-time, rejected):** We also tried baking endpoint taper directly into GT heatmap targets (ramp from 1.0 to 0.0 over the last N pixels at each endpoint) with taper sizes 4, 8, and 16 px. All three sizes produced identical recall regression (~0.82–0.84 vs 0.88 baseline) for modest endpoint improvement (~13–14 px). The model over-generalized "suppress near endpoints" and dropped borderline detections. Post-processing refinement is strictly better.

---

## Detector Lineage

| Model | Status | Notes |
|---|---|---|
| `vits_v10_no_sattrains_asl_cldice` | Evaluated, not promoted | 0.802 recall / 0.818 prec — sat-train exclusion slightly hurt (long identical, short/medium each ~1.5 pts lower); revert training exclusion for next run |
| `vits_v9_asl_cldice` | **Production** | ASL+clDice loss, best precision; use ppf=0.85 for endpoint accuracy |
| `vits_window_v4` | Retired (kept in API for comparison) | focal+Dice baseline |
| `vitb_window_v4` | Retired (kept in API for comparison) | ViT-B, no precision gain |
| `vitb_v10_asl_cldice` | Retired | ViT-B+clDice; worse than ViT-S on all bands |
| `vits_taper4/8/16_asl_cldice` | Archived | GT heatmap taper experiments; recall regression, superseded by ppf |
| `vits_window_v3` and earlier | Archived | Superseded; detectors in `inference/archive/` |
| Run 1–20 (DINO box / ConvNeXt) | Archived | Pre-window era; results in `results/archive/` |
