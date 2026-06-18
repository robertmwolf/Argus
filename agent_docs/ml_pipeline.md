# ML Pipeline

## Current Production Model

**`vits_v9_asl_cldice`** — DINOv3 ViT-S/16 backbone with ASL + clDice loss.

- Weights: `weights/vits_v9_asl_cldice/best.pt`
- Inference: `inference/vits_window_v9_detector.py`
- API detector ID: `vits_heatmap_v9`
- Env namespace: `VITS_V9_*` (see `.env`)
- Eval: 0.979 recall / 0.918 precision on `val_balanced_v1.json` (t=0.70, pf=0.85)

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
- Heatmap threshold: 0.70
- Peak floor: 0.85
- Do **not** enable Radon refinement — T2 raw OBB geometry is more accurate

---

## Training a New Model

```bash
# 1. Build dataset
python scripts/build_atwood_window_dataset.py \
  --version <N> --source data/annotations/all_train_run17_merged.json \
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
  --annotations /Volumes/External/TrainingData/annotations/val_balanced_v1.json \
  --output results/<tag>/pf85/geometry_eval.json
```

Feature caches go to `/Volumes/External/argus_caches/<tag>/` — delete after training.
Balcache (for eval) also goes there, deleted after eval. Never use `~/` for caches.

---

## Evaluation

Canonical eval: `val_balanced_v1.json`, threshold=0.70, peak_floor=0.85.

```bash
python scripts/compare_geometry_evals.py        # plain table
python scripts/compare_geometry_evals.py --md   # markdown table
```

Record only `geometry_eval.json` in git. Raw `predictions_t*.json` and
`metrics_t*.json` files are regenerable and gitignored.

---

## Detector Lineage

| Model | Status | Notes |
|---|---|---|
| `vits_v9_asl_cldice` | **Production** | ASL+clDice loss, best precision |
| `vits_window_v4` | Retired (kept in API for comparison) | focal+Dice baseline |
| `vitb_window_v4` | Retired (kept in API for comparison) | ViT-B, no precision gain |
| `vitb_v10_asl_cldice` | Retired | ViT-B+clDice; worse than ViT-S on all bands |
| `vits_window_v3` and earlier | Archived | Superseded; detectors in `inference/archive/` |
| Run 1–20 (DINO box / ConvNeXt) | Archived | Pre-window era; results in `results/archive/` |
