# Training Methods

## Active approach

ARGUS trains a one-channel centerline heatmap using a frozen DINOv3 ViT backbone
and a small convolutional head. Endpoint annotations are rasterized into
centerline heatmap targets. The head is trained on pre-cached ViT features so the
backbone forward pass runs once per dataset, not once per epoch.

**Loss function: ASL + clDice** — Asymmetric Loss zeros out easy-negative
gradient and focuses training on hard negatives; clDice rewards thin, connected,
linear predictions via soft morphological skeleton. Together they suppress false
positives without hurting recall. See
[`docs/loss_ablation_v9_v10_postmortem.md`](loss_ablation_v9_v10_postmortem.md)
for the ablation study that identified this combination.

## Training pipeline

See [`agent_docs/ml_pipeline.md`](../agent_docs/ml_pipeline.md) for the
canonical step-by-step training and evaluation recipe.

## Recording runs

For every run, record:
- Dataset version and seed
- Backbone (ViT-S/16 or ViT-B/16) and frozen/unfrozen status
- Native tile size and overlap
- Normalisation mode (`none` for caching, `zscore` for eval)
- Loss mode and hyperparameters
- Evaluation threshold and peak floor
- Path to `geometry_eval.json`

Evaluate with `eval.geometry_metrics`. Commit only `geometry_eval.json` — raw
`predictions_t*.json` and `metrics_t*.json` files are regenerable and gitignored.

## Retired approaches

Old experiment-specific paths (ConvNeXt heatmap, DINO box head, orientation-binned
centerline, classical ASTRiDE) are archived under `training/archive/`,
`scripts/archive/`, and `inference/archive/`. New work should extend the
`train_dinov3_heatmap_cached.py` path.
