# Run 20 — Controlled backbone experiment (ViT-S vs ViT-B on identical data)

**Status:** running (launched 2026-06-13). Fills the gap left by Runs 17–19.

## Why this exists

The ViT-B heatmap detector failed across Runs 17/18/19 (val_dice ~0.12, eval
recall 0.000 on `val_balanced_v1`). ViT-S (Run 15) succeeded (val_dice 0.77,
recall 0.93). **But the two were never trained on the same data**, so backbone
and data were confounded:

| Run | Backbone | Training data | val_dice | Notes |
|-----|----------|---------------|---------:|-------|
| 15  | ViT-S/16 | run10 NPY (pre-tiled, different lineage) | 0.77 | production detector |
| 17  | ViT-B/16 | mixed FITS+PNG+synthetic NPY | 0.105 | format-mixed |
| 18  | ViT-B/16 | train_run18.json (real-FITS tiled) | 0.122 | consistent split |
| 19  | ViT-B/16 | train_run18.json | 0.12 | hot LR, diverged |

Run 20 holds **everything constant except the backbone** to isolate the cause.

### Retracted intermediate conclusions (2026-06-13)

A series of fast linear-probe experiments on cached features suggested in turn:
"capacity wall", "frozen features don't generalize", and "mislabeled training
data". **All three are retracted.** The probe's held-out dice metric read
~0.10 *even on `val_balanced_v1`*, which the real detection pipeline scores at
0.93 recall — i.e. the probe metric cannot distinguish good data from bad and
proved nothing. Only the real training pipeline (`train_dinov3_heatmap_cached.py`)
and the real eval (`evaluate_dinov3_heatmap.py`) are trusted here. The
brightness-overlap "misalignment" check is likewise unreliable (too insensitive
to thin faint streaks; gave the same null on known-good data).

What *is* reliably known: ViT-B genuinely fails (real eval recall 0.000), and it
*can* memorize 24 train tiles (overfit dice 0.97), so features+targets are not
garbage — the failure is in fitting/generalizing the full set.

## Design (single variable = backbone)

| Held constant | Value |
|---------------|-------|
| Training data | `data/annotations/train_run18.json` (+ `val_run18.json`) |
| Eval data | `data/annotations/val_balanced_v1.json` |
| Cache geometry | image_size 518, native_tile 400, overlap 0.0, zscore, neg/img 4 |
| Recipe R | epochs 40, lr 1e-3, batch 32, pos_weight 20, geom_weight 0.25, hidden 256, cosine, no warmup |

| Variable | ViT-S arm | ViT-B arm |
|----------|-----------|-----------|
| Backbone | `dinov3_vits16_lvd1689m.pth` (384-d) | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (768-d) |
| Feature cache | `~/argus_run20_vits_cache` (new) | `~/argus_run19_cache` (reused from Run 19) |
| Head weights | `weights/run20_vits_control` | `weights/run20_vitb_control` |

The ViT-B arm retrains from Run 19's existing cache under Recipe R (no re-cache),
so both arms are byte-for-byte comparable on data and recipe.

## Readout

- **ViT-S also lands ~0.12** → the **data** (`train_run18`) is the limiting
  factor; ViT-B is exonerated. Next step: investigate what makes `train_run18`
  harder than Run 15's run10-NPY data (real-FITS faintness, tiling, target
  rendering) — and likely fix the data rather than the model.
- **ViT-S reaches ~0.7** → the failure is **ViT-B-specific** on this data.
  Next step: ViT-B-specific remediation (or just ship ViT-S, which has no
  demonstrated disadvantage).

## Reproduce

```bash
bash scripts/run20_vits_control_pipeline.sh 2>&1 | tee /tmp/run20_$(date +%Y%m%d_%H%M%S).log
```

Outputs: `weights/run20_{vits,vitb}_control/history.json`,
`results/run20_control/run20_{vits,vitb}_control/pf85/`, and a machine-readable
`results/run20_control/provenance.json` mapping each arm to its data, backbone,
recipe, and result paths.

## Results — CONCLUSION: it's the DATA (2026-06-13)

Both backbones fail **identically** on `train_run18` under Recipe R:

| Arm | final train_dice | best val_dice | vs same backbone on good data |
|-----|-----------------:|--------------:|-------------------------------|
| ViT-S on train_run18 | 0.130 | 0.122 | ViT-S on run10-NPY (Run 15) = 0.77 |
| ViT-B on train_run18 | 0.144 | 0.127 | — |

Same recipe, same data, swap the backbone → no change. **Backbone exonerated.**

Root cause traced to `build_run18_split.py`: it emitted **window-local** obb
coords against **full-frame** `file_name`s, so the cacher (which tiles the whole
`file_name`) placed every target ~`tile_origin` (~1800 px) off the real streak.
Proof: streak-on-target line contrast was -0.02 (chance) as-is, +15.87 once
`tile_origin` was added. Fix = `scripts/build_atwood_window_dataset.py`, which
materialises each window as a crop (obb and pixels share one frame; validated
contrast +16). New dataset: `train_atwood_synth_window_v1` /
`val_atwood_window_v1`. Retrain via `scripts/train_window_v1_pipeline.sh` (both
backbones, same Recipe R) → expect val_dice to recover toward ~0.77.

Related: `agent_docs/run18_vitb_handoff.md`, memory `eval-pipeline-overhaul`,
`project-run17-state`.
