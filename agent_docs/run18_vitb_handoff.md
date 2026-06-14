# Run 18 — ViT-B heatmap retrain handoff

**Status (2026-06-12):** ViT-B (Run 17) is undertrained and not deployable. This
doc is the handoff for retraining it correctly now that the evaluation pipeline
has been overhauled to measure it honestly.

## Why retrain

- **Run 17 ViT-B val_dice = 0.105** vs **Run 15 ViT-S val_dice = 0.770**. The
  ViT-B head never converged: train_dice plateaued ~0.37 from epoch 30, val_dice
  stuck ~0.105. The 768-dim ViT-B features carry more capacity than ViT-S's
  384-dim, but the frozen-backbone cached-head training run did not exploit it.
- Despite undertraining, earlier sweeps hinted ViT-B picks up **medium-band**
  streaks where ViT-S does not — the reason it's worth fixing rather than
  abandoning. Confirm this on the new balanced eval once retrained.

## What changed around it (so Run 18 is measured correctly)

The Run 17 eval numbers were untrustworthy for reasons unrelated to the model.
All fixed as of 2026-06-12:

1. **Architecture-aligned bands** (`eval/streak_metrics.py`): short [50,400),
   medium [400,1000), long [1000+). 400 = native tile size (single-tile vs
   multi-tile/stitch boundary); 50 = ~4-patch resolvability floor. Streaks
   <50px are dropped from GT and predictions (Frigate-scale, sub-resolvable).
2. **Balanced eval set** `data/annotations/val_balanced_v1.json` — 59 short /
   100 medium / 80 long, Frigate excluded. Rebuild: `scripts/build_balanced_eval.py`.
3. **Fixed stitch** (`inference/tiled_pipeline.py`): the union-find transitivity
   blow-up that turned clean streaks into frame-spanning blobs is fixed
   (guarded greedy merge; angle_tol 10°, gap 200, conf_floor 0.5).
4. **Peak/top-K filter** (`inference/postprocess.py::filter_peak_topk`): drops
   the heatmap noise floor by peak activation. Env: `VITB_HEATMAP_PEAK_FLOOR`,
   `VITB_HEATMAP_TOPK`.
5. **Zero-GPU threshold sweep**: `scripts/cache_heatmap_maps.py` →
   `evaluate_dinov3_heatmap.py --heatmap-cache`. Never re-run the backbone per
   threshold.

## ROOT CAUSE — CORRECTED 2026-06-13 (Run 18 result + audit)

**Run 18 (the consistent-data retrain) also failed**: train_dice 0.13, val_dice
0.12, eval 0.000 on val_balanced_v1. So **data inconsistency was NOT the cause**
(or not the whole one). A mechanical + optimization audit (2026-06-13) found:

- **No mechanical bug.** ViT-B checkpoint loads 0 missing / 0 unexpected keys;
  cached features sane and comparably scaled to ViT-S (std 0.44 vs 0.42); head
  `in_channels` auto-inferred to 768; zscore norm parity holds across
  cache/train/eval.
- **No feature deficiency.** Overfit probe: a frozen ViT-B head fits 12 streak
  tiles to **train_dice 0.95** (vs ViT-S 0.68) given enough steps. The frozen
  ViT-B features carry the streak signal *better* than ViT-S. **This disproves
  the earlier "unfreeze the backbone" recommendation — do NOT unfreeze.**
- **Real cause = UNDERTRAINING.** ViT-B's 768-d head converges slower per step.
  Run 17 (40 ep) crept to train_dice 0.37; Run 18 (22 ep, cosine T_max=22)
  strangled the LR before the head fit, landing at 0.13. ViT-S fit to 0.81 in
  the same code because 384-d converges faster.

Evidence table (identical training code, only the backbone/budget differ):

| Run | backbone | epochs | final train_dice | best val_dice |
|-----|----------|-------:|-----------------:|--------------:|
| 15  | ViT-S    | 40     | 0.813            | 0.770 |
| 17  | ViT-B    | 40     | 0.374            | 0.105 |
| 18  | ViT-B    | 22     | 0.131            | 0.122 |

**Fix = more training + hotter LR with warmup** (Run 19). Head training is cheap
(frozen backbone, cached features); the ~3.5 hr cost is the one-time caching.

### Historical note (superseded)
The 2026-06-12 hypothesis below — that Run 17 failed from training on mixed
FITS+PNG+synthetic-NPY (70% synthetic) vs pure-FITS validation — was plausible
but **wrong**: Run 18 on a clean consistent split failed the same way. The
consistent split (train_run18.json/val_run18.json) is still the right data; it
just needed the corrected training budget.

**Corrected pipeline (Run 19): `bash scripts/run19_vitb_pipeline.sh`**
- Same consistent split `data/annotations/{train,val}_run18.json` (still correct:
  real FITS only; synthetic streaks rendered ON real FITS; ~38% negative tiles;
  stratified by band; split by frame; val/test_atwood excluded).
- Corrected training budget: **epochs 80, lr 3e-3, 4-epoch linear warmup + cosine,
  pos_weight 20** (vs Run 18's 22 ep / default lr / no warmup). `--warmup-epochs`
  added to `train_dinov3_heatmap_cached.py`.
- **KEEPS** the train/val feature cache at `~/argus_run19_cache` on exit (Run 18
  deleted it) so LR/epoch re-sweeps are minutes, not another 3.5 hr re-cache.
- best.pt/latest.pt + history.json every epoch; Ctrl-C → best.pt, or `--resume`.
- **Success signal:** train_dice must climb past ~0.6 then toward ~0.8 (ViT-S
  baseline 0.81). If it plateaus well below, escalate to a larger head
  (`--hidden-channels`) — NOT a backbone unfreeze (probe disproved that).

## Recipe deltas to try (if the consistency fix alone isn't enough)

Run 17 baseline recipe: ViT-B/16, 400px tiles, zscore norm, frozen backbone +
cached-head, 40 epochs. Changes to try, roughly in priority order:

1. **Train longer / larger head.** val_dice was still flat at epoch 35 — the
   head likely lacks capacity or training time. Increase `hidden_channels`
   and/or epochs; watch for val_dice actually moving past ~0.11.
2. **Partial backbone fine-tune.** Frozen ImageNet/LVD features may not separate
   FITS streak signal at ViT-B scale. Unfreeze the last 2–4 transformer blocks
   with a low LR (cached-features training cannot do this — needs the live
   backbone in the loop).
3. **LR / warmup sweep.** A flat loss often means LR too low (or too high early).
   Try a short cosine warmup + higher peak LR.
4. **Confirm norm parity.** Training and inference must both be zscore. Verify
   the cache build and `VITB_HEATMAP_*` inference both use zscore.
5. **More medium-streak training data.** The medium band is the differentiator;
   bias augmentation toward 400–1000px streaks.

## Success gate

Beat **Run 15 ViT-S short-band recall on `val_balanced_v1`** — the short band is
single-tile and stitch-independent, so it is the cleanest measure of raw model
capability. Secondary: match-or-beat ViT-S medium-band recall (ViT-B's intended
advantage). Measure through the fixed pipeline below.

**Run 15 ViT-S baseline on val_balanced_v1** (fixed pipeline, peak-floor 0.85,
threshold 0.7 — see `results/run15_vits_segment_nms/balanced_v1/README.md`):

| Metric | short | medium | long | overall |
|--------|------:|-------:|-----:|--------:|
| recall | **0.746** | 0.850 | 0.938 | 0.933 |
| F1     | 0.261 | 0.791 | 0.893 | 0.619 |

Run 18 must beat **short recall 0.746** (single-tile, stitch-independent — the
clean model-capability number) to justify shipping over ViT-S.

## Eval commands (the fixed workflow)

```bash
PY=/Users/robert/miniconda3/envs/satid/bin/python
ANN=data/annotations/val_balanced_v1.json

# 1. Cache heatmaps once (backbone pass).
$PY scripts/cache_heatmap_maps.py --annotations $ANN \
    --checkpoint weights/run18_vitb/best.pt \
    --output-dir ~/argus_run18_bal_cache --norm-mode zscore

# 2. Zero-GPU threshold sweep through the fixed stitch + peak filter.
$PY scripts/evaluate_dinov3_heatmap.py --tiled --stitch \
    --heatmap-cache ~/argus_run18_bal_cache \
    --annotations $ANN --norm-mode zscore \
    --threshold 0.05 --threshold-sweep 0.5 0.6 0.7 0.8 \
    --peak-floor 0.0 \
    --output results/run18_vitb/balanced_v1/metrics_placeholder.json

# 3. Summarise.
$PY scripts/summarize_balanced_eval.py results/run18_vitb/balanced_v1
```

Related: [[project-run17-state]], `agent_docs/heatmap_training_lessons.md`.
