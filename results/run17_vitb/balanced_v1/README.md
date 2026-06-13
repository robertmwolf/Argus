# Run 17 (ViT-B) — val_balanced_v1 eval

**Date:** 2026-06-12
**Eval set:** `data/annotations/val_balanced_v1.json` — 59 short / 100 medium / 80 long.
**Pipeline:** same fixed stitch + peak/top-K + new bands as Run 15.

## Results

| Setting | Prec | Recall | F1 | short-R | med-R | long-R | Npred |
|---------|------|--------|-----|---------|-------|--------|-------|
| pf0 @0.5  | 0.150 | 0.586 | 0.239 | 0.576 | 0.540 | 0.412 | 1341 |
| pf0 @0.6  | 0.104 | 0.615 | 0.178 | 0.610 | 0.520 | 0.438 | 2040 |
| pf0 @0.7  | 0.172 | 0.569 | 0.265 | 0.610 | 0.480 | 0.287 | 1196 |
| pf0 @0.8  | 0.162 | 0.448 | 0.238 | 0.559 | 0.390 | 0.175 | 1045 |
| **pf85 @0.5** | **0.338** | **0.603** | **0.433** | 0.576 | 0.560 | 0.438 | 494 |
| pf85 @0.6 | 0.317 | 0.615 | 0.418 | 0.610 | 0.520 | 0.438 | 571 |
| pf85 @0.7 | 0.261 | 0.569 | 0.357 | 0.610 | 0.480 | 0.287 | 722 |
| pf85 @0.8 | 0.169 | 0.448 | 0.245 | 0.559 | 0.390 | 0.175 | 933 |

## Verdict: undertrained, confirmed

ViT-B (val_dice 0.105) is **worse than ViT-S in every band** on the corrected
eval:

| band recall | ViT-B (best) | ViT-S (best) |
|-------------|-------------:|-------------:|
| short  | 0.610 | 0.746 |
| medium | 0.560 | 0.870 |
| long   | 0.438 | 0.950 |
| overall | 0.615 | 0.933 |

**This overturns the earlier belief** (from the broken metric) that ViT-B
uniquely detects medium streaks. On the balanced set ViT-S dominates medium
0.85 vs 0.56. The earlier signal was a banding/stitch artifact, not a real
ViT-B advantage.

Two undertraining tells:
- **Non-monotonic pf0 sweep**: predictions go 1341 → 2040 → 1196 as the
  threshold *rises* 0.5 → 0.6 → 0.7. A converged heatmap shrinks monotonically
  with threshold; ViT-B's diffuse activation fragments erratically.
- **Long recall collapses** (0.44, vs ViT-S 0.95) — weak activation on long
  streaks the head never learned to light up cleanly.

The peak floor helps ViT-B's precision more than ViT-S's (0.150 → 0.338 at
t=0.5) precisely because it is noisier.

**Action:** retrain — see `agent_docs/run18_vitb_handoff.md`. Gate: beat ViT-S
short-band recall 0.746 (ViT-B is at 0.610).
