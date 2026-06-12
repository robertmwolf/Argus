# Run 17 (ViT-B) Threshold Sweep

**Date:** 2026-06-12  
**Dataset:** 20 clean single/double-streak FITS images, `~/val_sweep_subset/` (random.seed(42) from combined train+val sources)  
**Streak lengths:** 98–1094px, covering short/medium/long bands  
**Eval flags:** `--tiled --stitch --norm-mode zscore --threshold 0.05 --threshold-sweep 0.2 0.3 0.4 0.5 0.6 0.7`  
**Fixes applied:** angle_tol=15°, streak_length_px recomputed from merged OBB endpoints

| Threshold | Precision | Recall | F1    | Short F1 | Medium F1 | Long F1 | N preds |
|-----------|-----------|--------|-------|----------|-----------|---------|---------|
| 0.20      | 0.018     | 0.105  | 0.030 | 0.000    | 0.118     | 0.008   | 229     |
| 0.30      | 0.018     | 0.105  | 0.030 | 0.000    | 0.118     | 0.008   | 224     |
| 0.40      | 0.018     | 0.105  | 0.031 | 0.000    | 0.118     | 0.008   | 221     |
| 0.50      | 0.020     | 0.105  | 0.034 | 0.000    | 0.118     | 0.009   | 201     |
| **0.60**  | **0.143** | **0.105** | **0.121** | 0.000 | **0.118** | 0.043 | 28 |
| 0.70      | 0.160     | 0.105  | 0.127 | 0.000    | 0.118     | 0.046   | 25      |

**Recommended threshold:** 0.60 (precision jumps sharply, recall unchanged; F1=0.121)

**Observations:**
- Recall flat at 10.5% across all thresholds; model misses ~89% of streaks
- Uniquely detects medium streaks (F1=0.118) where Run 15 ViT-S gets 0% — likely a backbone capacity effect
- Large FP flood below t=0.60 (200+ predictions), precision collapses; sharp cliff at 0.60
- Short recall is 0% at all thresholds
- Despite val_dice=0.105 (undertrained), ViT-B medium recall outperforms the better-trained ViT-S
