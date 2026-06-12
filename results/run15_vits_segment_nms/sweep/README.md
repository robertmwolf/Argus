# Run 15 (ViT-S) Threshold Sweep

**Date:** 2026-06-12  
**Dataset:** 20 clean single/double-streak FITS images, `~/val_sweep_subset/` (random.seed(42) from combined train+val sources)  
**Streak lengths:** 98–1094px, covering short/medium/long bands  
**Eval flags:** `--tiled --stitch --norm-mode zscore --threshold 0.05 --threshold-sweep 0.2 0.3 0.4 0.5 0.6 0.7`  
**Fixes applied:** angle_tol=15°, streak_length_px recomputed from merged OBB endpoints

| Threshold | Precision | Recall | F1    | Short F1 | Medium F1 | Long F1 | N preds |
|-----------|-----------|--------|-------|----------|-----------|---------|---------|
| 0.20      | 0.100     | 0.132  | 0.114 | 0.000    | 0.000     | 0.081   | 50      |
| 0.30      | 0.139     | 0.132  | 0.135 | 0.000    | 0.000     | 0.100   | 36      |
| 0.40      | 0.143     | 0.132  | 0.137 | 0.000    | 0.000     | 0.102   | 35      |
| 0.50      | 0.147     | 0.132  | 0.139 | 0.000    | 0.000     | 0.103   | 34      |
| 0.60      | 0.179     | 0.132  | 0.151 | 0.000    | 0.000     | 0.115   | 28      |
| **0.70**  | **0.208** | **0.132** | **0.161** | 0.000 | 0.000 | **0.125** | 24 |

**Recommended threshold:** 0.70 (best F1)

**Observations:**
- Recall is flat across all thresholds — the model misses ~87% of streaks at all operating points
- Only long streaks (≥400px) are detected; medium and short recall is 0%
- Precision improves steadily with threshold (fewer FPs), recall unaffected
- Low overall recall likely reflects training data coverage gaps, not threshold sensitivity
