# GTImages Synthetic Dataset Manifest

- Seed: `42`
- FITS normalization: `zscale`
- Synthetic ratio: `0.1`
- GTImages P75 streak length: `815.225` px
- All strata: `{'long_streak': 145, 'no_streak': 91, 'short_streak': 433}`
- Train strata: `{'long_streak': 102, 'no_streak': 64, 'short_streak': 303}`
- Val strata: `{'long_streak': 29, 'no_streak': 18, 'short_streak': 87}`
- Test strata: `{'long_streak': 14, 'no_streak': 9, 'short_streak': 43}`

## Outputs

- `train_real`: `data/annotations/gtimages_train_real.json`
- `train_synth_paper_long`: `data/annotations/gtimages_train_synth_paper_long.json`
- `train_synth_adapted`: `data/annotations/gtimages_train_synth_adapted.json`
- `val`: `data/annotations/gtimages_val.json`
- `test`: `data/annotations/gtimages_test.json`
- `synthetic_dir`: `data/gtimages_synthetic`
