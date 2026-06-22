# ARGUS
**ARGUS** — **Automated Recognition and Grading of Unidentified Streaks**.
An end-to-end pipeline that detects satellite streaks in FITS telescope images. 
Also a reference to **Argus Panoptes** (Ἄργος Πανόπτης), the hundred-eyed giant of Greek mythology — always vigilant, always watching. 
Every detection is a line segment defined by two image-space endpoints (`x1`, `y1`, `x2`, `y2`).
When WCS is available those endpoints are resolved to sky coordinates and matched
against a local TLE catalog using SGP4 propagation. A FastAPI backend and React
frontend expose the full pipeline.

## How It Works

### Detection

Images are processed as overlapping 400 px tiles so that short streaks remain
visible at native scale. A frozen **DINOv3 ViT-S/16** backbone extracts patch
features from each tile; a lightweight convolutional head turns those features
into a one-channel **centerline heatmap** that is bright where a streak axis
runs and dark everywhere else. Connected bright regions are reduced to their
principal-axis endpoints, remapped to full-frame coordinates, and passed through
a suppression step that collapses duplicate detections from overlapping tiles.
Nearby collinear fragments are stitched into single segments.

### Why Endpoints, Not Boxes

A satellite streak is a 1D structure — a directed line across the sensor. Bounding
boxes and oriented rectangles introduce a width dimension that carries no physical
meaning and makes matching and evaluation unnecessarily complex. Using raw endpoints
preserves the information that matters (position, angle, extent) and keeps every
downstream step — annotation, training targets, post-processing, evaluation, and
API output — in the same coordinate space.

### Loss Function: ASL + clDice

Early models reached >97% recall but precision was poor; blob-shaped heatmap
activations triggered detections on noise and background gradients. Two changes
fixed this:

- **Asymmetric Loss (ASL)** zeroes out the gradient contribution from easy
  negatives, focusing training on the genuinely ambiguous background pixels.
- **clDice (Centerline Dice)** computes loss via the soft morphological skeleton
  of the prediction, rewarding thin connected linear responses and penalising any
  prediction that spreads laterally or forms a blob.

Together they brought detection precision from ~45% to **91.8%** with no recall
regression. See [`docs/loss_ablation_v9_v10_postmortem.md`](docs/loss_ablation_v9_v10_postmortem.md)
for the full ablation.

### Identification

Both endpoints of a matched detection are transformed to RA/Dec through the
frame's WCS solution. The resulting sky track and observation timestamp are
compared against pre-propagated TLE positions stored in a local SQLite catalog.
No live catalog query is required at inference time.

## Tech Stack and Architecture

```
FITS image
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Inference pipeline  (Python / PyTorch / astropy)   │
│                                                     │
│  FITS load → tile → DINOv3 ViT-S backbone (frozen)  │
│           → conv head → centerline heatmap          │
│           → connected-component endpoints           │
│           → stitch + dedup → WCS → sky coords       │
│           → SGP4 TLE match                         │
└──────────────────────┬──────────────────────────────┘
                       │  REST + WebSocket
                       ▼
           ┌───────────────────────┐
           │  FastAPI  (Python)    │
           │  SQLite  (SQLAlchemy) │
           └──────────┬────────────┘
                      │  JSON API
                      ▼
           ┌───────────────────────┐
           │  React + Tailwind CSS │
           │  Vite dev server      │
           └───────────────────────┘
```

| Layer | Technology |
|---|---|
| ML backbone | DINOv3 ViT-S/16 (frozen), PyTorch |
| Conv head | Custom single-channel heatmap head (PyTorch) |
| Image I/O | `astropy.io.fits`, `scipy` |
| Astrometry | `astropy` WCS |
| Satellite catalog | `sgp4` propagation, local SQLite TLE store |
| API | FastAPI, SQLAlchemy (async), SQLite |
| Frontend | React 19, Tailwind CSS v4, Vite |
| Environment | Conda (`satid`), Python 3.10+ |

## Data and Annotation

Training data is drawn from two sources:

- **Atwood Observatory FITS** — real observations captured across six sessions
  (`Img_20260412`, `Img_20260515`, `Img_20260527`, `Img_20260528`,
  `20260530`, `Geo_20260520`), totalling approximately 2,270 frames.
- **Synthetic augmentation** — ~3,840 frames generated from rendered streak
  composites (`argus_run13_npy`) to supplement scarce short-streak examples.

The merged training annotation pool (`all_train_run17_merged.json`) contains
**6,105 frames** — 5,958 positive and 147 negative — with **12,647 endpoint
annotations**. Streak lengths range from very short (~20 px) to long tracks
spanning most of the frame.

Positive frames receive oriented bounding box annotations in Frigate that are
immediately converted to endpoint pairs (`x1, y1, x2, y2`). Negative frames
(clear sky, no streaks) are retained as hard-negative tiles during dataset
construction. All annotation file paths are relative to `ARGUS_DATA_ROOT`.

A fixed 8% validation split and 42% negative-tile fraction are held constant
across experiments so results are directly comparable. Synthetic frames are only
used in training, never in the validation or test splits.

## Training

The backbone is always frozen; only the conv head is trained. Features are cached
to disk before training so the backbone forward pass runs **once per dataset**,
not once per epoch. This makes a 40-epoch run complete in minutes rather than
hours on an M-series Mac.

The full pipeline:

1. **Build a window dataset** from the merged annotation pool
2. **Cache ViT-S features** once for train and validation splits
3. **Train the conv head** with ASL + clDice loss, cosine LR schedule
4. **Evaluate** with `eval.geometry_metrics` on `val_balanced_v1.json`

See [`DEVELOPER.md`](DEVELOPER.md) for the full commands.

## Results

**Model: `vits_v9_asl_cldice`** — DINOv3 ViT-S/16, ASL + clDice loss, 400 px
tiles, evaluated on `val_balanced_v1.json` at threshold = 0.70, peak floor = 0.85.

### Detection

| Metric | Value |
|---|---|
| Recall | **97.9%** (234 / 239) |
| Precision | **91.8%** |
| False positives | 21 |

| Band | GT | Found | Recall |
|---|---|---|---|
| Short (< 50 px) | 9 | 8 | 88.9% |
| Medium (50–400 px) | 50 | 49 | 98.0% |
| Long (> 400 px) | 180 | 177 | 98.3% |

### Geometry (T2 raw OBB — production configuration)

| Metric | Mean | Median | P90 |
|---|---|---|---|
| Angle error | 0.37° | 0.19° | 0.99° |
| Endpoint error | 22 px | 16 px | 24 px |

Radon-based angle refinement (T3) was evaluated and found to degrade both
metrics significantly (angle error 0.37° → 10°). T2 raw geometry is used in
production.

### Backbone Comparison

ViT-B/16 was tested on identical data and hyperparameters (`vitb_v10_asl_cldice`).
ViT-S outperformed it on every metric. Both backbones use 16×16 patches and
produce a 32×32 spatial feature grid at 518 px input; ViT-B's wider feature
vectors do not translate to higher spatial resolution, and the same conv head
compresses them more aggressively. ViT-S is the production backbone.

## Repository Structure

```
inference/           — FITS loading, heatmap detection, post-processing, WCS, cross-ID
models/plain_dinov3/ — DINOv3 ViT heatmap model definition
training/            — endpoint datasets and cached-feature trainers
eval/geometry_metrics.py — canonical segment evaluator
scripts/             — dataset preparation, caching, evaluation, operations
api/, db/, frontend/ — FastAPI backend, SQLite persistence, React frontend
src/                 — classical astrometry and satellite matching components
docs/                — postmortems and experiment notes
agent_docs/          — developer and assistant reference guides
results/             — per-run geometry_eval.json files (committed)
weights/             — model checkpoints (gitignored)
```

## Acknowledgements and Citations

ARGUS builds on the following research and prior work:

- **DINOv3** by Meta AI Research provides the frozen visual backbones used by
  the heatmap detectors. See the
  [DINOv3 repository and license](https://github.com/facebookresearch/dinov3)
  and [Siméoni et al. (2025)](https://arxiv.org/abs/2508.10104).
- The training objective implements **Asymmetric Loss** from
  [Ridnik et al. (2021)](https://arxiv.org/abs/2009.14119) and **clDice** from
  [Shit et al. (2021)](https://arxiv.org/abs/2003.07311).
- The endpoint representation, segment-matching methodology, and optional
  Radon angle-refinement approach were informed by **StreakMind**.
- Cross-identification refinements, including midpoint scoring,
  along-track/cross-track residuals, direction disambiguation, expected streak
  length, and detection-quality conventions, were adapted from **SkyTrack**
  with acknowledgement to its author and contributors.
- The Gaussian satellite cross-identification confidence-scoring approach was
  informed by the work of **Danarianto et al.**

---

For environment setup, weight downloads, annotation workflow, training commands,
API/UI startup, and TLE catalog management, see [`DEVELOPER.md`](DEVELOPER.md).
