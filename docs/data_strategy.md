# Data Strategy

## Core principles

- Use reviewed real FITS observations as the primary training signal. Preserve
  source-night provenance so annotations can be traced back to raw frames.
- Balance training data across streak length (short < 50 px, medium 50–400 px,
  long > 400 px), brightness, orientation, and background conditions.
- Synthetic examples may supplement scarce bands (currently short streaks) but
  must never enter validation or test splits.
- All annotations are endpoint segments (`x1, y1, x2, y2`). Convert historical
  OBB/polygon labels through `training.annotation_endpoints` at ingestion; never
  let the old format propagate further.
- Canonical evaluation uses `val_balanced_v1_no_sattrains.json` (241 images,
  247 annotations). Do not modify this file mid-experiment; create a new versioned
  val set if the annotation set changes substantially.

---

## Data sources

### BrentImages (primary)
Ongoing FITS captures from Atwood Observatory. New batches arrive in
`/Volumes/External/TrainingData/raw/BrentImages/Img_YYYYMMDD_Atwood/`.
Each batch gets its own COCO annotation JSON inside the folder, containing
reviewed streak endpoints for positive frames plus negative frame entries.

### Synthetic short streaks
Rendered at dataset build time by `scripts/build_atwood_window_dataset.py`
(`--n-synth-short 400`). These are short centerline segments embedded in real
sky patches to address the short-streak scarcity in natural observations.

---

## Exclusion policy: satellite trains

**Definition.** A frame is a satellite train event if it contains ≥ 2 annotated
streaks satisfying both:
- `angle_diff < 5°` (nearly parallel), **and**
- `perp_dist < 30 px` (the tracks are less than 30 pixels apart)

Satellite trains (Starlink and similar constellations) produce multiple
near-parallel streaks in a single frame. Including them in training is harmful
because adjacent streak heatmaps bleed into each other, blurring the model's
understanding of individual track boundaries. They are also statistically rare in
normal astronomical observing and not representative of the single-streak detection
use case.

**Exclusion manifest.** All excluded source files are recorded in:

```
/Volumes/External/TrainingData/annotations/sat_train_excluded.json
```

The file lists each excluded `file_name` and the detection criterion so the
decision is auditable.

**When adding new data**, run the satellite-train check before merging into the
source annotation (see the integration workflow below). Any new frames that
trigger the criterion should be appended to the manifest rather than silently
omitted.

---

## Merged source annotation

`all_train_run17_merged_no_sattrains.json` is the canonical training source (as
of June 2026):

- 6052 images, 12 329 annotations
- Satellite-train frames excluded (53 frames removed from the raw merged file)
- Used as `--source` for `scripts/build_atwood_window_dataset.py`

The older `all_train_run17_merged.json` (6105 images) is kept for reference but
should not be used for new training runs.

---

## Integrating a new batch of BrentImages

### 1. Copy raw data
```
/Volumes/External/TrainingData/raw/BrentImages/Img_YYYYMMDD_Atwood/
```
Follow the existing `Img_YYYYMMDD_Atwood` naming convention.

### 2. Apply dark calibration

Subtract the master dark before annotating or feeding frames into any pipeline
step. The master dark lives at:

```
/Volumes/External/TrainingData/raw/BrentImages/masterDark_0.50s.fit
```

It was created from the ZWO ASI2600MM Pro at 0.50 s exposure (6248 × 4176,
`int16` raw). Apply it to every science frame whose `EXPTIME` header matches
0.50 s; flag or skip frames with a different exposure time until a matching dark
is available.

```python
from astropy.io import fits
import numpy as np, pathlib

DARK_PATH = "/Volumes/External/TrainingData/raw/BrentImages/masterDark_0.50s.fit"
dark_data = fits.getdata(DARK_PATH).astype(np.float32)

def calibrate_frame(fits_path: str | pathlib.Path) -> np.ndarray:
    """Return dark-subtracted float32 array, clipped to ≥ 0."""
    with fits.open(fits_path) as hdul:
        exptime = float(hdul[0].header.get("EXPTIME", 0))
        if abs(exptime - 0.50) > 0.01:
            raise ValueError(f"{fits_path}: EXPTIME={exptime} does not match dark (0.50 s)")
        science = hdul[0].data.astype(np.float32)
    return np.clip(science - dark_data, 0, None)
```

Write calibrated arrays to a `calibrated/` sub-directory inside the batch
folder, or apply on-the-fly during annotation review — whichever fits the
workflow. **Never overwrite the raw `.fit` files.**

If the batch contains frames from a different setup (different exposure, binning,
or sensor temperature range), note the mismatch in the batch README and hold
those frames out until a matched dark is produced.

### 3. Annotate
Review each frame. For each positive frame, record the streak endpoints as a COCO
annotation JSON inside the batch directory. Negative frames should also be listed
as images with an empty annotations array so they can contribute background tiles
during training.

Use `scripts/annotate.py` for the annotation workflow.

### 4. Check for satellite trains
Run the exclusion check against the new batch's annotation file:

```bash
python - <<'PYEOF'
import json, math, pathlib
from collections import defaultdict

batch_ann = "/Volumes/External/TrainingData/raw/BrentImages/Img_YYYYMMDD_Atwood/annotations.json"
data = json.loads(pathlib.Path(batch_ann).read_text())
id_to_img = {int(img['id']): img for img in data['images']}
anns_by_img = defaultdict(list)
for ann in data['annotations']:
    anns_by_img[int(ann['image_id'])].append(ann)

def ann_endpoints(ann):
    seg = ann.get('segmentation', [])
    if seg and isinstance(seg[0], list) and len(seg[0]) >= 8:
        pts = seg[0]; corners = [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]
        best = 0; x1 = x2 = y1 = y2 = 0
        for i in range(len(corners)):
            for j in range(i+1, len(corners)):
                d = (corners[i][0]-corners[j][0])**2+(corners[i][1]-corners[j][1])**2
                if d > best: best=d; x1,y1=corners[i]; x2,y2=corners[j]
        return x1, y1, x2, y2
    b = ann.get('bbox', [0,0,1,1]); return b[0], b[1], b[0]+b[2], b[1]+b[3]

flagged = []
for img_id, anns in anns_by_img.items():
    if len(anns) < 2: continue
    segs = []
    for ann in anns:
        x1,y1,x2,y2 = ann_endpoints(ann)
        dx=x2-x1; dy=y2-y1
        if math.hypot(dx,dy) < 10: continue
        segs.append({'angle': math.degrees(math.atan2(dy,dx)) % 180,
                     'mx': (x1+x2)/2, 'my': (y1+y2)/2})
    for i in range(len(segs)):
        for j in range(i+1, len(segs)):
            a, b = segs[i], segs[j]
            da = abs(a['angle']-b['angle']); da = min(da, 180-da)
            if da > 5: continue
            ux = math.cos(math.radians(a['angle'])); uy = math.sin(math.radians(a['angle']))
            dx = b['mx']-a['mx']; dy = b['my']-a['my']
            if abs(dx*(-uy)+dy*ux) < 30:
                flagged.append(id_to_img[img_id]['file_name']); break
        else: continue
        break

print(f"{len(flagged)} satellite-train frames found:")
for f in flagged: print(f"  {f}")
PYEOF
```

Any flagged frames must be added to `sat_train_excluded.json` and excluded from
the merge in step 5.

### 5. Merge into the source annotation

```bash
python scripts/merge_brentimages_batch.py \
  --base     /Volumes/External/TrainingData/annotations/all_train_run17_merged_no_sattrains.json \
  --add      /Volumes/External/TrainingData/raw/BrentImages/Img_YYYYMMDD_Atwood/annotations.json \
  --exclude-manifest /Volumes/External/TrainingData/annotations/sat_train_excluded.json \
  --output   /Volumes/External/TrainingData/annotations/all_train_run18_merged_no_sattrains.json
```

The script re-assigns image and annotation IDs to avoid collisions, skips any
frame whose `file_name` appears in the exclusion manifest, skips exact duplicates,
and writes a `provenance` block into the output JSON recording every source used.

### 6. Rebuild the tile dataset
```bash
python scripts/build_atwood_window_dataset.py \
  --version <N+1> \
  --source /Volumes/External/TrainingData/annotations/all_train_run<N>_merged_no_sattrains.json \
  --eval-frames-json /Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json \
  --val-frac 0.08 --neg-frac 0.42 --bg-per-frame 3 --seed 42
```

This creates fresh self-contained NPY tile directories:
```
train_atwood_synth_window_v<N+1>/
val_atwood_window_v<N+1>/
```

### 7. Train
Follow the standard pipeline in `agent_docs/ml_pipeline.md`, pointing
`TRAIN_ANN` and `VAL_ANN` at the new v<N+1> annotation files. Use a fresh
`TAG` so weights and results land in a uniquely named directory.

### 8. Evaluate and compare
Run geometry metrics at `t=0.85, pf=0.85, ppf=0.85` against
`val_balanced_v1_no_sattrains.json` and compare against the previous production
model using `scripts/compare_geometry_evals.py --md`.

---

## Updating the canonical eval set

`val_balanced_v1_no_sattrains.json` should be stable across training runs. Only
create a new versioned eval set when:

- A significant number of annotation errors are found in the current set.
- New annotation categories or difficulty tiers are added.
- The val set has grown noticeably stale relative to the production image
  distribution.

When creating a new eval set, run the satellite-train check on it before
finalising, and update all references in `agent_docs/ml_pipeline.md` and
`agent_docs/datasets.md`.
