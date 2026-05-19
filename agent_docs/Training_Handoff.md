# ARGUS Training Handoff

## Two Training Routes

Phase D (DINOv3 ViT-L, frozen backbone, full dataset) can be executed via
either of two independent paths. Choose one — they produce the same target
artifacts and can be run simultaneously if both resources are available.

| | Route 1 — Colleague's Workstation | Route 2 — Cloud GPU Rental |
|-|-----------------------------------|---------------------------|
| **Hardware** | RTX 5070 Ti 16 GB, 24-core i9 (Windows/WSL2) | RTX 4090 24 GB (Vast.ai / RunPod) |
| **Est. cost** | $0 (colleague's machine) | ~$7–18 on-demand, $5–13 spot |
| **Est. time** | ~30–40 hrs (50 epochs, 512 px) | ~15–38 hrs (early stopping, 800 px) |
| **Image size** | 512 px (VRAM-constrained to 16 GB) | 800 px first; fall back to 512 px if OOM |
| **Setup** | [Route 1 — Workstation](#route-1--workstation-rtx-5070-ti) | [Route 2 — Cloud GPU](#route-2--cloud-gpu-rental-rtx-4090) |
| **Phase F fallback** | Rent A100 80 GB if targets missed | Included in the $50–150 aspirational path |

> **If both routes run simultaneously**, use separate `--work-dir` paths and
> separate results directories (e.g. `weights/run_5070ti_dinov3_vitl` vs
> `weights/run_4090_dinov3_vitl`) so outputs never collide.
>
> **Windows note**: Route 1 on a Windows workstation must use WSL2 with Ubuntu
> 22.04. Native Windows/PowerShell training is not supported because the
> MMDetection stack requires Linux mmcv CUDA ops.

---

## What Trains (Both Routes)

The DINOv3 backbone is frozen throughout, what specifically needs to be trained is the neck and head:

1. ChannelMapper neck (~4M params)
Converts DINOv3's flat 1024-dim patch grid into 4 feature pyramid levels at 256-dim each. This is just 4 × 1×1 convolutions — it has no pretrained weights and must learn from scratch.

2. Co-DINO detection head (~40M params)

The transformer encoder/decoder (6 layers each) that refines features into object queries
The classification head (streak vs background)
The bounding box regression head (cx, cy, w, h)
The denoising (DN) auxiliary head
None of these components have ever seen a satellite streak. They come from COCO pretrain weights (object detection on everyday images), so they need fine-tuning to learn what a streak looks like, where it is, and how to draw a tight box around a thin diagonal line.

The backbone is already done — the 1.7B-image pretraining that gave DINOv3 rich visual features is baked into the .pth file. The workstation training is purely teaching the neck and head to interpret those features in the context of streak detection on astronomical FITS images.

## Short Prompt for the Colleague's Codex

Give the colleague's Codex this instruction:

```text
Check out repo at https://github.com/robertmwolf/Argus.git
Read instructions at agent_docs/Training_Handoff.md
Follow that document to download training data, run DINOv3 ViT-L training
(Phase D), evaluate results, and optionally check the outputs back into GitHub.
```

## Route 1 — Workstation (RTX 5070 Ti)

The sections below through "Train DINOv3 ViT-L (Phase D — primary run)" cover
Route 1: the colleague's high-performance Windows workstation (WSL2 + RTX 5070 Ti).
For Route 2 (cloud GPU rental), jump to
[Route 2 — Cloud GPU Rental](#route-2--cloud-gpu-rental-rtx-4090).

The historical Lambda A100 path remains valid, but the data download flow and
expected output branch below are tailored to the 5070 Ti run.

## Windows Workstation Rule

Run the training stack inside **WSL2 Ubuntu 22.04**, even when the physical
machine is Windows. Do not install the training environment directly in native
Windows, Anaconda Prompt, or PowerShell.

PowerShell setup on the Windows host:

```powershell
wsl --install -d Ubuntu-22.04
```

After rebooting and opening the Ubuntu terminal, verify the GPU is exposed to
WSL2:

```bash
nvidia-smi
```

If `nvidia-smi` does not show the RTX GPU inside Ubuntu, update the Windows
NVIDIA driver with WSL CUDA support before continuing. Once `nvidia-smi` works,
treat the machine as Linux for every command in this document.

## Hardware Requirements

| Component | Minimum | Tested on |
|-----------|---------|-----------|
| GPU | 12 GB VRAM (CUDA) | NVIDIA RTX 5070 Ti 16 GB |
| CPU | 8 cores | Intel i9-24 core |
| RAM | 32 GB | — |
| Disk | 200 GB free | — |
| OS | Ubuntu 22.04 LTS, including WSL2 Ubuntu | WSL2 Ubuntu 22.04 on Windows |
| CUDA | 12.6+ | 12.8 recommended for Blackwell |
| Driver | 560+ | — |

The config is tuned for **16 GB VRAM** (batch_size=1, gradient accumulation=2,
mixed precision, gradient checkpointing). If you have more VRAM (e.g. A100 40 GB)
you can raise `batch_size` to 2 in `models/dino/streak_codino_swin_l.py` and
remove `accumulative_counts`.

## Dependency Model

The training machine should be reproducible from the repository plus the
external data and weight files listed in this document. Install in this order:

| Layer | Source | Why it is separate |
|-------|--------|--------------------|
| Python | conda env `satid`, Python 3.11 | Python 3.12 is not validated with mmcv/numpy compiled packages |
| PyTorch | PyTorch `cu128` wheel index | Must match RTX 50-series / Blackwell CUDA support |
| MMDetection | `mmengine==0.10.4`, `mmcv==2.1.0`, `mmdet==3.3.0` | `mmcv` must come from a CUDA-specific Linux wheel with compiled ops |
| ARGUS packages | `requirements.txt` | Project dependencies only; intentionally excludes torch/mmcv/mmdet |
| DINOv3 code | `git+https://github.com/facebookresearch/dinov3.git` | Required by `models/dino/dinov3_adapter.py` |
| Model weights | Shared handoff folder or Meta portal | Large `.pth` files are gitignored and not installed by pip |

`scripts/cloud_setup.sh` performs this order automatically for WSL2/Linux.
`scripts/prepare_cloud_training.py` verifies the high-risk pieces before a full
run: CUDA visibility, PyTorch version, mmcv compiled ops, DINOv3 import,
annotation files, model weights, and a training smoke test.

## Parallel Native PyTorch Spike

If the goal is only streak detection and not preserving the current
MMDetection/DETR architecture, there is a parallel plain-PyTorch DINOv3 spike in
`agent_docs/plain_dinov3_spike.md`. It uses a frozen DINOv3 encoder plus a
small heatmap head and does not require `mmcv`, `mmdet`, or `mmengine`.

## Dataset Decision — Read This First

**The trainer is expected to bring their own dataset.** The GTImages and
SatStreaks sources listed below are the legacy reference datasets used in
earlier phases. They may be retained, augmented, or entirely replaced depending
on the quality and coverage of the incoming dataset.

Before downloading or preparing any data, evaluate your dataset against these
criteria:

- **Coverage**: Does it include the streak morphologies (short, medium, long)
  and FITS/image formats the model must generalize to?
- **Label quality**: Are bounding boxes tight and consistent? Is the category
  set compatible with ARGUS (streak vs. background)?
- **Scale**: Is the dataset large enough to meet or exceed the current split
  counts (≈3,000 images)?  Larger is generally better.
- **Overlap**: Does it contain sufficient diversity to avoid distribution
  mismatch with real observation data?

**Decision protocol:**

1. Inspect a representative sample before committing to a conversion pipeline.
2. If the incoming dataset is high quality and large, **discard GTImages and
   SatStreaks entirely** — do not merge poor-quality legacy data just to inflate
   counts.
3. If the incoming dataset has gaps, selectively retain the legacy sources to
   fill them, noting the mixing rationale in `training_summary.md`.
4. Document your dataset decision clearly in `training_summary.md` (which
   dataset(s) were used, why, and what was discarded).

After the decision, continue below with whichever data applies.

## Evaluate Your Training Data

Before converting or merging anything, run these checks on the incoming dataset.
Stop and record any failing gate in `training_summary.md` before proceeding.

### 1. Annotation statistics

If the dataset ships in COCO format:

```bash
python - <<'PY'
import json, pathlib, sys

for split in ["train", "val", "test"]:
    p = pathlib.Path(f"data/annotations/{split}.json")
    if not p.exists():
        print(f"{split}.json — NOT FOUND")
        continue
    d = json.loads(p.read_text())
    imgs  = len(d.get("images", []))
    anns  = len(d.get("annotations", []))
    cats  = [c["name"] for c in d.get("categories", [])]
    empty = sum(1 for img in d["images"]
                if not any(a["image_id"] == img["id"] for a in d["annotations"]))
    print(f"{split:5s}  images={imgs:5d}  annotations={anns:5d}  "
          f"empty_images={empty:4d}  categories={cats}")
PY
```

Expected minimums after merging: train ≥ 2,000 images, val ≥ 400, test ≥ 400.
If `empty_images` exceeds 30% of any split, the negative-sample balance is off —
check `merge_annotations.py --val-fraction` and the source dataset masks.

### 2. Streak morphology distribution

ARGUS targets three morphology bands: short (<269 px diagonal), medium
(269–800 px), long (>800 px). A heavily skewed distribution will hurt recall
on the underrepresented band.

```bash
python - <<'PY'
import json, math, pathlib

for split in ["train", "val", "test"]:
    p = pathlib.Path(f"data/annotations/{split}.json")
    if not p.exists():
        continue
    anns = json.loads(p.read_text())["annotations"]
    short = medium = long_ = 0
    for a in anns:
        w, h = a["bbox"][2], a["bbox"][3]
        diag = math.hypot(w, h)
        if diag < 269:
            short += 1
        elif diag <= 800:
            medium += 1
        else:
            long_ += 1
    total = short + medium + long_ or 1
    print(f"{split:5s}  short={short:4d} ({short/total:.0%})  "
          f"medium={medium:4d} ({medium/total:.0%})  "
          f"long={long_:4d} ({long_/total:.0%})")
PY
```

Flag the result in `training_summary.md` if any band is below 10% of annotated
images — consider augmenting or rebalancing before training.

### 3. Annotation sanity — degenerate boxes

```bash
python - <<'PY'
import json, pathlib

for split in ["train", "val", "test"]:
    p = pathlib.Path(f"data/annotations/{split}.json")
    if not p.exists():
        continue
    anns = json.loads(p.read_text())["annotations"]
    bad = [a for a in anns if a["bbox"][2] < 2 or a["bbox"][3] < 2 or a["area"] < 4]
    print(f"{split:5s}  degenerate boxes={len(bad)}")
    for a in bad[:5]:
        print(f"        id={a['id']} bbox={a['bbox']} area={a['area']}")
PY
```

Any degenerate box (width or height < 2 px, area < 4 px²) should be removed
before training — they produce NaN losses in the regression head.

### 4. Image file integrity

```bash
python - <<'PY'
import json, pathlib
from PIL import Image

ann_file = pathlib.Path("data/annotations/train.json")
if not ann_file.exists():
    print("train.json not found — skipping")
    raise SystemExit

d = json.loads(ann_file.read_text())
missing = corrupt = ok = 0
for img in d["images"]:
    p = pathlib.Path(img["file_name"])
    if not p.exists():
        missing += 1
        if missing <= 5:
            print(f"MISSING  {p}")
        continue
    try:
        Image.open(p).verify()
        ok += 1
    except Exception as e:
        corrupt += 1
        print(f"CORRUPT  {p}: {e}")

print(f"ok={ok}  missing={missing}  corrupt={corrupt}")
PY
```

Zero missing and zero corrupt is the gate. Resolve path mismatches by checking
whether the annotation `file_name` field is absolute or relative and aligning
it with the actual image tree.

### 5. FITSStreakDataset iteration smoke test

Confirms the full PyTorch data pipeline loads without error on a sample of images:

```bash
USE_DEV_SUBSET=false ARGUS_NORM=autostretch python - <<'PY'
import os, sys
os.environ.setdefault("USE_DEV_SUBSET", "false")
os.environ.setdefault("ARGUS_NORM", "autostretch")

from training.dataset import FITSStreakDataset
import json, pathlib

ann = pathlib.Path("data/annotations/train.json")
ds  = FITSStreakDataset(ann_file=ann, img_dir=pathlib.Path("data"), transforms=None)
errors = 0
for i in range(min(50, len(ds))):
    try:
        ds[i]
    except Exception as e:
        print(f"[{i}] ERROR: {e}")
        errors += 1
print(f"Sampled 50 items — {errors} errors")
PY
```

Zero errors required before starting full training.

### 6. Decision gate summary

Record the following block in `training_summary.md` under "Dataset Decision":

```text
Dataset used: <name(s)>
Legacy datasets retained: GTImages=yes/no  SatStreaks=yes/no  Reason: <...>
Split counts: train=N  val=N  test=N
Empty images (train): N  (<pct>%)
Morphology distribution (train): short=N%  medium=N%  long=N%
Degenerate boxes removed: N
Image integrity: ok=N  missing=0  corrupt=0
FITSStreakDataset smoke test: PASS / FAIL
```

Do not start training until every field in this block is filled in and all
gates pass.

## Data Sources

The big datasets should be downloaded directly on the training workstation.
Do not copy the full local data tree through the shared handoff folder.

### Your dataset

Place your dataset images in `data/` following the structure expected by
`training/dataset.py`.  If your dataset ships with COCO-format annotations,
place them in `data/annotations/`.  If it uses a different format, write or
adapt a conversion script (see `scripts/convert_gtimages.py` as a reference)
and output `data/annotations/train.json`, `val.json`, and `test.json`.

### Legacy reference datasets (use only if needed per the Dataset Decision above)

GTImages direct download:

```text
https://1drv.ms/u/c/f9b9ba14546c7993/IQDsL-bDtjgrSZK8oBfpozNyAT1gfpMbgM3YUjbMJeZLMDU?e=puwV4T
```

SatStreaks upstream source:

```text
https://github.com/jijup/SatStreaks
```

SatStreaks dataset folder link from the upstream README:

```text
https://smuhalifax-my.sharepoint.com/:f:/g/personal/susrita_chatterjee_smu_ca/EsbHlOO3pMRKiN6yIZT54CoBaIaZSsHhYgRZswt-erqxmg?e=pcQ8Xk
```

### ARGUS lightweight handoff folder (annotations, catalogs, and trained weights)

```text
https://1drv.ms/f/c/f9b9ba14546c7993/IgBKfTqYuQuWTZcfBhvDWoARAUD1kC9YfTDE70F9rHKH-o8?e=w4EplD
```

This shared folder is the canonical storage location for both small project
files and all trained model weights.  Weights are **never committed to git**
(they are gitignored at the repo root).  Upload final checkpoints here so they
are accessible to the team without bloating the repository.

The ARGUS lightweight handoff folder should contain only small project-specific
files that are inconvenient to regenerate, not the full image datasets:

```text
Argus-training-handoff/
|-- data/
|   |-- annotations/
|   |   |-- train.json
|   |   |-- val.json
|   |   |-- test.json
|   |   |-- dev_subset.json
|   |   |-- gtimages.json
|   |   `-- gtimages_negatives.json
|   |-- catalogs/
|   |   `-- active_sats.tle
|   `-- tle_zips/
|-- weights/
|   |-- dinov3_vitl16_lvd1689m.pth.zip   ← ViT-L backbone (~1.0 GB zip, ~1.1 GB extracted)
|   |-- dinov3_vitb16_lvd1689m.pth.zip   ← ViT-B backbone (~327 MB, needed for Mac dev)
|   `-- co_dino_swin_l_coco.pth          ← Swin-L COCO pretrain (optional, ~828 MB)
`-- MANIFEST.txt
```

> **Note on backbone weights:** `dinov3_vitl16_lvd1689m.pth` is the critical
> file for Phase D training.  It is staged here because it requires Meta portal
> access to download directly and is too large to re-download reliably on a
> training workstation.  Always verify the file size (~1.1 GB) after extraction.

Required training inputs:

```text
data/satstreaks/Data/Images/
data/satstreaks/Data/Masks/
data/satstreaks/Data/labels.json
data/GTImages/
```

Small optional handoff inputs:

```text
data/annotations/
data/Manifest.txt
data/catalogs/active_sats.tle
data/tle_zips/
optional_weights/co_dino_swin_l_coco.pth
```

Do not stage these local-only or regenerated paths:

```text
data/uploads/
data/cache/
weights/run/
runs/
*.db
argus.db*
```

## Clone and Restore Data

Fork and clone the project:

1. On GitHub, fork `https://github.com/robertmwolf/Argus` to your own account.
2. Clone your fork:

```bash
git clone https://github.com/<your-username>/Argus.git
cd Argus
git remote add upstream https://github.com/robertmwolf/Argus.git
```

All training work lives in your fork. To pull in upstream code changes at any
point: `git pull upstream main`.

When training is complete, open a pull request from your fork targeting
`robertmwolf/Argus:main`. The PR should include **only** result files,
manifests, and checksums — not dataset prep commits or training-run artifacts.

Download GTImages from the direct OneDrive file link and extract it so the
files land here:

```text
Argus/data/GTImages/
```

Expected GTImages contents include `.fits`, `.wcs`, `.ini`, and `.strk` files.
If the archive extracts with an extra top-level folder, move the contents so
`data/GTImages/*.fits` exists.

Download SatStreaks from the upstream dataset source. Start from the upstream
repo and README:

```bash
git clone https://github.com/jijup/SatStreaks.git /tmp/SatStreaks
```

Then use the "Entire Dataset" link in `/tmp/SatStreaks/README.md`, or the
SharePoint link listed in this document, to download the full dataset. Extract
or copy it so these paths exist in ARGUS:

```text
Argus/data/satstreaks/Data/Images/
Argus/data/satstreaks/Data/Masks/
Argus/data/satstreaks/Data/labels.json
```

The SatStreaks source should provide about 3,074 image files and 3,073 mask
files. If those counts are materially different, stop and record the discrepancy
before training.

Copy any small files from the ARGUS lightweight handoff folder into the repo
root so these paths match exactly when present:

```text
Argus/data/annotations/
Argus/data/catalogs/
Argus/data/tle_zips/
```

If the optional Swin-L pretrained checkpoint is present, copy it to:

```text
weights/co_dino_swin_l_coco.pth
```

Otherwise, `scripts/cloud_setup.sh` or `scripts/download_weights.py` will
download it.

## Environment Setup

Run the project setup script:

```bash
chmod +x scripts/cloud_setup.sh
./scripts/cloud_setup.sh
```

Apply environment variables:

```bash
source ~/.bashrc
export MODEL_SIZE=large
export USE_DEV_SUBSET=false
export DATABASE_URL=sqlite+aiosqlite:///./argus.db
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For RTX 50-series Blackwell GPUs, PyTorch must be 2.6 or newer with a CUDA
build that supports the card. The pre-flight script checks this.

## Rebuild Annotation Splits

If `data/annotations/gtimages.json` or `data/annotations/gtimages_negatives.json`
is missing, rebuild them:

```bash
python scripts/convert_gtimages.py \
    --strk-dir data/GTImages \
    --output data/annotations/gtimages.json \
    --negatives-output data/annotations/gtimages_negatives.json
```

Then rebuild the combined train, validation, and test splits:

```bash
python scripts/merge_annotations.py --seed 42 --val-fraction 0.2
```

`merge_annotations.py` now materializes SatStreaks masks into real COCO
bounding boxes and areas at merge time. Missing or empty masks are skipped, so
if the final split counts differ materially from `data/Manifest.txt`, stop and
record the discrepancy before training.

Expected split files:

```text
data/annotations/train.json
data/annotations/val.json
data/annotations/test.json
```

## Optional: Dual-Track Weight Production

If you have proprietary data, consider producing two independent sets of weights
in the same training run so the project has both a sharable artefact and a
highest-performance artefact.

**Open-source track (`oss`)** — trained exclusively on GTImages and SatStreaks.
These weights can be uploaded to the shared OneDrive folder and attached to a
publication.

**Proprietary track (`prop`)** — trained on your incoming dataset, optionally
merged with the OSS sources to fill morphology gaps. These weights stay private
unless your data licence explicitly permits redistribution.

If you have no proprietary dataset, skip this section entirely and proceed with
the single training run described later.

### Step 1 — Prepare separate annotation splits

Run `merge_annotations.py` once per track, then save the outputs under a
track-specific prefix so the two splits never overwrite each other.

```bash
# ---- Open-source track (GTImages + SatStreaks only) ----
python scripts/merge_annotations.py --seed 42 --val-fraction 0.2 \
    --sources gtimages satstreaks
for split in train val test; do
    cp data/annotations/${split}.json data/annotations/oss_${split}.json
done

# ---- Proprietary track ----
# Option A: proprietary data only (recommended if large and high-quality)
python scripts/merge_annotations.py --seed 42 --val-fraction 0.2 \
    --sources proprietary

# Option B: merge proprietary with OSS sources to fill morphology gaps
python scripts/merge_annotations.py --seed 42 --val-fraction 0.2 \
    --sources proprietary gtimages satstreaks

for split in train val test; do
    cp data/annotations/${split}.json data/annotations/prop_${split}.json
done
```

Run the data evaluation checks from the previous section on both
`oss_train.json` and `prop_train.json` (substitute the filename in each
script) before continuing.

### Step 2 — Train one run per track

Swap the active split files before each run so the training script picks up the
correct data.  Use a track-specific `--work-dir` so the two runs are fully
isolated.

```bash
# ---- Open-source track ----
for split in train val test; do
    cp data/annotations/oss_${split}.json data/annotations/${split}.json
done

MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false ARGUS_NORM=autostretch \
python -m training.train_dino \
    --backbone dinov3_vitl \
    --work-dir weights/run_oss_dinov3_vitl

# ---- Proprietary track ----
for split in train val test; do
    cp data/annotations/prop_${split}.json data/annotations/${split}.json
done

MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false ARGUS_NORM=autostretch \
python -m training.train_dino \
    --backbone dinov3_vitl \
    --work-dir weights/run_prop_dinov3_vitl
```

### Step 3 — Evaluate each track independently

```bash
# ---- Open-source track ----
mkdir -p results/oss_dinov3_vitl
BEST_OSS="$(ls weights/run_oss_dinov3_vitl/*best*.pth | head -n 1)"
MODEL_WEIGHTS="$BEST_OSS" MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false \
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/oss_test.json \
    --output results/oss_dinov3_vitl/phase8_benchmark.json
cp eval/results/dino_predictions.json results/oss_dinov3_vitl/dino_predictions.json

# ---- Proprietary track ----
mkdir -p results/prop_dinov3_vitl
BEST_PROP="$(ls weights/run_prop_dinov3_vitl/*best*.pth | head -n 1)"
MODEL_WEIGHTS="$BEST_PROP" MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false \
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/prop_test.json \
    --output results/prop_dinov3_vitl/phase8_benchmark.json
cp eval/results/dino_predictions.json results/prop_dinov3_vitl/dino_predictions.json
```

Each result directory needs the same `training_summary.md` and `environment.txt`
fields as the primary Phase D run (see Required result files below), plus a
**Data track** field recording which sources were used and why.

### Weights and sharing

| Track | Work dir | Results dir | Share to OneDrive? | Commit results? |
|-------|----------|-------------|-------------------|-----------------|
| OSS | `weights/run_oss_dinov3_vitl/` | `results/oss_dinov3_vitl/` | Yes | Yes |
| Proprietary | `weights/run_prop_dinov3_vitl/` | `results/prop_dinov3_vitl/` | Only if licence permits | Only if licence permits |

All `.pth` files are gitignored regardless of track.  Upload OSS weights to the
shared OneDrive handoff folder under `weights/run_oss_dinov3_vitl/`.  Store
proprietary weights privately — do not upload to the shared folder unless your
data licence explicitly allows redistribution.

```bash
# Commit OSS results only (proprietary results conditional on licence)
git add results/oss_dinov3_vitl/
git commit -m "Add OSS-track DINOv3 ViT-L training results"
```

---

## Pre-Flight Checks

Run the full training pre-flight:

```bash
python scripts/prepare_cloud_training.py
```

Do not start full training until this exits with status 0. If the training
smoke test is too slow during setup, it can be skipped once:

```bash
SKIP_SMOKE_TRAIN=1 python scripts/prepare_cloud_training.py
```

Before the real run, execute the full pre-flight without `SKIP_SMOKE_TRAIN`.

## Train DINOv3 ViT-L (Phase D — primary run)

This is the Phase D training run. The DINOv3 backbone is **fully frozen** —
only the ChannelMapper neck and DINO-DETR head train. This is the primary
evaluation before considering any backbone fine-tuning.

All work for this run takes place in your fork. Verify the upstream remote is
set before proceeding:

```bash
git remote -v   # should show both origin (your fork) and upstream (robertmwolf/Argus)
```

### DINOv3 ViT-L weights

The ViT-L backbone weights (`dinov3_vitl16_lvd1689m.pth`, ~1.1 GB) are **not**
downloaded automatically by `cloud_setup.sh` because they require Meta portal
access.  Obtain them via one of the following methods, in order of preference:

**Option A — OneDrive handoff folder (fastest for teammates)**

The zip is staged in the shared ARGUS handoff folder:

```text
https://1drv.ms/f/c/f9b9ba14546c7993/IgBKfTqYuQuWTZcfBhvDWoARAUD1kC9YfTDE70F9rHKH-o8?e=w4EplD
```

Download `weights/dinov3_vitl16_lvd1689m.pth.zip` from the `weights/` subfolder,
then extract it into the repo:

```bash
# Linux / WSL2
mkdir -p weights
cd weights
unzip dinov3_vitl16_lvd1689m.pth.zip
cd ..
```

**Option B — SCP from the Mac**

```bash
scp mac:~/Argus/weights/dinov3_vitl16_lvd1689m.pth weights/
```

Replace `mac` with the actual hostname or IP of the development Mac.

**Option C — Meta DINOv3 portal**

Download the ViT-L/16 LVD-1689M checkpoint from Meta's DINOv3 distribution.
The file used by ARGUS is expected at:

```text
weights/dinov3_vitl16_lvd1689m.pth
```

Do not substitute a DINOv2 ViT-L/14 checkpoint for this run. The adapter and
config expect DINOv3 ViT-L/16 constructor arguments and checkpoint keys.

**Verify before proceeding:**

```bash
ls -lh weights/dinov3_vitl16_lvd1689m.pth
# Expected: ~1.1 GB
python - <<'PY'
import torch, pathlib
w = pathlib.Path("weights/dinov3_vitl16_lvd1689m.pth")
assert w.stat().st_size > 1_000_000_000, f"File too small: {w.stat().st_size}"
sd = torch.load(w, map_location="cpu")
keys = list(sd.keys()) if isinstance(sd, dict) else list(sd.get("model", sd).keys())
print(f"OK — {len(keys)} weight tensors loaded")
PY
```

### Environment variables

```bash
export MODEL_SIZE=dinov3_vitl
export USE_DEV_SUBSET=false
export ARGUS_NORM=autostretch
export DATABASE_URL=sqlite+aiosqlite:///./argus.db
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Pre-flight

```bash
MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false \
python scripts/prepare_cloud_training.py
```

### Start training

```bash
python -m training.train_dino \
    --backbone dinov3_vitl \
    --work-dir weights/run_5070ti_dinov3_vitl
```

Restart / timebox options:

```bash
# Resume interrupted run
python -m training.train_dino \
    --backbone dinov3_vitl \
    --work-dir weights/run_5070ti_dinov3_vitl \
    --resume

# New run from a checkpoint
python -m training.train_dino \
    --backbone dinov3_vitl \
    --work-dir weights/run_5070ti_dinov3_vitl_retry \
    --load-from weights/run_5070ti_dinov3_vitl/best_coco_bbox_mAP_epoch_50.pth \
    --max-epochs 10 --val-interval 2 --checkpoint-interval 2
```

Expected VRAM: ~12 GB at 512px batch=1.
If OOM, reduce image size by editing `_img_scale` in `streak_dinov3_vitl.py` to `(384, 384)`.

### Evaluate

```bash
mkdir -p results/5070ti_dinov3_vitl
BEST_CKPT="$(ls weights/run_5070ti_dinov3_vitl/*best*.pth | head -n 1)"

MODEL_WEIGHTS="$BEST_CKPT" \
MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false \
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/test.json \
    --output results/5070ti_dinov3_vitl/phase8_benchmark.json

cp eval/results/dino_predictions.json \
    results/5070ti_dinov3_vitl/dino_predictions.json
```

### Update Unified Confidence Score weights

After evaluation, open `inference/confidence.py` and update the `"dinov3_vitl"`
entry in `DETECTOR_PROFILES` with the measured precision and recall from
`results/5070ti_dinov3_vitl/phase8_benchmark.json`:

```python
"dinov3_vitl": DetectorProfile(
    name="DINOv3 ViT-L",
    precision=<measured>,          # from phase8_benchmark.json "precision"
    recall=<measured>,             # from phase8_benchmark.json "recall"
    confidence_ceiling=None,       # ML detector — confidence is well-calibrated
    notes="Phase D results/5070ti_dinov3_vitl/phase8_benchmark.json",
),
```

Only set `confidence_ceiling` if the detector is observed to emit misleadingly high
scores on false positives (i.e. its confidence magnitude does not correlate with
true-positive probability).  ML detectors trained with cross-entropy loss are
generally well-calibrated and should leave this as `None`.

Verify:
```bash
python -m inference.confidence          # scores should reflect new weights
python -m pytest tests/test_confidence.py -v   # all tests must pass
```

Commit this change alongside the result files.

### Required result files

```text
results/5070ti_dinov3_vitl/phase8_benchmark.json
results/5070ti_dinov3_vitl/confusion_matrix.png
results/5070ti_dinov3_vitl/training_summary.md
results/5070ti_dinov3_vitl/environment.txt
results/5070ti_dinov3_vitl/dino_predictions.json
```

`training_summary.md` must include the same fields as the Swin-L summary
(see below), plus:
- Backbone: DINOv3 ViT-L/16 LVD-1689M (frozen)
- Whether Phase 8 targets met: ≥94% precision, ≥97% recall
- Comparison note vs Swin-T baseline (mAP@0.5=0.190 on test.json, from results/phase_e/phase_e_comparison_test.json)
- Confirmation that `DETECTOR_PROFILES["dinov3_vitl"]` in `inference/confidence.py` was updated

Generate environment metadata:

```bash
{
    echo "Date UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Git commit: $(git rev-parse HEAD)"
    echo "Backbone: DINOv3 ViT-L/16 LVD-1689M (frozen)"
    echo "GPU:"
    nvidia-smi
    echo ""
    echo "Python:"
    python --version
    echo ""
    echo "PyTorch:"
    python - <<'PY'
import torch
print(torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("cuda:", torch.version.cuda)
PY
    echo ""
    echo "Installed packages:"
    pip freeze
} > results/5070ti_dinov3_vitl/environment.txt
```

### Stage weights to OneDrive

Checkpoints are gitignored — upload them to the shared OneDrive handoff folder:

```text
https://1drv.ms/f/c/f9b9ba14546c7993/IgBKfTqYuQuWTZcfBhvDWoARAUD1kC9YfTDE70F9rHKH-o8?e=w4EplD
```

```bash
sha256sum weights/run_5070ti_dinov3_vitl/*.pth \
    > results/5070ti_dinov3_vitl/weights_sha256.txt
```

Upload `weights/run_5070ti_dinov3_vitl/*.pth` to OneDrive under
`weights/run_5070ti_dinov3_vitl/`, then create
`results/5070ti_dinov3_vitl/weights_manifest.json`:

```json
{
  "best_checkpoint": "best_coco_bbox_mAP_epoch_N.pth",
  "storage": "OneDrive shared training handoff folder",
  "onedrive_subfolder": "weights/run_5070ti_dinov3_vitl/",
  "sha256_file": "results/5070ti_dinov3_vitl/weights_sha256.txt"
}
```

### Check results back into GitHub

```bash
git add results/5070ti_dinov3_vitl/
git commit -m "Add RTX 5070 Ti DINOv3 ViT-L training results"
git push origin main
```

Open a pull request from your fork's `main` targeting `robertmwolf/Argus:main`
titled:

```text
Add RTX 5070 Ti DINOv3 ViT-L training results
```

---

## Route 2 — Cloud GPU Rental (RTX 4090)

This is the self-contained cloud path for Phase D. It does not require access
to the colleague's workstation. Results land in a separate directory so both
routes can coexist if run simultaneously.

**Provider**: Vast.ai or RunPod — RTX 4090 24 GB
**On-demand rate**: ~$0.44/hr | **Spot rate**: ~$0.30/hr (use `--resume` for recovery)
**Estimated total cost**: $7–18 on-demand / $5–13 spot
**Advantage over Route 1**: Tries 800 px image size (more patch coverage for thin
streaks); early stopping keeps cost low if convergence is fast.

### Step 1 — Pre-flight on Mac (free)

```bash
MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false \
python scripts/prepare_cloud_training.py
```

Resolve all failures before provisioning an instance.

### Step 2 — Provision the instance

1. On Vast.ai or RunPod, filter for **RTX 4090** nodes with ≥24 GB VRAM and
   PyTorch ≥ 2.2 pre-installed (no Blackwell requirement — 4090 is Ada Lovelace).
2. Prefer on-demand for a first run; spot is fine with `--resume` recovery.
3. SSH in, then clone the repo and run setup:

```bash
git clone https://github.com/robertmwolf/Argus.git && cd Argus
chmod +x scripts/cloud_setup.sh && ./scripts/cloud_setup.sh
```

4. Copy DINOv3 ViT-L weights from Mac (~1.1 GB):

```bash
scp mac:~/Argus/weights/dinov3_vitl16_lvd1689m.pth weights/
```

### Step 3 — Update the cost guardrail rate

In `training/train_dino.py`, update the hardcoded `$1.29/hr` to your actual
provider rate (e.g. `$0.44/hr`) so the epoch-1 cost estimate is accurate.

### Step 4 — Set image resolution to 800 px

In `models/dino/streak_dinov3_vitl.py`:

```python
_img_scale = (800, 800)   # change from (512, 512)
```

At 800 px, frozen ViT-L uses ~18 GB VRAM — within the 4090's 24 GB with
gradient checkpointing enabled. If the first epoch OOMs, revert to `(512, 512)`
and resume.

### Step 5 — Set environment variables and train

```bash
export MODEL_SIZE=dinov3_vitl
export USE_DEV_SUBSET=false
export ARGUS_NORM=autostretch
export DATABASE_URL=sqlite+aiosqlite:///./argus.db
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m training.train_dino \
    --backbone dinov3_vitl \
    --work-dir weights/run_4090_dinov3_vitl
```

To resume after a spot preemption:

```bash
python -m training.train_dino \
    --backbone dinov3_vitl \
    --work-dir weights/run_4090_dinov3_vitl \
    --resume
```

**Stop early** if val mAP@0.5 has not improved for 5 consecutive epochs and is
already ≥ 0.74. Terminate, save the best checkpoint, and proceed to evaluation.

### Step 6 — Evaluate

```bash
mkdir -p results/4090_dinov3_vitl
BEST_CKPT="$(ls weights/run_4090_dinov3_vitl/*best*.pth | head -n 1)"

MODEL_WEIGHTS="$BEST_CKPT" \
MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false \
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/test.json \
    --output results/4090_dinov3_vitl/phase8_benchmark.json

cp eval/results/dino_predictions.json results/4090_dinov3_vitl/dino_predictions.json
```

Generate environment metadata:

```bash
{
    echo "Date UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Git commit: $(git rev-parse HEAD)"
    echo "Route: Cloud GPU (RTX 4090 on Vast.ai/RunPod)"
    echo "Backbone: DINOv3 ViT-L/16 LVD-1689M (frozen)"
    nvidia-smi
    python --version
    python -c "import torch; print(torch.__version__)"
    pip freeze
} > results/4090_dinov3_vitl/environment.txt
```

### Step 7 — Rsync weights back to Mac and upload to OneDrive

```bash
# On Mac
scripts/fetch_weights.sh <user@instance-ip>
sha256sum weights/run_4090_dinov3_vitl/*.pth \
    > results/4090_dinov3_vitl/weights_sha256.txt
```

Upload `weights/run_4090_dinov3_vitl/*.pth` to the shared OneDrive folder under
`weights/run_4090_dinov3_vitl/`.

### Step 8 — Commit results

```bash
git add results/4090_dinov3_vitl/
git commit -m "Add RTX 4090 cloud DINOv3 ViT-L training results"
git push origin main
```

### Cost reference

| Scenario | Epochs | Hours (800 px) | On-demand | Spot |
|----------|--------|----------------|-----------|------|
| Early stop ~15 epochs | 15 | ~11 hr | ~$5 | ~$3 |
| Early stop ~25 epochs | 25 | ~19 hr | ~$8 | ~$6 |
| Full 50 epochs | 50 | ~38 hr | ~$17 | ~$11 |

### Aspirational: $50–150 — Phase F Partial Unfreeze

If Phase D (either route) falls short of targets (precision <94% or recall
<97%), Phase F unfreezes the last 4 ViT-L transformer blocks (`lr_mult=0.01`,
10–15 epochs from the Phase D best checkpoint). VRAM at 800 px with unfrozen
blocks: ~18–22 GB — requires an A100 80 GB instance (~$1.79/hr on RunPod).
Estimated Phase F cost: $9–14. Do not start Phase F without Phase D eval results
in hand.

---

## Phase E — DINOv3 ViT-L vs Swin-T/L Comparison

Run this after the DINOv3 ViT-L training above is complete and the best
checkpoint is saved in `weights/run_5070ti_dinov3_vitl/` (Route 1) or
`weights/run_4090_dinov3_vitl/` (Route 2).

### What Phase E does

Evaluates DINOv3 ViT-L (Phase D) head-to-head against the Swin-T baseline
on the held-out `test` split using MMDetection CocoMetric.  Outputs a
Markdown comparison table and a combined JSON.

### Run Phase E comparison

```bash
mkdir -p results/phase_e

# Evaluate DINOv3 ViT-L (Phase D) only — Swin-T baseline is already present
# from the Mac Phase C run; to regenerate it here, drop --model:
python scripts/phase_e_compare.py \
    --model dinov3_vitl \
    --split test \
    --dinov3-checkpoint "$(ls weights/run_5070ti_dinov3_vitl/best_coco_bbox_mAP_epoch_*.pth | tail -1)" \
    --output-dir results/phase_e
```

For a full comparison (re-evaluates both models):

```bash
python scripts/phase_e_compare.py \
    --split test \
    --output-dir results/phase_e
```

### Required result files

```text
results/phase_e/phase_e_comparison_test.json
results/phase_e/swin_t_test_metrics.json
results/phase_e/dinov3_vitl_test_metrics.json
```

### Interpretation gates

| Metric | Gate |
|--------|------|
| DINOv3 mAP@0.5 > Swin-T mAP@0.5 | Phase D is better |
| DINOv3 mAP@0.5 within 5 pp of Swin-T | Acceptable — frozen backbone competitive |
| DINOv3 mAP@0.5 > 5 pp below Swin-T | Consider Phase F (partial unfreeze, A100) |

Phase 8 hard targets: ≥94% precision, ≥97% recall (measured on `test.json`).

---

## Train Swin-L

Start full training:

```bash
MODEL_SIZE=large USE_DEV_SUBSET=false \
python -m training.train_dino \
    --work-dir weights/run_5070ti_swin_l
```

Useful restart/timebox options:

```bash
python -m training.train_dino \
    --work-dir weights/run_5070ti_swin_l \
    --resume

python -m training.train_dino \
    --work-dir weights/run_5070ti_swin_l_retry \
    --load-from weights/run_5070ti_swin_l/best_coco_bbox_mAP_epoch_50.pth \
    --max-epochs 10 \
    --val-interval 2 \
    --checkpoint-interval 2
```

Use `--resume` for an interrupted same-work-dir run. Use `--load-from` for a
new run initialized from a selected checkpoint.

The RTX 5070 Ti has 16 GB VRAM. The Swin-L config is tuned for this with:

```text
batch_size=1
AMP enabled
gradient checkpointing enabled
gradient accumulation enabled
```

If CUDA out-of-memory occurs, capture the full error in
`results/5070ti_swin_l/training_summary.md` before changing the configuration.

## Evaluate

After training, identify the best checkpoint:

```bash
ls -lh weights/run_5070ti_swin_l/*best*.pth
```

Run benchmark evaluation on the held-out test split:

```bash
mkdir -p results/5070ti_swin_l
BEST_CKPT="$(ls weights/run_5070ti_swin_l/*best*.pth | head -n 1)"

MODEL_WEIGHTS="$BEST_CKPT" \
MODEL_SIZE=large USE_DEV_SUBSET=false \
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/test.json \
    --output results/5070ti_swin_l/phase8_benchmark.json
```

If `BEST_CKPT` is empty, replace it with the exact checkpoint filename from
`weights/run_5070ti_swin_l/`.

The evaluation should produce:

```text
results/5070ti_swin_l/phase8_benchmark.json
results/5070ti_swin_l/confusion_matrix.png
eval/results/dino_predictions.json
```

Copy predictions into the results folder for check-in:

```bash
cp eval/results/dino_predictions.json results/5070ti_swin_l/dino_predictions.json
```

## Update Unified Confidence Score Weights

After evaluation, open `inference/confidence.py` and update the `"large"` entry in
`DETECTOR_PROFILES` with the measured precision and recall from
`results/5070ti_swin_l/phase8_benchmark.json`:

```python
"large": DetectorProfile(
    name="DINO Swin-L",
    precision=<measured>,          # from phase8_benchmark.json "precision"
    recall=<measured>,             # from phase8_benchmark.json "recall"
    confidence_ceiling=None,       # ML detector — confidence is well-calibrated
    notes="Swin-L results/5070ti_swin_l/phase8_benchmark.json",
),
```

Only set `confidence_ceiling` if the detector emits unreliably high scores on false
positives.  ML detectors are generally well-calibrated and should leave this as `None`.

Verify:
```bash
python -m inference.confidence          # scores should reflect new weights
python -m pytest tests/test_confidence.py -v   # all tests must pass
```

Commit this change alongside the result files.

## Required Result Files

Create and commit these files:

```text
results/5070ti_swin_l/phase8_benchmark.json
results/5070ti_swin_l/confusion_matrix.png
results/5070ti_swin_l/training_summary.md
results/5070ti_swin_l/environment.txt
results/5070ti_swin_l/dino_predictions.json
```

Generate environment metadata:

```bash
{
    echo "Date UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Git commit: $(git rev-parse HEAD)"
    echo "GPU:"
    nvidia-smi
    echo ""
    echo "Python:"
    python --version
    echo ""
    echo "PyTorch:"
    python - <<'PY'
import torch
print(torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("cuda:", torch.version.cuda)
PY
    echo ""
    echo "Installed packages:"
    pip freeze
} > results/5070ti_swin_l/environment.txt
```

`training_summary.md` must include:

```text
GPU model and VRAM
CPU model/core count
PyTorch version
CUDA version
Driver version
Training command
Work dir
Best checkpoint filename
Best epoch
Total training time
Final train loss
Best validation coco/bbox_mAP
mAP@0.5
mAP@0.75
Precision
Recall
F1
Mean angle error in degrees
Per-band short/medium/long precision, recall, and F1
Confusion matrix TP, FP, FN, TN
Whether precision >= 0.94
Whether recall >= 0.97
DETECTOR_PROFILES updated in inference/confidence.py: yes/no
Any errors, retries, or config changes
```

## Weights

Model checkpoints are gitignored and must **never** be committed to this
repository.  All trained weights are staged to the shared OneDrive handoff
folder:

```text
https://1drv.ms/f/c/f9b9ba14546c7993/IgBKfTqYuQuWTZcfBhvDWoARAUD1kC9YfTDE70F9rHKH-o8?e=w4EplD
```

After training completes:

1. Generate checksums:

```bash
mkdir -p results/5070ti_swin_l
sha256sum weights/run_5070ti_swin_l/*.pth \
    > results/5070ti_swin_l/weights_sha256.txt
```

2. Upload the checkpoint files to the OneDrive folder under a subfolder named
   for this run (e.g. `weights/run_5070ti_swin_l/`).

3. Create `results/5070ti_swin_l/weights_manifest.json`:

```json
{
  "best_checkpoint": "argus_swin_l_5070ti_best.pth",
  "final_checkpoint": "argus_swin_l_5070ti_epoch_50.pth",
  "storage": "OneDrive shared training handoff folder",
  "onedrive_subfolder": "weights/run_5070ti_swin_l/",
  "sha256_file": "results/5070ti_swin_l/weights_sha256.txt"
}
```

4. Commit only the manifest and checksum file — not the `.pth` files:

```bash
git add results/5070ti_swin_l/weights_manifest.json
git add results/5070ti_swin_l/weights_sha256.txt
```

## Check Results Back Into GitHub

Stage results (weights stay on OneDrive — never commit `.pth` files):

```bash
git add results/5070ti_swin_l/
git commit -m "Add RTX 5070 Ti Swin-L training results"
git push origin main
```

Open a pull request from your fork's `main` targeting `robertmwolf/Argus:main`
titled:

```text
Add RTX 5070 Ti Swin-L training results
```

The pull request body should summarize the training hardware, best checkpoint,
precision, recall, mAP@0.5, mAP@0.75, F1, angle error, and whether the Phase 8
targets were met.

---

## File Layout (after full setup)

```
data/
├── GTImages/              ← 759 FITS + .strk + .ini + .wcs (from zip)
├── satstreaks/Data/
│   ├── Images/            ← 3073 JPG (downloaded from SharePoint)
│   ├── Masks/             ← 3073 PNG masks
│   └── labels.json        ← committed to repo
└── annotations/
    ├── dev_subset.json    ← committed (50-image Mac dev subset)
    ├── gtimages.json      ← generated by convert_gtimages.py
    ├── gtimages_negatives.json
    ├── train.json         ← generated by merge_annotations.py
    ├── val.json
    └── test.json

weights/
├── co_dino_swin_l_coco.pth   ← downloaded by cloud_setup.sh
└── run_001/                  ← training output
    ├── best_coco_bbox_mAP_epoch_NN.pth
    ├── latest.pth
    └── *.log

results/
├── final_eval.json
└── confusion_matrix.png
```

---

## Troubleshooting

### Windows — mmcv CUDA ops (`No module named 'mmcv._ext'`)

**Symptom:** `mmcv` imports version `2.1.0` but `from mmcv.ops import ...`
raises `No module named 'mmcv._ext'`.  Training fails at the first deformable
convolution op.  OpenMIM (`mim install mmcv==2.1.0`) falls back to source
compilation and fails inside CUDA/PyTorch C++ headers.

**Root cause:** As of early 2026 there is no pre-built Windows wheel for:
```
Windows + Python 3.11 + PyTorch 2.6+/cu128 + CUDA 12.8
```
OpenMMLab publishes Linux and macOS wheels only.  Building mmcv from source on
Windows requires MSVC 2022 + a perfectly matching CUDA SDK version against the
PyTorch headers — a notoriously fragile combination that currently does not work
for the Blackwell/cu128 stack.

**Resolution — use WSL2 (strongly recommended):**

```powershell
# PowerShell (admin)
wsl --install -d Ubuntu-22.04
# Reboot when prompted, then open the Ubuntu terminal
```

Inside WSL2 Ubuntu:

```bash
# Verify GPU is visible
nvidia-smi   # must show your GPU; if not, install WSL CUDA driver first:
             # https://developer.nvidia.com/cuda/wsl

# Clone repo and run the Linux setup script
git clone https://github.com/<your-username>/Argus.git
cd Argus
git remote add upstream https://github.com/robertmwolf/Argus.git
chmod +x scripts/cloud_setup.sh
./scripts/cloud_setup.sh
```

All Linux mmcv/cu128 wheels install without compilation inside WSL2.

**Alternative — Docker:**

```bash
docker compose -f docker-compose.cloud.yml up --build worker
```

The CUDA base image handles Linux dependency installation, including mmcv CUDA
ops, inside the container.

**Do not attempt native Windows:** `mmcv-lite` (without `_ext`) will not train
Co-DINO.  Spending time on VS 2022 + CUDA source builds for this stack is not
productive — the WSL2 path is the correct one.

### CUDA not found after installing PyTorch

```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False:
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### mmdet import error

```bash
pip install mmdet mmengine mmcv
```

### `EnvironmentError: MODEL_SIZE=large requires CUDA`

The Swin-L model refuses to load on CPU. Ensure CUDA is available:
```bash
python -c "import torch; assert torch.cuda.is_available()"
```

### OOM on first batch

Try `IMAGE_SIZE=640`:
```bash
export IMAGE_SIZE=640
python -m training.train_dino --work-dir weights/run_001
```

### Training resumes from wrong epoch

Check `weights/run_001/` for checkpoint files. Pass the correct checkpoint
to `--resume`:
```bash
ls weights/run_001/*.pth
python -m training.train_dino --work-dir weights/run_001 \
    --resume weights/run_001/epoch_20.pth
```
