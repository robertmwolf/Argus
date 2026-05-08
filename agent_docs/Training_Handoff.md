# ARGUS Training Handoff

This document is the handoff checklist for training ARGUS on a more powerful
Windows or Linux workstation with an NVIDIA RTX 5070 Ti 16 GB GPU and a
24-core i9 CPU.

## Short Prompt for the Colleague's Codex

Give the colleague's Codex this instruction:

```text
Check out repo at https://github.com/robertmwolf/Argus.git
Read instructions at agent_docs/Training_Handoff.md
Follow that document exactly to download data, run Swin-L training, evaluate
results, and check the outputs back into GitHub.
```

## Current Handoff Scope

This handoff is for workstation training on an NVIDIA RTX 5070 Ti 16 GB GPU.
The historical Lambda A100 path remains valid, but the data download flow and
expected output branch below are tailored to the 5070 Ti run.

## Data Sources

The big datasets should be downloaded directly on the training workstation.
Do not copy the full local data tree through the shared handoff folder.

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

ARGUS lightweight handoff folder:

```text
https://1drv.ms/f/c/f9b9ba14546c7993/IgBKfTqYuQuWTZcfBhvDWoARAUD1kC9YfTDE70F9rHKH-o8?e=w4EplD
```

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
|-- optional_weights/
|   `-- co_dino_swin_l_coco.pth
`-- MANIFEST.txt
```

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

Clone the project:

```bash
git clone https://github.com/robertmwolf/Argus.git
cd Argus
```

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
Any errors, retries, or config changes
```

## Weights

Model checkpoints are large and gitignored. Prefer Git LFS if it is enabled for
the repository:

```bash
git lfs install
git lfs track "weights/final/*.pth"
mkdir -p weights/final
cp weights/run_5070ti_swin_l/best_coco_bbox_mAP*.pth \
    weights/final/argus_swin_l_5070ti_best.pth
cp weights/run_5070ti_swin_l/epoch_50.pth \
    weights/final/argus_swin_l_5070ti_epoch_50.pth
git add .gitattributes weights/final/*.pth
```

If Git LFS is not available, upload the final weights to the shared OneDrive
folder and commit only a manifest:

```bash
mkdir -p results/5070ti_swin_l
sha256sum weights/run_5070ti_swin_l/*.pth \
    > results/5070ti_swin_l/weights_sha256.txt
```

Create `results/5070ti_swin_l/weights_manifest.json` with:

```json
{
  "best_checkpoint": "argus_swin_l_5070ti_best.pth",
  "final_checkpoint": "argus_swin_l_5070ti_epoch_50.pth",
  "storage": "OneDrive shared training handoff folder",
  "sha256_file": "results/5070ti_swin_l/weights_sha256.txt"
}
```

Commit the manifest and checksum file.

## Check Results Back Into GitHub

Create a results branch:

```bash
git checkout -b codex/5070ti-swin-l-training-results
```

Stage results:

```bash
git add results/5070ti_swin_l/
git add .gitattributes weights/final/*.pth
```

If Git LFS is not being used, skip the weights add and stage only the manifest:

```bash
git add results/5070ti_swin_l/weights_manifest.json
git add results/5070ti_swin_l/weights_sha256.txt
```

Commit and push:

```bash
git commit -m "Add RTX 5070 Ti Swin-L training results"
git push -u origin codex/5070ti-swin-l-training-results
```

Open a pull request titled:

```text
Add RTX 5070 Ti Swin-L training results
```

The pull request body should summarize the training hardware, best checkpoint,
precision, recall, mAP@0.5, mAP@0.75, F1, angle error, and whether the Phase 8
targets were met.
