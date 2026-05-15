# ARGUS Training Handoff

This document is the handoff checklist for training ARGUS on a more powerful
Windows or Linux workstation with an NVIDIA RTX 5070 Ti 16 GB GPU and a
24-core i9 CPU.

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

## Train DINOv3 ViT-L (Phase D — primary run)

This is the Phase D training run for the `feature/dinov3-backbone` branch.
The DINOv3 backbone is **fully frozen** — only the ChannelMapper neck and
DINO-DETR head train. This is the primary evaluation before considering any
backbone fine-tuning.

### Branch

Check out the DINOv3 branch before training:

```bash
git fetch origin
git checkout feature/dinov3-backbone
```

### DINOv3 ViT-L weights

The ViT-L backbone weights are not downloaded by `cloud_setup.sh` automatically
because they require Meta portal access. Copy from the Mac:

```bash
scp mac:~/Argus/weights/dinov3_vitl16_lvd1689m.pth weights/
```

Verify the file is ~1.1 GB before proceeding.

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
- Comparison note vs Swin-T dev baseline (mAP@0.5=0.657)

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

### Check results back into GitHub

```bash
git checkout -b codex/5070ti-dinov3-vitl-training-results
git add results/5070ti_dinov3_vitl/
git commit -m "Add RTX 5070 Ti DINOv3 ViT-L training results"
git push -u origin codex/5070ti-dinov3-vitl-training-results
```

Open a pull request against `feature/dinov3-backbone` (not `main`) titled:

```text
Add RTX 5070 Ti DINOv3 ViT-L training results
```

---

## Phase E — DINOv3 ViT-L vs Swin-T/L Comparison

Run this after the DINOv3 ViT-L training above is complete and the best
checkpoint is saved in `weights/run_5070ti_dinov3_vitl/`.

### What Phase E does

Evaluates DINOv3 ViT-L (Phase D) head-to-head against the Swin-T baseline
on the held-out `val` split using MMDetection CocoMetric.  Outputs a
Markdown comparison table and a combined JSON.

### Run Phase E comparison

```bash
mkdir -p results/phase_e

# Evaluate DINOv3 ViT-L (Phase D) only — Swin-T baseline is already present
# from the Mac Phase C run; to regenerate it here, drop --model:
python scripts/phase_e_compare.py \
    --model dinov3_vitb \
    --split val \
    --dinov3-checkpoint "$(ls weights/run_5070ti_dinov3_vitl/best_coco_bbox_mAP_epoch_*.pth | tail -1)" \
    --output-dir results/phase_e
```

For a full comparison (re-evaluates both models):

```bash
python scripts/phase_e_compare.py \
    --split val \
    --output-dir results/phase_e
```

### Required result files

```text
results/phase_e/phase_e_comparison_val.json
results/phase_e/swin_t_val_metrics.json
results/phase_e/dinov3_vitb_val_metrics.json   (or dinov3_vitl if adapted)
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
