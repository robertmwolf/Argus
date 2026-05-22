# ARGUS Cloud Training Preparation

Use this checklist before renting GPU time for Phase D DINOv3 ViT-L training.
The goal is to make the cloud run reproducible, resumable, and boring: code,
data, weights, environment, logs, checkpoints, and final metrics should all have
a clear owner and storage location before the instance starts.

## Target Rental

For the primary Phase D cloud route, rent:

| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA RTX 4090, 24 GB VRAM |
| OS | Ubuntu 22.04 or equivalent Linux CUDA image |
| RAM | 32 GB minimum, 64 GB preferred |
| CPU | 8+ vCPU |
| Disk | 200 GB minimum, 250-300 GB preferred |
| Storage | Persistent volume or a tested off-machine sync path |

Prefer an on-demand instance for the first real run. Spot/preemptible instances
are acceptable only if checkpoint sync and `--resume` have already been tested.

## Standard Practice

Every serious training run should record:

- Exact code version: Git branch, commit, and whether the worktree was dirty.
- Dataset version: source, split files, image/mask counts, and checksums.
- Backbone weight version: file path, size, and SHA-256 checksum.
- Environment: GPU details, driver/CUDA, Python, PyTorch, MMDetection, and
  `pip freeze`.
- Run command: all environment variables, CLI arguments, work directory, and
  image resolution.
- Recovery path: resume command and checkpoint sync destination.
- Evaluation: held-out annotation file, metrics output path, prediction dump,
  and final checkpoint manifest.

If any of these cannot be recorded, treat the run as exploratory rather than a
paper or decision-grade result.

## Local Preparation Before Renting

Run these steps on the development machine before provisioning cloud hardware.

### 1. Freeze the Code State

Use a committed branch whenever possible:

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

If the worktree is dirty and cannot be committed yet, create a source archive
from the exact working tree and record that archive name in the manifest. Do
not rely on memory to reconstruct local edits after training.

### 2. Verify Required Inputs Exist

For the current Phase D route, these are the minimum files/directories:

```text
data/annotations/train.json
data/annotations/val.json
data/annotations/test.json
data/satstreaks/Data/Images/
data/satstreaks/Data/Masks/
data/satstreaks/Data/labels.json
weights/dinov3_vitl16_lvd1689m.pth
```

Check counts and sizes:

```bash
ls -lh data/annotations/train.json data/annotations/val.json data/annotations/test.json
ls -lh weights/dinov3_vitl16_lvd1689m.pth
find data/satstreaks/Data/Images -maxdepth 1 -type f | wc -l
find data/satstreaks/Data/Masks -maxdepth 1 -type f | wc -l
```

Expected local reference counts from the current ARGUS dataset are about 3,074
SatStreaks images and 3,073 masks. Record any discrepancy before training.

### 3. Record Checksums

Create a local manifest directory and checksum the high-value files:

```bash
mkdir -p results/cloud_training_prep

sha256sum \
  data/annotations/train.json \
  data/annotations/val.json \
  data/annotations/test.json \
  weights/dinov3_vitl16_lvd1689m.pth \
  > results/cloud_training_prep/input_sha256.txt
```

On macOS, use `shasum -a 256` if `sha256sum` is unavailable:

```bash
shasum -a 256 \
  data/annotations/train.json \
  data/annotations/val.json \
  data/annotations/test.json \
  weights/dinov3_vitl16_lvd1689m.pth \
  > results/cloud_training_prep/input_sha256.txt
```

### 4. Create the Run Manifest

Copy `docs/templates/cloud_training_manifest.md` to the run results directory:

```bash
mkdir -p results/4090_dinov3_vitl
cp docs/templates/cloud_training_manifest.md \
  results/4090_dinov3_vitl/training_summary.md
```

Fill in everything known before provisioning: code state, dataset decision,
input counts, intended provider, target image size, expected hourly rate, and
the planned checkpoint sync destination.

### 5. Plan the Transfer

Use `rsync` when possible because it resumes partial transfers:

```bash
rsync -avh --progress \
  data/annotations/ \
  <user@host>:~/Argus/data/annotations/

rsync -avh --progress \
  data/satstreaks/Data/ \
  <user@host>:~/Argus/data/satstreaks/Data/

rsync -avh --progress \
  weights/dinov3_vitl16_lvd1689m.pth \
  <user@host>:~/Argus/weights/
```

If provider storage is rebuilt often, upload the same inputs to durable storage
first, then download from the cloud instance. Avoid spending paid GPU time on
manual browser downloads.

## Cloud Bring-Up

After SSHing into the rented instance:

```bash
git clone https://github.com/robertmwolf/Argus.git
cd Argus
chmod +x scripts/cloud_setup.sh
./scripts/cloud_setup.sh
```

Then restore the data and weights, verify checksums, and run preflight:

```bash
export MODEL_SIZE=dinov3_vitl
export USE_DEV_SUBSET=false
export ARGUS_NORM=autostretch
export DATABASE_URL=sqlite+aiosqlite:///./argus.db
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/prepare_cloud_training.py
```

The setup script intentionally prints warnings for missing data/weights and
continues. The explicit preflight command above is the go/no-go gate; it must
exit 0 before the long run starts.

## Smoke Test

Run the training smoke test on the cloud GPU before the full job:

```bash
python -m training.train_dino \
  --backbone dinov3_vitl \
  --smoke-test
```

This catches broken CUDA ops, bad imports, missing weights, and annotation path
problems before committing to a 15-38 hour run.

## Full Run

For an RTX 4090 route, set `_img_scale = (800, 800)` in
`models/dino/streak_dinov3_vitl.py` before the real run. If the first epoch OOMs,
revert to `(512, 512)` and resume.

```bash
python -m training.train_dino \
  --backbone dinov3_vitl \
  --work-dir weights/run_4090_dinov3_vitl
```

Resume command:

```bash
python -m training.train_dino \
  --backbone dinov3_vitl \
  --work-dir weights/run_4090_dinov3_vitl \
  --resume
```

## Checkpoint Sync

During the run, periodically copy checkpoints to durable storage:

```bash
rsync -avh --progress \
  weights/run_4090_dinov3_vitl/ \
  <backup-target>/weights/run_4090_dinov3_vitl/
```

After training, record checksums:

```bash
mkdir -p results/4090_dinov3_vitl
sha256sum weights/run_4090_dinov3_vitl/*.pth \
  > results/4090_dinov3_vitl/weights_sha256.txt
```

## Final Required Artifacts

The run is not complete until these files exist:

```text
results/4090_dinov3_vitl/phase8_benchmark.json
results/4090_dinov3_vitl/confusion_matrix.png
results/4090_dinov3_vitl/training_summary.md
results/4090_dinov3_vitl/environment.txt
results/4090_dinov3_vitl/dino_predictions.json
results/4090_dinov3_vitl/weights_sha256.txt
results/4090_dinov3_vitl/weights_manifest.json
```

Upload `.pth` checkpoints to the shared handoff storage; do not commit them to
Git. Commit only metrics, summaries, environment metadata, checksums, and the
weights manifest.
