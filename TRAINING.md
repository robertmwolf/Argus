# ARGUS — GPU Training Handover Guide

This guide is written for someone coming in cold to train ARGUS on a GPU machine.
Follow each step in order; the pre-flight checklist at the end will catch anything
you missed before GPU time starts.

> **Active training track:** DINOv3 ViT-L (Phase D) is the current priority run.
> See [`agent_docs/Training_Handoff.md`](agent_docs/Training_Handoff.md) for the
> detailed workstation handoff. Phase C² (ViT-B, full dataset, 4 epochs) is
> complete with mAP@0.5=0.74; Phase D targets 50 epochs with ViT-L.
> This file covers the general Swin-L training path, which remains valid.

---

## Hardware requirements

| Component | Minimum | Tested on |
|-----------|---------|-----------|
| GPU | 12 GB VRAM (CUDA) | NVIDIA RTX 5070 Ti 16 GB |
| CPU | 8 cores | Intel i9-24 core |
| RAM | 32 GB | — |
| Disk | 200 GB free | — |
| OS | Ubuntu 22.04 LTS | — |
| CUDA | 12.6+ | 12.8 recommended for Blackwell |
| Driver | 560+ | — |

The config is tuned for **16 GB VRAM** (batch_size=1, gradient accumulation=2,
mixed precision, gradient checkpointing).  If you have more VRAM (e.g. A100 40 GB)
you can raise `batch_size` to 2 in `models/dino/streak_codino_swin_l.py` and
remove `accumulative_counts`.

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/<your-org>/argus.git
cd argus
```

---

## Step 2 — Run the environment setup script

This installs Miniconda, creates the `satid` conda environment, installs all
dependencies, downloads Swin-L weights, and converts the GTImages annotations.

```bash
chmod +x scripts/cloud_setup.sh
./scripts/cloud_setup.sh
```

After it finishes, start a new shell (or `source ~/.bashrc`) so the environment
variables take effect.

**What the script sets permanently in `~/.bashrc`:**

```bash
MODEL_SIZE=large
USE_DEV_SUBSET=false
DATABASE_URL=sqlite+aiosqlite:///./argus.db
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Step 3 — Download the training datasets

You need two datasets.  Download both before merging annotations.

### 3a. GTImages (provided by the project — 37 GB)

This dataset was collected at a ground station in Ontario, Canada.
You will receive a link to `GTImages.zip` from the project owner.

```bash
# Download GTImages.zip from the shared link you were given, then:
cd /path/to/argus
unzip GTImages.zip -d data/
# Result: data/GTImages/*.fits  (759 images) + *.strk + *.ini + *.wcs
```

Verify the download:
```bash
ls data/GTImages/*.fits | wc -l   # should print 759
ls data/GTImages/*.strk | wc -l   # should print 68
```

Optionally verify integrity (SHA-256 provided separately by the project owner):
```bash
sha256sum GTImages.zip
```

### 3b. SatStreaks (public — ~2 GB)

Download from the link in `data/satstreaks/README.md`:

```
https://smuhalifax-my.sharepoint.com/:f:/g/personal/susrita_chatterjee_smu_ca/...
```

Extract so the directory structure is:
```
data/satstreaks/Data/Images/*.jpg    (3,073 images)
data/satstreaks/Data/Masks/*.png     (segmentation masks)
data/satstreaks/Data/labels.json     (already present in the repo)
```

---

## Step 4 — Convert GTImages and merge annotations

### Convert GTImages → COCO JSON

```bash
conda activate satid
python scripts/convert_gtimages.py \
    --strk-dir data/GTImages \
    --output data/annotations/gtimages.json \
    --negatives-output data/annotations/gtimages_negatives.json
```

Expected output:
```
INFO  Wrote data/annotations/gtimages.json  (593 images, 593 annotations)
INFO  Wrote data/annotations/gtimages_negatives.json  (93 images, 0 annotations)
```

### Merge into train/val/test splits

```bash
python scripts/merge_annotations.py
```

Expected output:
```
INFO  SatStreaks: train=2488, val=277, test=308 images
INFO  GTImages: train=549, val=137 images (80/20 random split, seed=42)
INFO  Wrote data/annotations/train.json  (3037 images, ...)
INFO  Wrote data/annotations/val.json    (414 images, ...)
INFO  Wrote data/annotations/test.json   (308 images, ...)
INFO  Done.  Total: train=3037, val=414, test=308 images
```

---

## Step 5 — Run the pre-flight checklist

This validates every precondition before GPU time starts.
It will catch missing data, misconfigured env vars, and broken configs.

```bash
MODEL_SIZE=large USE_DEV_SUBSET=false \
python scripts/prepare_cloud_training.py
```

All checks must show ✓.  Fix any failures before proceeding.

The `Training smoke test` runs 2 epochs on 10 images (~5 minutes).
To skip it for a faster check:

```bash
SKIP_SMOKE_TRAIN=1 MODEL_SIZE=large USE_DEV_SUBSET=false \
python scripts/prepare_cloud_training.py
```

---

## Step 6 — Launch training

```bash
conda activate satid
MODEL_SIZE=large USE_DEV_SUBSET=false \
python -m training.train_dino \
    --work-dir weights/run_001
```

### What happens

| Epochs | Stage | Behaviour |
|--------|-------|-----------|
| 1–20 | Stage 1 | Backbone frozen; only neck + head train |
| 21–50 | Stage 2 | Backbone unfrozen; lower LR (lr_mult=0.1) |

**After epoch 1**, the script prints estimated total time and cost, then
sleeps 30 seconds.  Press **Ctrl+C during that window** to abort without
incurring further charges.  After the 30-second window, training continues
automatically.

### Monitoring

Checkpoints are saved every 5 epochs to `weights/run_001/`.
The best checkpoint (by COCO bbox mAP) is saved as
`weights/run_001/best_coco_bbox_mAP_epoch_NN.pth`.

Watch the loss live:
```bash
tail -f weights/run_001/*.log
```

### Out-of-memory (OOM) recovery

If you hit a CUDA OOM error:

```bash
# Option 1: reduce image size (trades accuracy for memory)
export IMAGE_SIZE=640
python -m training.train_dino --work-dir weights/run_001

# Option 2: resume from the last checkpoint
python -m training.train_dino \
    --work-dir weights/run_001 \
    --resume weights/run_001/latest.pth
```

The config already uses:
- `batch_size=1` with `accumulative_counts=2` (effective batch = 2)
- `AmpOptimWrapper` (automatic mixed precision, ~30% VRAM savings)
- `with_cp=True` (gradient checkpointing on the backbone)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reduces fragmentation)

---

## Step 7 — Evaluate after training

Run the benchmark on the held-out test split to produce all statistics.

```bash
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/test.json \
    --output results/final_eval.json
```

This produces:
- `results/final_eval.json` — all metrics (precision, recall, F1, mAP@0.5, mAP@0.75, angle error, per-band)
- `results/confusion_matrix.png` — 2×2 confusion matrix plot (TP/FP/FN/TN)
- `eval/results/dino_predictions.json` — per-image prediction dump
- Markdown table printed to stdout

### Target metrics (from StreakMind paper, Swin-L)

| Metric | Target |
|--------|--------|
| Precision | ≥ 94% |
| Recall | ≥ 97% |
| mAP@0.5 | — |

### Cross-ID accuracy (optional)

To also benchmark satellite identification accuracy against GTImages ground truth:

```bash
python -m eval.benchmark \
    --run-pipeline \
    --annotations data/annotations/gtimages.json \
    --output results/gtimages_crossid.json
```

This requires Space-Track credentials (see `agent_docs/assistant_guide.md` for setup).

---

## Step 8 — Send results back

From the training machine, notify the project owner that training is done,
then they will run:

```bash
# On the project owner's Mac:
./scripts/fetch_weights.sh user@<your-machine-ip>
```

This rsync's `weights/run_001/`, `results/`, `eval/results/`, and
`training/logs/` back to their machine.

Alternatively, you can upload `weights/run_001/best_coco_bbox_mAP*.pth`
and `results/` to the shared drive.

---

## Troubleshooting

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

The Swin-L model refuses to load on CPU.  Ensure CUDA is available:
```bash
python -c "import torch; assert torch.cuda.is_available()"
```

### OOM on first batch

Try `IMAGE_SIZE=640` (see Step 6, OOM recovery above).

### Training resumes from wrong epoch

Check `weights/run_001/` for checkpoint files.  Pass the correct checkpoint
to `--resume`:
```bash
ls weights/run_001/*.pth
python -m training.train_dino --work-dir weights/run_001 \
    --resume weights/run_001/epoch_20.pth
```

---

## File layout (after full setup)

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
