# ARGUS Cloud Training Manifest

## Run Identity

- Run name:
- Route: Cloud GPU RTX 4090 / Workstation RTX 5070 Ti / other
- Date started UTC:
- Operator:
- Provider:
- Instance ID:
- Hourly rate:
- Work directory:
- Results directory:

## Code State

- Repository:
- Branch:
- Commit:
- Worktree dirty: yes/no
- Source archive used: yes/no
- Notes on uncommitted changes:

## Dataset Decision

- Dataset used:
- Legacy datasets retained: GTImages=yes/no, SatStreaks=yes/no
- Proprietary/third-party data included: yes/no
- Reason for inclusion/exclusion:
- Train split:
- Val split:
- Test split:

## Input Counts

- Train images:
- Train annotations:
- Val images:
- Val annotations:
- Test images:
- Test annotations:
- SatStreaks image count:
- SatStreaks mask count:
- Empty image count:
- Degenerate boxes removed:

## Input Checksums

Record SHA-256 checksums for:

```text
data/annotations/train.json
data/annotations/val.json
data/annotations/test.json
weights/dinov3_vitl16_lvd1689m.pth
```

Checksum file:

```text
results/cloud_training_prep/input_sha256.txt
```

## Environment

- OS image:
- GPU:
- VRAM:
- NVIDIA driver:
- CUDA:
- Python:
- PyTorch:
- torchvision:
- mmengine:
- mmcv:
- mmdet:
- dinov3 package:

Environment metadata file:

```text
results/4090_dinov3_vitl/environment.txt
```

## Training Configuration

- Backbone: DINOv3 ViT-L/16 LVD-1689M
- Backbone frozen: yes
- Image scale:
- Batch size:
- Max epochs:
- Validation interval:
- Early stopping rule:
- Random seed:
- Normalization:
- Cost guardrail rate:

Environment variables:

```bash
export MODEL_SIZE=dinov3_vitl
export USE_DEV_SUBSET=false
export ARGUS_NORM=autostretch
export DATABASE_URL=sqlite+aiosqlite:///./argus.db
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Training command:

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

## Preflight

- `scripts/prepare_cloud_training.py` passed: yes/no
- Training smoke test passed: yes/no
- First epoch completed: yes/no
- First epoch wall time:
- Estimated total cost:
- OOM encountered: yes/no
- If OOM, final image scale:

## Checkpoint Storage

- Local checkpoint directory:
- Durable storage destination:
- Sync command:
- Best checkpoint:
- Final checkpoint:
- Checksum file:
- Weights manifest:

## Evaluation

- Evaluation checkpoint:
- Evaluation annotations:
- Metrics file:
- Predictions file:
- Precision:
- Recall:
- F1:
- mAP:
- mAP@0.5:
- Short streak recall:
- Medium streak recall:
- Long streak recall:
- Phase 8 targets met: yes/no

## Notes

- Issues encountered:
- Deviations from handoff:
- Follow-up actions:
