# ARGUS Cloud Training Manifest

## Run Identity

- Run name: 4090_dinov3_vitl
- Route: Cloud GPU RTX 4090
- Date started UTC: not started
- Operator: TBD
- Provider: TBD
- Instance ID: TBD
- Hourly rate: TBD
- Work directory: `weights/run_4090_dinov3_vitl`
- Results directory: `results/4090_dinov3_vitl`

## Code State

- Repository: `https://github.com/robertmwolf/Argus`
- Branch: `main`
- Commit: `0cac51d519a8c957cf08c212be518448aa48837e`
- Worktree dirty: yes
- Source archive used: TBD
- Notes on uncommitted changes: local worktree contains modified docs, setup,
  dependency, API, frontend, annotation, and training files. Commit or
  archive the exact working tree before renting cloud time.

## Dataset Decision

- Dataset used: Current ARGUS SatStreaks-derived split
- Legacy datasets retained: GTImages=no, SatStreaks=yes
- Proprietary/third-party data included: proprietary=no; public third-party
  SatStreaks=yes
- Reason for inclusion/exclusion: use the existing committed/current Phase D
  split unless a newer paper-run dataset is selected before rental.
- Train split: `data/annotations/train.json`
- Val split: `data/annotations/val.json`
- Test split: `data/annotations/test.json`

## Input Counts

- Train images: 3023
- Train annotations: 2957
- Val images: 411
- Val annotations: 386
- Test images: 308
- Test annotations: 308
- SatStreaks image count: 3074
- SatStreaks mask count: 3073
- Empty image count: TBD
- Degenerate boxes removed: TBD

## Input Checksums

Checksum file:

```text
results/cloud_training_prep/input_sha256.txt
```

```text
43665a1d8156d0d6572dcaebe81ddc66eb0e699d2e2d86af4d2ef8c8a95060f1  data/annotations/train.json
0faed8cde99c20c71ae16569a7bca17b626463a1b65399447835650571bbc3b4  data/annotations/val.json
4aaad5d953bb99664064d57dd5bec2de2d8ad5f8e57cb8c026aee0190322596c  data/annotations/test.json
8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035  weights/dinov3_vitl16_lvd1689m.pth
```

## Environment

- OS image: TBD
- GPU: target NVIDIA RTX 4090
- VRAM: target 24 GB
- NVIDIA driver: TBD
- CUDA: TBD
- Python: Python 3.11
- PyTorch: TBD on cloud
- torchvision: TBD
- mmengine: 0.10.4 target
- mmcv: 2.1.0 target
- mmdet: 3.3.0 target
- dinov3 package: `git+https://github.com/facebookresearch/dinov3.git`

Environment metadata file:

```text
results/4090_dinov3_vitl/environment.txt
```

## Training Configuration

- Backbone: DINOv3 ViT-L/16 LVD-1689M
- Backbone frozen: yes
- Image scale: 800x800 target on RTX 4090; fall back to 512x512 if OOM
- Batch size: 1
- Max epochs: 50
- Validation interval: 5
- Early stopping rule: stop if val mAP@0.5 has not improved for 5 consecutive
  epochs and is already >= 0.74
- Random seed: TBD
- Normalization: `autostretch`
- Cost guardrail rate: update to actual provider rate before run

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

- `scripts/prepare_cloud_training.py` passed: not yet on cloud
- Training smoke test passed: not yet
- First epoch completed: no
- First epoch wall time: TBD
- Estimated total cost: TBD
- OOM encountered: TBD
- If OOM, final image scale: TBD

## Checkpoint Storage

- Local checkpoint directory: `weights/run_4090_dinov3_vitl`
- Durable storage destination: TBD
- Sync command: TBD
- Best checkpoint: TBD
- Final checkpoint: TBD
- Checksum file: `results/4090_dinov3_vitl/weights_sha256.txt`
- Weights manifest: `results/4090_dinov3_vitl/weights_manifest.json`

## Evaluation

- Evaluation checkpoint: TBD
- Evaluation annotations: `data/annotations/test.json`
- Metrics file: `results/4090_dinov3_vitl/phase8_benchmark.json`
- Predictions file: `results/4090_dinov3_vitl/dino_predictions.json`
- Precision: TBD
- Recall: TBD
- F1: TBD
- mAP: TBD
- mAP@0.5: TBD
- Short streak recall: TBD
- Medium streak recall: TBD
- Long streak recall: TBD
- Phase 8 targets met: TBD

## Notes

- Issues encountered: none yet; cloud training not started.
- Deviations from handoff: none yet.
- Follow-up actions: commit or archive current source state; choose provider;
  fill hourly rate and durable checkpoint storage destination before launch.
