# ARGUS Model Comparison — All Models to Date

Generated: 2026-05-18

Numbers are reported on the **test split** (`test.json` / `dm_merged_test.json`) unless
otherwise noted in the Status column.  mAP values are COCO bbox_mAP metrics from
MMDetection's evaluator.

| Model | Backbone | Dataset | Train images | mAP@0.5 | Precision | Recall | Status |
|---|---|---|---:|---:|---:|---:|---|
| DINO Swin-T (Phase 8 local) | Swin-T | Synthetic dev subset (50 imgs, 256 px) | 50 | 0.716 | 0.667 | 0.733 | Complete — CPU, synthetic only |
| YOLO11n-OBB baseline (Phase 8 local) | YOLO11n | Synthetic dev subset (50 imgs, 256 px) | 50 | 0.036 | 0.632 | 0.400 | Complete — CPU, synthetic only |
| StreakMindYOLO real-only | YOLOv8 OBB | GTImages real only | ~469 | 0.018 | 0.075 | 0.140 | Complete — 2 epochs, underfitted |
| StreakMindYOLO paper_long | YOLOv8 OBB | GTImages + paper synth streaks | ~469 | 0.010 | 0.071 | 0.123 | Complete — 2 epochs, underfitted |
| StreakMindYOLO adapted | YOLOv8 OBB | GTImages + adapted synth streaks | ~469 | 0.005 | 0.041 | 0.070 | Complete — 2 epochs, underfitted |
| DINO Swin-T (Phase E) | Swin-T | GTImages + SatStreaks (full, 3 023 imgs) | 3 023 | 0.190 | — | — | Complete — test.json |
| YOLO11n-OBB full dataset | YOLO11n | GTImages + SatStreaks (tiled, 14 385 tiles) | 3 023 | 0.673 | 0.572 | 0.846 | Complete — tiled val split; not directly comparable to full-image COCO |
| **DINOv3 ViT-B (Phase C², best model)** | **DINOv3 ViT-B frozen** | **GTImages + SatStreaks (full, 3 023 imgs)** | **3 023** | **0.740** | — | — | **Complete — test.json, 4 epochs** |
| DINOv3 ViT-L (Phase D) | DINOv3 ViT-L frozen | GTImages + SatStreaks (full, 3 023 imgs) | 3 023 | — | — | — | Pending — RTX 5070 Ti workstation |
| **DINOv3 ViT-B — GT + DM + SatStreaks** | **DINOv3 ViT-B frozen** | **GTImages + SatStreaks + DarkMatters (3 172 imgs)** | **3 172** | **0.740** | — | — | **Complete — test.json (original, no DM), 4 epochs. mAP@0.75=0.631 (+0.025 vs Phase C²)** |

## Notes

- **Phase 8 local models** (rows 1–2): trained on a 50-image synthetic dev subset at
  256 × 256 px on CPU.  Not representative of full-dataset performance; mAP@0.5
  from `results/phase8_benchmark.json`.

- **StreakMindYOLO rows** (rows 3–5): 2-epoch runs on the GTImages split only.
  Severely underfitted; reported for completeness.  Data from
  `results/streakmind_yolo/comparison.json`.

- **DINO Swin-T / DINOv3 ViT-B Phase E** (rows 6, 8): evaluated on `test.json`
  (308 images, GTImages + SatStreaks).  Data from
  `results/phase_e/phase_e_comparison_test.json` and
  `results/phase_e/dinov3_vitb_test_metrics.json`.

- **YOLO11n-OBB full dataset** (row 7): 15 epochs, evaluated on YOLO tiled val
  split (~2 881 tiles).  Direct mAP comparison to full-image COCO numbers above is
  not valid.  Data from `results/full_yolo_obb/yolo_benchmark.json`.

- **GT + DM + SatStreaks model** (last row): 4 epochs, 14:25→22:44 on Mac M3 MPS.
  Evaluated on **original `test.json`** (308 images, GTImages + SatStreaks only — no
  DarkMatters) for direct comparison with Phase C².  Results: mAP@0.5 = **0.740**
  (matches Phase C²), mAP@0.5:0.95 = **0.594** (+0.014), mAP@0.75 = **0.631** (+0.025).
  Adding 239 DarkMatters images did not hurt the original distribution and improved
  precision at stricter IoU thresholds.  Best checkpoint:
  `weights/run_gt_dm_satstreaks_dinov3_vitb/best_coco_bbox_mAP_epoch_4.pth`.
  Training val curve on `dm_merged_val.json`: mAP@0.5 0.257→0.341→0.392→0.436
  (lower because DarkMatters JPEGs are a harder distribution — different PSF/scale).

## Phase 8 Targets

| Metric | Target | Best achieved |
|---|---|---|
| Precision | ≥ 0.94 | 0.667 (Swin-T dev subset) |
| Recall | ≥ 0.97 | 0.846 (YOLO11n full dataset, tiled) |
| mAP@0.5 | maximise | **0.740** (DINOv3 ViT-B Phase C² and GT+DM+SatStreaks, both on test.json). GT+DM+SatStreaks also improves mAP@0.75 to 0.631 (+0.025). |
