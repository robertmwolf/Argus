# ARGUS Multi-Model Ensemble Benchmark

**Date:** 2026-05-26 21:24
**DINO model:** `dinov3_vitb_multisource`
**YOLO-OBB GTImages weights:** `best.pt`

## Summary

This benchmark evaluates four detectors independently and as a unified ensemble:

| Detector | Role |
|---|---|
| **DINOv3 Multisource** | Primary ML detector — high recall on long streaks, loose axis-aligned boxes |
| **YOLO-OBB GTImages** | Segment detector — tight OBBs, dominant recall on medium-length streaks (single-pass, TILE_SIZE=8192) |
| **ASTRiDE** | Classical detector — many false positives, corroboration signal only |
| **Unified Ensemble v2** | Updated profiles + YOLO geometry preference + per-band weights |

Confidence threshold for P/R evaluation: **0.30**. IoU threshold: **0.50**.
Band thresholds: short < 150 px, 150 ≤ medium < 400 px, long ≥ 400 px.

### BrentImages (50-image sample)

| Metric | DINOv3 Multisource | YOLO-OBB GTImages | ASTRiDE | Unified Ensemble (v2) |
|--------|--------------------|-------------------|---------|-----------------------|
| mAP@0.50 | 0.002 | 0.000 | 0.001 | 0.002 |
| mAP@0.75 | 0.000 | 0.000 | 0.000 | 0.000 |
| P @conf≥0.30 | 2.0% | 0.0% | 0.0% | 2.0% |
| R @conf≥0.30 | 2.3% | 0.0% | 2.3% | 2.3% |
| F1 @conf≥0.30 | 2.2% | 0.0% | 0.1% | 2.2% |
| N preds (raw) | 74 | 8 | 2500 | 2500 |
| N preds @0.30 | 49.000 | 2.000 | 2500.000 | 49.000 |
| Recall short | 0.0% | 0.0% | 0.0% | 0.0% |
| Recall medium | 0.0% | 0.0% | 0.0% | 0.0% |
| Recall long | 2.4% | 0.0% | 2.4% | 2.4% |
