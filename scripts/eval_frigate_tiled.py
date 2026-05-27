"""Tiled inference evaluation on Frigate images.

Runs tiled inference on every image in frigate_streaks_eval.json and reports
COCO mAP + P/R/F1 + per-band recall using the adaptive tiling parameters
appropriate for Frigate's short streaks (20–80 px native).

Default parameters:
  * ``native_tile_size=110``  — 110×110 px source crop
  * ``model_input_size=400``  — resize to 400×400 before inference
  * ``magnification≈3.6×``    — brings 20–80 px streaks to ~70–290 px at model input
  * ``overlap=0.5``           — high overlap to avoid missing short streaks

The full-frame baseline for Frigate is 0.000 mAP@50 (streaks vanish after
full-image downsampling to 400 px).  Any improvement from adaptive tiling
passes the §6 verification criterion.

Usage::

    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_frigate_tiled.py \\
        --checkpoint weights/run_best_400px_nodm/best_coco_bbox_mAP_epoch_15.pth

    # Try a different native tile size:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_frigate_tiled.py \\
        --checkpoint weights/... --native-tile-size 80

Source: adaptive_tiling_plan.md §5 & §6 — files to create / verification plan
Ref: docs/adaptive_tiling_plan.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

SHORT_MAX = 269.0
LONG_MIN  = 800.0

ANN_FILE   = _REPO_ROOT / "data/annotations/frigate_streaks_eval.json"
CONFIG     = _REPO_ROOT / "models/dino/streak_dinov3_vitb_400px.py"


# ---------------------------------------------------------------------------
# Metric helpers (mirrored from eval_brentimages_tiled.py)
# ---------------------------------------------------------------------------

def _bbox_diag(bbox: list[float]) -> float:
    _, _, w, h = bbox
    return math.sqrt(w * w + h * h)


def _band(bbox: list[float]) -> str:
    d = _bbox_diag(bbox)
    if d < SHORT_MAX:
        return "short"
    if d < LONG_MIN:
        return "medium"
    return "long"


def _bbox_iou(b1: list[float], b2: list[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def compute_pr(gt_coco, pred_list, conf_threshold=0.3, iou_threshold=0.5):
    """Compute precision, recall, F1 and per-band recall."""
    gt_by_image: dict[int, list] = {}
    for ann in gt_coco.get("annotations", []):
        gt_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

    preds = [p for p in pred_list if p["score"] >= conf_threshold]
    preds.sort(key=lambda p: p["score"], reverse=True)

    matched = {iid: [False] * len(bboxes) for iid, bboxes in gt_by_image.items()}
    tp = fp = 0
    n_gt = sum(len(v) for v in gt_by_image.values())

    for pred in preds:
        iid = pred["image_id"]
        gt_boxes = gt_by_image.get(iid, [])
        matched_flags = matched.get(iid, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if matched_flags[j]:
                continue
            iou = _bbox_iou(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            matched_flags[best_j] = True
            tp += 1
        else:
            fp += 1

    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-band recall
    band_gt: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    band_tp: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    for ann in gt_coco.get("annotations", []):
        band_gt[_band(ann["bbox"])] += 1

    matched2 = {iid: [False] * len(bboxes) for iid, bboxes in gt_by_image.items()}
    for pred in preds:
        iid = pred["image_id"]
        gt_boxes = gt_by_image.get(iid, [])
        matched_flags = matched2.get(iid, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if matched_flags[j]:
                continue
            iou = _bbox_iou(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            matched_flags[best_j] = True
            b = _band(gt_boxes[best_j])
            band_tp[b] += 1

    return dict(
        tp=tp, fp=fp, fn=fn, n_gt=n_gt,
        precision=precision, recall=recall, f1=f1,
        band_gt=band_gt, band_tp=band_tp,
    )


def compute_coco_map(gt_coco, pred_list):
    """Compute COCO mAP using pycocotools."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import io, contextlib

        gt = COCO()
        gt.dataset = gt_coco
        gt.createIndex()

        dt = gt.loadRes(pred_list) if pred_list else gt.loadRes([])

        ev = COCOeval(gt, dt, "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        stats = ev.stats
        return {
            "mAP":   float(stats[0]),
            "mAP50": float(stats[1]),
            "mAP75": float(stats[2]),
            "mAP_s": float(stats[3]),
            "mAP_m": float(stats[4]),
            "mAP_l": float(stats[5]),
        }
    except Exception as e:
        logger.warning("pycocotools mAP failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=_REPO_ROOT / "weights/run_best_400px_nodm/best_coco_bbox_mAP_epoch_15.pth")
    parser.add_argument("--conf",  type=float, default=0.3)
    parser.add_argument(
        "--native-tile-size",
        type=int,
        default=110,
        help=(
            "Native crop footprint in source-image pixels.  Default 110 gives "
            "~3.6× magnification, bringing 20–80 px Frigate streaks to "
            "~70–290 px at model input."
        ),
    )
    parser.add_argument("--overlap",  type=float, default=0.5)
    parser.add_argument(
        "--interp",
        choices=["bilinear", "lanczos", "cubic", "nearest"],
        default="bilinear",
        help="Interpolation method for tile resize (default: bilinear).",
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Run collinear-fragment stitcher after NMS.",
    )
    parser.add_argument(
        "--stitch-max-gap",
        type=float,
        default=110.0,
        help="Max gap in native pixels for stitching (default 110 = one Frigate tile).",
    )
    parser.add_argument("--ann-file", type=Path, default=ANN_FILE)
    args = parser.parse_args()

    sys.path.insert(0, str(_REPO_ROOT))

    from inference.tiled_pipeline import (
        tile_image, remap_predictions, nms_predictions,
        stitch_collinear_fragments,
        _clip_predictions_to_image, _pipeline_det_to_prediction,
    )
    from inference.pipeline import _load_model, _select_config, _run_inference
    from inference.device import get_device
    import torch
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    model_input_size = 400
    magnification = model_input_size / args.native_tile_size

    logger.info(
        "Frigate tiled eval: native_tile_size=%d, model_input=%d, "
        "magnification=%.2f×, overlap=%.2f",
        args.native_tile_size, model_input_size, magnification, args.overlap,
    )

    if not args.ann_file.exists():
        logger.error("Annotation file not found: %s", args.ann_file)
        sys.exit(1)

    gt_coco = json.loads(args.ann_file.read_text())
    images  = gt_coco["images"]
    logger.info("Evaluating %d images", len(images))

    device = get_device()
    inf_device = torch.device("cpu") if device.type == "mps" else device
    config_path = _select_config("dinov3_vitb_multisource")
    model = _load_model(config_path, args.checkpoint, inf_device)

    import astropy.io.fits as astrofits
    import numpy as np
    import cv2 as _cv2

    def _load_image_array(path: Path) -> np.ndarray:
        """Load a FITS or PNG/JPEG image as an H×W×3 uint8 array.

        PNG/JPEG: loaded directly with OpenCV (RGB).
        FITS:     z-score normalised to uint8, duplicated to 3 channels.
        """
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            bgr = _cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"cv2 could not read {path}")
            return _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)

        # FITS path
        with astrofits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    data = hdu.data.astype(np.float32)
                    break
            else:
                raise ValueError(f"No 2-D image HDU found in {path}")
        mean, std = data.mean(), data.std()
        if std > 0:
            data = (data - mean) / std
        data = np.clip(data, -3.0, 3.0)
        data = ((data + 3.0) / 6.0 * 255).astype(np.uint8)
        return np.stack([data, data, data], axis=2)  # H×W×3

    all_preds = []
    t_start = time.perf_counter()
    resize_to = model_input_size if magnification != 1.0 else None

    for i, img_info in enumerate(images, 1):
        img_path = Path(img_info["file_name"])
        iid = img_info["id"]
        try:
            arr = _load_image_array(img_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s — skipping", img_path.name, e)
            continue

        tiles = list(tile_image(arr, tile_size=args.native_tile_size,
                                overlap=args.overlap, resize_to=resize_to,
                                interp=args.interp))
        preds_this: list[dict] = []
        for tile, x0, y0 in tiles:
            dets = _run_inference(model, tile, image_size=model_input_size,
                                  confidence_threshold=0.05,
                                  model_name="tiled_frigate_eval")
            tile_preds = [_pipeline_det_to_prediction(d) for d in dets]
            preds_this.extend(remap_predictions(tile_preds, x0, y0,
                                                magnification=magnification))

        after_nms = nms_predictions(preds_this, iou_threshold=0.4)
        if args.stitch:
            after_nms = stitch_collinear_fragments(
                after_nms, max_gap_px=args.stitch_max_gap
            )
        final = _clip_predictions_to_image(
            after_nms, width=arr.shape[1], height=arr.shape[0],
        )
        for p in final:
            all_preds.append({
                "image_id":    iid,
                "category_id": 1,
                "bbox":        p["bbox"],
                "score":       p["score"],
            })

        if i % 20 == 0 or i == len(images):
            elapsed = time.perf_counter() - t_start
            logger.info("  %d/%d  %.1fs elapsed  %.2fs/img",
                        i, len(images), elapsed, elapsed / i)

    logger.info("Inference complete — %d total predictions on %d images",
                len(all_preds), len(images))

    # Metrics
    pr   = compute_pr(gt_coco, all_preds, conf_threshold=args.conf)
    coco = compute_coco_map(gt_coco, all_preds)

    mag_str = f"{magnification:.2f}×"
    stitch_label = f", stitch(gap≤{args.stitch_max_gap:.0f}px)" if args.stitch else ""
    print(
        f"\n=== Frigate — ADAPTIVE TILED INFERENCE "
        f"(native_tile={args.native_tile_size}, model_input=400, "
        f"mag={mag_str}, overlap={args.overlap}, interp={args.interp}{stitch_label}) ==="
    )
    if coco:
        print(f"COCO mAP:       {coco['mAP']:.3f}")
        print(f"COCO mAP@50:    {coco['mAP50']:.3f}")
        print(f"COCO mAP@75:    {coco['mAP75']:.3f}")
        print(f"COCO mAP_s:     {coco['mAP_s']:.3f}  (short streaks)")
    print(f"\nP/R @ conf≥{args.conf}, IoU≥0.50:")
    print(f"  Precision: {pr['precision']:.1%}   Recall: {pr['recall']:.1%}   F1: {pr['f1']:.1%}")
    print(f"  TP={pr['tp']}  FP={pr['fp']}  FN={pr['fn']}  GT={pr['n_gt']}")
    print(f"\nPer-band recall (at model input after {mag_str} magnification):")
    for band in ("short", "medium", "long"):
        n = pr['band_gt'][band]
        t = pr['band_tp'][band]
        r = t / n if n else float('nan')
        print(f"  {band:6s}: {r:.1%}  (TP={t}, n={n})")
    print()

    # Verification criterion (§6.1 in adaptive_tiling_plan.md)
    print("=== VERIFICATION (adaptive_tiling_plan.md §6.1) ===")
    print("  Baseline (full-frame):   mAP@50 = 0.000  (streaks vanish at 400 px resize)")
    if coco:
        result = "✅ PASS" if coco['mAP50'] > 0.0 else "❌ FAIL — no improvement over baseline"
        print(f"  Adaptive tiling result:  mAP@50 = {coco['mAP50']:.3f}  {result}")


if __name__ == "__main__":
    main()
