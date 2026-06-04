"""Tiled inference evaluation on BrentImages Night 2.

Runs run_tiled_inference (tile_size=400, overlap=0.5) on every image in
brentimages_20260515_eval.json and reports COCO mAP + P/R/F1 + per-band
recall — identical metrics to evaluate_comprehensive.py but using native-
resolution tiles instead of full-frame downsampling.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_brentimages_tiled.py \
        --checkpoint weights/run_clean_vitb_nodm/best_coco_bbox_mAP_epoch_15.pth
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

ANN_FILE   = _REPO_ROOT / "data/annotations/brentimages_20260515_eval.json"
CONFIG     = _REPO_ROOT / "models/dino/streak_dinov3_vitb_400px.py"


# ---------------------------------------------------------------------------
# Metric helpers (mirrored from evaluate_comprehensive.py)
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
    band_gt: dict[str, int]   = {"short": 0, "medium": 0, "long": 0}
    band_tp: dict[str, int]   = {"short": 0, "medium": 0, "long": 0}
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
        stats = ev.stats  # [mAP, mAP50, mAP75, mAP_s, mAP_m, mAP_l, ...]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=_REPO_ROOT / "weights/run_clean_vitb_nodm/best_coco_bbox_mAP_epoch_15.pth")
    parser.add_argument("--conf",  type=float, default=0.2)
    parser.add_argument("--tile-size", type=int, default=400)
    parser.add_argument("--overlap",   type=float, default=0.5)
    parser.add_argument("--ann-file",  type=Path, default=ANN_FILE)
    parser.add_argument("--model-size", type=str, default="dinov3_vitb_multisource",
                        help="Model size key passed to _select_config (e.g. dinov3_vits_run4)")
    parser.add_argument("--pretiled-ann", type=Path, default=None,
                        help="Pre-tiled annotation JSON with .npy tile paths. When provided, "
                             "skips in-memory FITS tiling and loads tiles directly. The GT "
                             "metrics still use --ann-file (full-image annotations).")
    args = parser.parse_args()

    sys.path.insert(0, str(_REPO_ROOT))

    from inference.tiled_pipeline import run_tiled_inference

    gt_coco = json.loads(args.ann_file.read_text())
    images  = gt_coco["images"]
    logger.info("Evaluating %d images with tiled inference (tile=%d, overlap=%.2f)",
                len(images), args.tile_size, args.overlap)

    # Load model once and reuse across images
    from inference.pipeline import _load_model, _select_config
    from inference.device import get_device
    import torch
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)  # suppress astropy FITS verify noise

    device = get_device()
    inf_device = torch.device("cpu") if device.type == "mps" else device
    config_path = _select_config(args.model_size)
    model = _load_model(config_path, args.checkpoint, inf_device)

    from inference.tiled_pipeline import tile_image, remap_predictions, nms_predictions, _clip_predictions_to_image, _pipeline_det_to_prediction
    from inference.pipeline import _run_inference
    import astropy.io.fits as astrofits
    import numpy as np

    def _load_fits_array(path: Path) -> np.ndarray:
        """Load pixel data only — no WCS, no plate solve, no sidecar lookup."""
        with astrofits.open(path, memmap=False) as hdul:
            # Find the first HDU with 2-D image data
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    data = hdu.data.astype(np.float32)
                    break
            else:
                raise ValueError(f"No 2-D image HDU found in {path}")
        # Z-score normalise to uint8, same as FITSLoader
        mean, std = data.mean(), data.std()
        if std > 0:
            data = (data - mean) / std
        data = np.clip(data, -3.0, 3.0)
        data = ((data + 3.0) / 6.0 * 255).astype(np.uint8)
        return np.stack([data, data, data], axis=2)  # H×W×3

    all_preds = []
    t_start = time.perf_counter()

    import re as _re
    _TILE_RE = _re.compile(r"^(.+?)__tx(\d+)_ty(\d+)_ts(\d+)$")

    def _load_tile_arr(tile_path: Path) -> tuple[np.ndarray, int, int] | None:
        """Load a pre-extracted tile (.npy or virtual FITS path) → (arr_uint8_3ch, x0, y0)."""
        from inference.fits_loader import apply_norm
        m = _TILE_RE.match(tile_path.stem)
        if tile_path.suffix == ".npy":
            raw = np.load(str(tile_path))  # (H, W) float32
            arr = apply_norm(raw, "zscore")  # (H, W, 3) uint8
            # origin encoded in filename: <id>.npy — origin comes from annotation
            return arr, 0, 0  # caller handles origin from annotation
        elif m:
            # virtual tile path — load parent and crop
            real_stem, x0, y0, ts = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            parent = tile_path.parent / (real_stem + tile_path.suffix)
            try:
                raw_arr = _load_fits_array(parent)
            except Exception as e:
                logger.warning("Failed to load parent %s: %s", parent.name, e)
                return None
            return raw_arr[y0:y0+ts, x0:x0+ts], x0, y0
        return None

    if args.pretiled_ann is not None:
        # --- Pre-tiled path: one inference call per tile, no in-memory tiling ---
        logger.info("Pre-tiled mode: loading tiles from %s", args.pretiled_ann)
        tiled_coco = json.loads(args.pretiled_ann.read_text())
        tile_images = tiled_coco["images"]

        # Group tiles by their orig_image_id (stored in annotation or derived from file_name)
        # We need to remap tile-local predictions to full-image coords for NMS per source image.
        # tile_origin is stored as "tile_origin" key if present, else parse from filename.
        from collections import defaultdict as _defaultdict
        tiles_by_orig: dict[int, list] = _defaultdict(list)
        for ti in tile_images:
            orig_id = ti.get("orig_image_id", ti["id"])
            tiles_by_orig[orig_id].append(ti)

        # Build orig_image_id → source image size map from gt_coco
        img_size_map = {img["id"]: img for img in images}

        done_orig = 0
        for orig_iid, tile_list in tiles_by_orig.items():
            preds_this: list[dict] = []
            src_info = img_size_map.get(orig_iid)

            for ti in tile_list:
                tile_path = Path(ti["file_name"])
                origin = ti.get("tile_origin", [0, 0])
                x0, y0 = int(origin[0]), int(origin[1])

                # Load tile
                if tile_path.suffix == ".npy":
                    from inference.fits_loader import apply_norm as _apply_norm
                    raw = np.load(str(tile_path))
                    arr_tile = _apply_norm(raw, "zscore")
                else:
                    result = _load_tile_arr(tile_path)
                    if result is None:
                        continue
                    arr_tile, x0, y0 = result

                try:
                    dets = _run_inference(model, arr_tile, image_size=args.tile_size,
                                          confidence_threshold=0.05,
                                          model_name="tiled_eval")
                    tile_preds = [_pipeline_det_to_prediction(d) for d in dets]
                    preds_this.extend(remap_predictions(tile_preds, x0, y0))
                except Exception as e:
                    logger.warning("Inference failed on tile %s: %s", tile_path.name, e)

            if src_info:
                w, h = src_info.get("width", 99999), src_info.get("height", 99999)
                final = _clip_predictions_to_image(
                    nms_predictions(preds_this, iou_threshold=0.4), width=w, height=h
                )
            else:
                final = nms_predictions(preds_this, iou_threshold=0.4)

            for p in final:
                all_preds.append({
                    "image_id":    orig_iid,
                    "category_id": 1,
                    "bbox":        p["bbox"],
                    "score":       p["score"],
                })

            done_orig += 1
            if done_orig % 20 == 0 or done_orig == len(tiles_by_orig):
                elapsed = time.perf_counter() - t_start
                logger.info("  %d/%d source images  %.1fs elapsed  %.2fs/img",
                            done_orig, len(tiles_by_orig), elapsed, elapsed / done_orig)

    else:
        # --- Original path: load full FITS and tile in memory ---
        for i, img_info in enumerate(images, 1):
            img_path = Path(img_info["file_name"])
            iid = img_info["id"]
            try:
                arr = _load_fits_array(img_path)
            except Exception as e:
                logger.warning("Failed to load %s: %s — skipping", img_path.name, e)
                continue

            tiles = list(tile_image(arr, tile_size=args.tile_size, overlap=args.overlap))
            preds_this: list[dict] = []
            for tile, x0, y0 in tiles:
                dets = _run_inference(model, tile, image_size=args.tile_size,
                                      confidence_threshold=0.05,
                                      model_name="tiled_eval")
                tile_preds = [_pipeline_det_to_prediction(d) for d in dets]
                preds_this.extend(remap_predictions(tile_preds, x0, y0))

            final = _clip_predictions_to_image(
                nms_predictions(preds_this, iou_threshold=0.4),
                width=arr.shape[1], height=arr.shape[0],
            )
            for p in final:
                all_preds.append({
                    "image_id":   iid,
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
    pr = compute_pr(gt_coco, all_preds, conf_threshold=args.conf)
    coco = compute_coco_map(gt_coco, all_preds)

    print("\n=== BrentImages Night 2 — TILED INFERENCE (tile=400, overlap=0.5) ===")
    if coco:
        print(f"COCO mAP:       {coco['mAP']:.3f}")
        print(f"COCO mAP@50:    {coco['mAP50']:.3f}")
        print(f"COCO mAP@75:    {coco['mAP75']:.3f}")
        print(f"COCO mAP_m:     {coco['mAP_m']:.3f}")
        print(f"COCO mAP_l:     {coco['mAP_l']:.3f}")
    print(f"\nP/R @ conf≥{args.conf}, IoU≥0.50:")
    print(f"  Precision: {pr['precision']:.1%}   Recall: {pr['recall']:.1%}   F1: {pr['f1']:.1%}")
    print(f"  TP={pr['tp']}  FP={pr['fp']}  FN={pr['fn']}  GT={pr['n_gt']}")
    print(f"\nPer-band recall:")
    for band in ("short", "medium", "long"):
        n = pr['band_gt'][band]
        t = pr['band_tp'][band]
        r = t / n if n else float('nan')
        print(f"  {band:6s}: {r:.1%}  (TP={t}, n={n})")
    print()

    # Compare with full-frame result
    print("=== COMPARISON: full-frame (original eval) vs tiled ===")
    print("  Full-frame:  mAP@50=0.296  P=47.8%  R=31.9%  medium=14.2%  long=60.8%")
    if coco:
        print(f"  Tiled:       mAP@50={coco['mAP50']:.3f}  P={pr['precision']:.1%}  R={pr['recall']:.1%}  "
              f"medium={pr['band_tp']['medium']/pr['band_gt']['medium']:.1%}  "
              f"long={pr['band_tp']['long']/pr['band_gt']['long']:.1%}")


if __name__ == "__main__":
    main()
